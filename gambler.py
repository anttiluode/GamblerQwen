"""GamblerQwen: hidden-regime two-armed bandit and optional Qwen J-lens observer.

The program never tells Qwen true win probabilities or change points.
No real money or external gambling service is involved.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
WATCH = (
    "change", "changed", "different", "switch", "repeat", "continue",
    "same", "random", "pattern", "luck", "risk", "loss", "losing",
    "winning", "unfair", "trap", "predict", "uncertain", "surprise"
)
CHANGE_WORDS = {"change", "changed", "different", "switch", "surprise"}
STAY_WORDS = {"same", "repeat", "continue"}


def odds(scenario: str, t: int, n: int, switch_at: int) -> tuple[float, float]:
    """P(win | A), P(win | B) at zero-based round t."""
    if scenario == "stationary":
        return 0.72, 0.28
    if scenario == "reversal":
        return (0.76, 0.24) if t < switch_at else (0.24, 0.76)
    if scenario == "rugpull":
        return (0.80, 0.45) if t < switch_at else (0.08, 0.66)
    if scenario == "drift":
        x = max(0.0, min(1.0, (t - switch_at + 5) / 10))
        return 0.76 - 0.52 * x, 0.24 + 0.52 * x
    if scenario == "double":
        if t < n // 3:
            return 0.75, 0.25
        if t < 2 * n // 3:
            return 0.20, 0.80
        return 0.75, 0.25
    raise ValueError("Unknown scenario: " + scenario)


def make_world(scenario: str, n: int, switch_at: int, seed: int) -> list[dict]:
    """Create all random events before the policy acts; same events for every policy."""
    if n < 4 or not 1 <= switch_at < n:
        raise ValueError("Need rounds >= 4 and 1 <= change-round < rounds")
    rng = random.Random(seed)
    world = []
    for t in range(n):
        pa, pb = odds(scenario, t, n, switch_at)
        ua, ub = rng.random(), rng.random()
        world.append({
            "round": t + 1, "p_a": pa, "p_b": pb,
            "reward_a": int(ua < pa), "reward_b": int(ub < pb),
            "regime": "pre" if t < switch_at else "post",
        })
    return world


def choose_baseline(name: str, history: list[dict], rng: random.Random) -> str:
    if name == "random":
        return rng.choice(("A", "B"))
    if name == "alternate":
        return "A" if len(history) % 2 == 0 else "B"
    if name == "win-stay-lose-switch":
        if not history:
            return "A"
        last = history[-1]
        if last["reward"]:
            return last["action"]
        return "B" if last["action"] == "A" else "A"
    if name == "window":
        if len(history) < 2 or rng.random() < 0.14:
            return rng.choice(("A", "B"))
        def rate(arm):
            recent = [r["reward"] for r in history[-12:] if r["action"] == arm]
            return (sum(recent) + 1.0) / (len(recent) + 2.0)
        return "A" if rate("A") >= rate("B") else "B"
    raise ValueError(name)


def history_text(history: list[dict]) -> str:
    if not history:
        return "(no rounds yet)"
    return "\n".join(
        f'{r["round"]:02d} {r["action"]} {r["reward"]}'
        for r in history
    )


def choice_prompt(history: list[dict]) -> str:
    # No labels such as "rigged" or "switch" in the prompt: lens labels should
    # not be literal echoes of a suggested explanation.
    return (
        "You are playing a two-machine learning game. Each time you choose A or B, "
        "you receive 1 point or 0 points. You only see the result of the machine "
        "you chose. Maximize your total points. Decide from the evidence.\n"
        "History columns are round, machine, result:\n"
        + history_text(history)
        + "\nChoose the next machine. Answer with just A or B."
    )


class QwenPolicy:
    """Constrained next-token A/B choice; optional *actual* calibrated J-lens."""

    def __init__(self, model_name: str, device: str, lens_path: str | None,
                 read_layers: tuple[int, ...]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.read_layers = read_layers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model = model.to(device).eval()
        self.layers = self.model.model.layers
        if max(read_layers) >= len(self.layers):
            raise ValueError(f"Model has only {len(self.layers)} layers")
        self.tokens = {}
        for arm in ("A", "B"):
            candidates = [self.tokenizer.encode(s, add_special_tokens=False)
                          for s in (" " + arm, arm)]
            valid = [ids[0] for ids in candidates if len(ids) == 1]
            if not valid:
                raise ValueError(f"Cannot score arm {arm} as a single token")
            self.tokens[arm] = sorted(set(valid))
        self.lens = None
        self.lens_source = "none"
        if lens_path:
            # The exact qimag.Lens implementation is reused, not renamed a
            # logit lens. Ensure sibling Qwenimagination is checked out.
            import sys
            sibling = Path(__file__).resolve().parent.parent / "Qwenimagination"
            if sibling.is_dir():
                sys.path.insert(0, str(sibling))
            from qimag.lens import Lens
            self.lens = Lens.load(lens_path, device)
            self.lens_source = str(lens_path)
        self.concepts = {}
        if self.lens:
            self.concepts = {
                word.strip().casefold(): i
                for i, word in enumerate(self.lens.words)
            }
            present = sorted(set(WATCH) & set(self.concepts))
            print("J-lens watch words:", ", ".join(present))
            if not present:
                print("WARNING: no watched concepts in lens. Run build_lens.py.")

    def pick(self, history: list[dict]) -> dict:
        torch = self.torch
        prompt = choice_prompt(history)
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        saved = {}
        hooks = []
        if self.lens:
            for layer in self.read_layers:
                def hook(_module, _inputs, output, layer=layer):
                    saved[layer] = output[0] if isinstance(output, tuple) else output
                hooks.append(self.layers[layer].register_forward_hook(hook))
        try:
            with torch.inference_mode():
                logits = self.model(input_ids=ids, use_cache=False).logits[0, -1].float()
                scores = {
                    arm: float(torch.logsumexp(logits[indices], 0))
                    for arm, indices in self.tokens.items()
                }
                decision = "A" if scores["A"] >= scores["B"] else "B"
                # The logit-based choice is constrained; it is not a verbal
                # chain of thought, nor an explicit change-point report.
                reads = {}
                for layer, activations in saved.items():
                    z = self.lens.read(activations[0, -1].float(), layer).float()
                    reads[str(layer)] = {
                        word: round(float(z[i]), 4)
                        for word, i in self.concepts.items() if word in WATCH
                    }
        finally:
            for h in hooks:
                h.remove()
        return {"choice": decision, "logits": scores, "lens": reads}


def summarize_signal(reads: dict, final_layer: int) -> float | None:
    row = reads.get(str(final_layer), {})
    higher = [row[w] for w in CHANGE_WORDS if w in row]
    lower = [row[w] for w in STAY_WORDS if w in row]
    if not higher or not lower:
        return None
    return round(sum(higher) / len(higher) - sum(lower) / len(lower), 4)


def simulate(world: list[dict], name: str, seed: int,
             qwen: QwenPolicy | None = None, execute: str = "qwen",
             read_layer: int = 26) -> list[dict]:
    rng = random.Random(seed + 9001)
    history: list[dict] = []
    for w in world:
        if qwen is not None:
            q = qwen.pick(history)
            proposed = q["choice"]
            action = proposed if execute == "qwen" else choose_baseline("alternate", history, rng)
        else:
            q = {"logits": {}, "lens": {}}
            proposed = None
            action = choose_baseline(name, history, rng)
        reward = w["reward_a"] if action == "A" else w["reward_b"]
        entry = {k: w[k] for k in ("round", "p_a", "p_b", "regime")}
        entry.update({
            "action": action, "proposed": proposed, "reward": reward,
            "total": (history[-1]["total"] if history else 0) + reward,
            "lens": q["lens"], "signal": summarize_signal(q["lens"], read_layer),
            "logits": q["logits"],
        })
        history.append(entry)
        print(f'{w["round"]:02d}: {action} => {reward}  total {entry["total"]}'
              + (f'  Qwen wanted {proposed}' if proposed and action != proposed else "")
              + (f'  J-change {entry["signal"]:+.2f}' if entry["signal"] is not None else ""),
              flush=True)
    return history


def report_metrics(rows: list[dict], change_at: int) -> dict:
    post = rows[change_at:]
    best = sum(int(r["action"] == ("A" if r["p_a"] >= r["p_b"] else "B"))
               for r in post)
    return {
        "wins": sum(r["reward"] for r in rows),
        "rounds": len(rows),
        "post_change_optimal_arm_fraction": round(best / len(post), 4) if post else None
    }


def run(args: argparse.Namespace) -> dict:
    world = make_world(args.scenario, args.rounds, args.change_round, args.seed)
    lens_path = Path(args.lens) if args.lens else Path("cache/gambler_lens.pt")
    qwen = None
    if args.mode == "qwen":
        path = str(lens_path) if lens_path.is_file() else None
        if path is None:
            print("WARNING: J-lens unavailable. Qwen choices will be real, but readouts empty.")
            print("Run build_lens.py once, or pass --lens path/to/lens.pt.")
        qwen = QwenPolicy(args.model, args.device, path, tuple(args.layers))
    rows = simulate(world, "window" if args.mode == "demo" else args.mode, args.seed, qwen, args.execute, args.layers[-1])
    baselines = {}
    for name in ("random", "alternate", "win-stay-lose-switch", "window"):
        rr = simulate_quiet(world, name, args.seed)
        baselines[name] = report_metrics(rr, args.change_round)
    return {
        "meta": {
            "title": "GamblerQwen", "source": args.mode, "scenario": args.scenario,
            "model": args.model if qwen else None,
            "lens_source": qwen.lens_source if qwen else "none",
            "read_layers": list(args.layers) if qwen else [],
            "seed": args.seed, "rounds": args.rounds,
            "change_round": args.change_round + 1,
            "execution_policy": args.execute if qwen else args.mode,
            "note": ("No Qwen used: synthetic demonstration only." if not qwen else
                     "Actual Qwen logits and optional calibrated J-lens concept scores; no chain-of-thought claims.")
        },
        "rounds": rows,
        "baselines": baselines,
        "metrics": report_metrics(rows, args.change_round),
    }


def simulate_quiet(world: list[dict], name: str, seed: int) -> list[dict]:
    rng = random.Random(seed + 9001)
    history = []
    for w in world:
        a = choose_baseline(name, history, rng)
        reward = w["reward_a"] if a == "A" else w["reward_b"]
        history.append({"action": a, "reward": reward, "p_a": w["p_a"], "p_b": w["p_b"]})
    return history


def save(report: dict, dest: Path, json_path: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # Escape HTML closing tags so data can be safely loaded as a JS assignment.
    dest.write_text("window.GAMBLER_DATA=" + serialized.replace("<", "\\u003c") + ";\n",
                    encoding="utf-8")
    print(f"Saved: {json_path}\nOpen index.html (it loads {dest.name}).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["demo", "qwen"], default="demo")
    ap.add_argument("--scenario", choices=["stationary", "reversal", "rugpull",
                                           "drift", "double"], default="reversal")
    ap.add_argument("--rounds", type=int, default=48)
    ap.add_argument("--change-round", type=int, default=24,
                    help="Zero-based index at which the first changed odds apply")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lens", help="Path to existing calibrated qimag lens file")
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 20, 26])
    ap.add_argument("--execute", choices=["qwen", "alternate"], default="qwen",
                    help="Alternate executes fixed actions but still records Qwen's choice")
    ap.add_argument("--output", type=Path, default=Path("data.js"))
    ap.add_argument("--json", type=Path, default=Path("results/latest.json"))
    args = ap.parse_args()
    if sorted(set(args.layers)) != args.layers:
        ap.error("--layers must be strictly ascending and distinct")
    report = run(args)
    save(report, args.output, args.json)
    print("Performance:", report["metrics"], "Baselines:", report["baselines"])


if __name__ == "__main__":
    main()
