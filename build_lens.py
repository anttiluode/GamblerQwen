"""Build a small, calibrated *Jacobian* lens for GamblerQwen.

Reuses the tested code from the neighbouring Qwenimagination repository.
Do not substitute this with a final-layer logit lens and call it a J-lens.

Unlike the Anthropic paper's 1000-prompt calibration, the default corpus is
small for RTX 3060 practicality. This is an exploratory approximation.
"""
import argparse
import sys
from pathlib import Path

WORDS = (
    "change", "changed", "different", "switch", "repeat", "continue",
    "same", "random", "pattern", "luck", "risk", "loss", "losing",
    "winning", "unfair", "trap", "predict", "uncertain", "surprise"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path("../Qwenimagination"),
                    help="Checkout containing qimag/lens.py and qimag/model.py")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("cache/gambler_lens.pt"))
    ap.add_argument("--calibration-texts", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        ap.error(f"{args.out} already exists; use --force to rebuild")
    repo = args.repo.resolve()
    if not (repo / "qimag" / "lens.py").is_file():
        ap.error(f"Cannot find Qwenimagination lens: {repo}/qimag/lens.py")
    sys.path.insert(0, str(repo))
    import torch
    from qimag.lens import Lens, compute_jlens, reading_stats
    from qimag.model import load
    from qimag.items import CALIBRATION

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    lm = load(args.model, args.device, dtype=dtype)
    filtered = [(w, lm.single_token(w)) for w in WORDS]
    accepted = [(w, token) for w, token in filtered if token is not None]
    skipped = [w for w, token in filtered if token is None]
    if not accepted:
        raise SystemExit("No concepts tokenize as one token in this model.")
    texts = CALIBRATION[:args.calibration_texts]
    print("One-token targets:", [w for w, _ in accepted])
    print("Skipped (multi-token):", skipped)
    print("Calibration paragraphs:", len(texts))
    print("Computing corpus-averaged Jacobian directions; this uses backward passes.")
    J = compute_jlens(lm, [token for _, token in accepted], texts,
                      batch=args.batch, log=lambda s: print(s, flush=True))
    with torch.no_grad():
        mu, sd, hmu = reading_stats(lm, J, texts)
    lens = Lens(J, mu, sd, hmu, [w for w, _ in accepted])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lens.save(args.out)
    print(f"Saved actual (small-corpus) J-lens to {args.out}")


if __name__ == "__main__":
    main()
