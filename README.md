# GamblerQwen — a hidden-rule observatory

An offline two-armed bandit experiment in which Qwen2.5-1.5B-Instruct chooses between two simulated machines. Their win probabilities may reverse, drift, collapse, reverse twice, or remain stationary. Qwen does **not** see the probabilities, algorithm name, or change point. It sees only actions actually played and their resulting 0/1 rewards.

Main question: **do internal verbalizable concept readouts become informative about a hidden rule change before Qwen makes a decision, and does its behavior adapt?**

The visualizer is [index.html](index.html), an ordinary static webpage that works from your filesystem. No server, JS build tools, backend, real money, or cloud GPU are required.

## Quick start — no Qwen required

From the Windows terminal:

    cd D:\anttiluode\GamblerQwen
    git pull --ff-only
    python3.13 gambler.py --mode demo --scenario reversal --rounds 48 --change-round 24
    start index.html

Demo mode uses a simple window-based policy, **not Qwen**. It tests the game simulation and browser replay. The dashboard labels demonstration data DEMO and shows no invented J-lens readings.

The webpage can also load any saved JSON via **Load run JSON**, including when deployed on GitHub Pages. Local runs write data.js, which the browser loads without file-URL CORS restrictions.

## Actual Qwen on the RTX 3060

Install dependencies once:

    python3.13 -m pip install -r requirements.txt

Keep Qwenimagination at D:\anttiluode\Qwenimagination, since we reuse its Jacobian-lens code (qimag/lens.py). If it is not there, clone it:

    cd D:\anttiluode
    git clone https://github.com/anttiluode/Qwenimagination.git
    cd GamblerQwen

Build a small casino-concept Jacobian lens (one-time CUDA calibration):

    python3.13 build_lens.py --device cuda

This produces cache/gambler_lens.pt. The script filters words to single tokens and computes corpus-averaged output Jacobian directions plus readout-normalization statistics. It uses far fewer calibration prompts than Anthropic's study, so the result is an **exploratory approximation** rather than a full reproduction. This phase performs repeated backward passes; to reduce computation, use the less reliable option --calibration-texts 2.

Then run real Qwen:

    python3.13 gambler.py --mode qwen --scenario reversal --rounds 48 --change-round 24 --seed 17
    start index.html

The run generates:

- **data.js**: self-contained static-page data, intentionally tracked so you can publish the actual run.
- **results/latest.json**: complete source record for local analysis, ignored by Git.

An existing lens can be supplied using --lens path\to\existing_lens.pt, but it must contain the casino concepts you want to track. If no lens exists, Qwen still makes genuine A/B decisions; the webpage truthfully shows that J-lens readouts are missing.

Default layers are 12, 20 and 26 of Qwen's 28 layers. Scores are standardized directional readouts at the **last input token**. A reading such as changed +1.3 is **not** a probability, internal confession, or evidence of subjective feelings.

## Hidden algorithms and controls

    python3.13 gambler.py --mode qwen --scenario stationary --seed 17 --json results/stationary17.json
    python3.13 gambler.py --mode qwen --scenario rugpull --seed 17 --json results/rugpull17.json
    python3.13 gambler.py --mode qwen --scenario drift --seed 17 --json results/drift17.json
    python3.13 gambler.py --mode qwen --scenario double --seed 17 --json results/double17.json

The stationary scenario is the critical **false-alarm control**: normal unlucky streaks can trigger change-sounding concepts. Repeat across several seeds rather than trusting one plot.

For controlled action histories, Qwen can propose actions while the experiment always executes alternating A/B choices:

    python3.13 gambler.py --mode qwen --scenario reversal --execute alternate --seed 17 --json results/reversal_alternate17.json

The recorded proposed and action fields are intentionally distinct.

Change-round is **zero-based on the command line**: --change-round 24 first changes odds on the displayed round 25.

## What would count as success?

1. **Behavior:** Qwen adapts toward the profitable arm after reversal; compare rewards and post-change arm selection against random, alternation, win-stay-lose-switch and a sliding-window policy on the same random events.
2. **Internal signal:** a predeclared, change-associated J-lens signal shifts after real changes more reliably than during stationary unlucky streaks.
3. **Generalization:** repeat across unseen seeds, change directions, switch times, and payout schedules.
4. **Later causal test:** intervene on a putative change-associated representation and test whether its presence causally alters actions without simply damaging the model. Readouts alone only establish correlations.
5. **No persistent weight learning:** Qwen's weights are frozen. Success would show **in-context regime inference**, not persistent synaptic learning or consciousness.

The game pre-generates independent random outcomes for both machines and every round; different policies therefore encounter the same latent random table. Each agent sees only rewards from the arm actually played. It is a scientific toy environment, not a real-money casino strategy.

## Paper connection

Inspired by Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models* (Anthropic, arXiv:2607.15495, July 2026), especially the **bandit example in Figure 14** and Jacobian lens in Section 2. The paper's bandit example explicitly supplies positive/negative feedback to elicit repeat/switch representations. This repository instead tests inference from **unannounced changes in reward probabilities**.

We reuse [Qwenimagination's calibrated Jacobian-lens code](https://github.com/anttiluode/Qwenimagination/blob/main/qimag/lens.py), not a plain logit lens. A limited concept vocabulary does not reconstruct an entire mind or prove what algorithm the model uses.

Run CPU-only tests:

    python3.13 -m unittest discover -s tests -v

## Published recordings and website

The repository already contains **two real CUDA Qwen runs**:
- [Original free-choice run](results/latest.json) — Qwen plays its own proposals; 24 rewards / 48.
- [Forced alternating-observation run](results/reversal_alternate17.json) — Qwen proposes, A/B alternation is played; 21 rewards / 48 on the *forced schedule*.

[The analysis](FINDINGS.md) explains why Qwen preferring B in both runs is not evidence that it detects an unannounced reversal. The recorded game, readouts, and comparisons are displayed in [index.html](index.html) with a **Run** selector. The tracked [data.js](data.js) contains both actual recordings so static hosting works without a backend, local Python server, or runtime fetch.

To regenerate the website's bundled recordings, from the repository root:

    python3.13 publish_results.py
    git add data.js
    git commit -m "Update published Qwen replays"
    git push

To include another checked-in JSON result, pass the full desired list (the order selects the default recording):

    python3.13 publish_results.py results/newrun.json results/reversal_alternate17.json results/latest.json

Note: running gambler.py writes data.js with the **single newly recorded experiment** for immediate local viewing. Before committing a multi-run site update, use publish_results.py to rebuild the multi-run bundle.

GitHub Pages can serve the static index and bundled data at https://anttiluode.github.io/GamblerQwen/ if enabled for **Settings → Pages → Deploy from branch → main / root**. GitHub Actions tests the simulator and data fidelity, but a passing workflow does not itself mean that Pages is enabled.
