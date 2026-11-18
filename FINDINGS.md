# Experiment 001: two actual Qwen runs, one hidden reversal

Data inspected November 18, 2026:
- [Free-choice Qwen](results/latest.json), seed 17, 48 rounds
- [Forced alternating observations](results/reversal_alternate17.json), seed 17, 48 rounds

Both use Qwen/Qwen2.5-1.5B-Instruct, the same pre-generated reward table, an actual (small-corpus) Jacobian lens, and a hidden probability change at **displayed round 25**. P(win | A/B) changes from 0.76 / 0.24 to 0.24 / 0.76. Qwen is never told the rates or the change round.

**Important timing:** Before proposing an action on round 25, Qwen has seen outcomes only through round 24. The first new-regime outcome can affect the prompt for **round 26**.

## Behavioral results

| Measurement | Qwen free-choice | Qwen proposes, A/B alternation executed |
|---|---:|---:|
| Qwen proposes A / B | 3 / 45 | 1 / 47 |
| Proposed optimal arm, rounds 1–24 | 3 / 24 | 1 / 24 |
| Proposed optimal arm, rounds 25–48 | 24 / 24 | 24 / 24 |
| Actually executed A / B | 3 / 45 | 24 / 24 |
| Recorded rewards, rounds 1–24 | 5 / 24 | 10 / 24 |
| Recorded rewards, rounds 25–48 | 19 / 24 | 11 / 24 |
| Recorded reward total | 24 / 48 | 21 / 48 |

**The 21 wins in the alternating run belong to the *forced execution policy*, not to Qwen's proposed actions.** Those rewards are not a fair direct comparison of Qwen policy performance against 24 wins in free choice.

In the first run Qwen chooses B on **39 straight rounds (10–48)**, losing on rounds 14–23 ten times straight. After the reversal it wins much more often because B is now the favorable machine, not because it switched.

In the alternating-observation run, the evidence is clearer: before the change, it records **8 A wins in 12 A trials** and **2 B wins in 12 B trials**. Nevertheless Qwen chooses B on **23 of 24** proposals in that period. The only A proposal is at round 3. After the reversal B becomes genuinely better (10 B wins / 12 B trials; 1 A win / 12 A trials), yet Qwen's proposed policy never visibly changes.

This makes a frozen **A/B answer-format bias**, weak numerical inference, or an inability to use reward evidence in the choice head more plausible explanations than successful adaptive sensing. The data do not distinguish these possibilities yet.

## What the J-lens shows

The experiment sampled verbalizable one-token concept directions at layers 12, 20, 26. These z-scored scores are not probabilities, reported intentions, or a complete account of computation.

| Readout | Free pre → post | Alternating pre → post |
|---|---|---|
| L20 "changed" | +0.0385 → +0.0475 | +0.0369 → +0.0405 |
| L26 "changed" | −1.2344 → −1.1512 | −1.2240 → −1.1653 |
| L26 "switch" | −0.7332 → −0.6161 | −0.7324 → −0.6889 |
| L26 composite change-vs-stay signal | −0.0996 → −0.1300 | −0.1415 → −0.1241 |

The composite even changes *opposite directions* on the two histories across full pre/post halves. There is a brief rise near round 25 in both, but **a rise at round 25 cannot be caused by a new-regime outcome**, since Qwen has not seen one yet. The trends may reflect accumulated reward history, prompt length, phrasing, or other context. No stationary/no-change run or matched fake reversal has been recorded for this experimental pair, so a true-positive change detector and its false-positive rate cannot be established.

## Conclusion

The J-lens recording and replay work. The **change-point sensing hypothesis is unsupported on these two runs**. The new forced-alternation control shows that Qwen keeps proposing B even after frequent, informative evidence favoring A.

The next falsifiable diagnostic is not a bigger network. Before adding persistent memory:
1. Run stationary-world and alternate change-time controls on the **same observation schedule**.
2. Counterbalance names A/B (and positional order / single-token scoring): a strong B prior can masquerade as a policy.
3. Force evaluation of the **estimated probability of reward for each arm**, not only one-letter choice, and test whether those estimates match evidence.
4. Match context length and histories when reading concept scores; choose threshold / alarm rule before viewing test seeds.
5. After behavioral calibration, test activation interventions with negative controls. Readout correlations by themselves cannot prove the representation caused the decision.

This is a useful *negative* result: access to history and recognizable internal concept directions is not enough to demonstrate that an LLM notices a changing reward generator.
