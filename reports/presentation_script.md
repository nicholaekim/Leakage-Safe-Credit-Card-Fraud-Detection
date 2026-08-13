# Presentation Script — Fraud Detection, Measured in CAD

**Deck:** `final video take presentation.pptx` (20 slides) · **Team:** Nicholas Kim, Balaji S Kumar, Vivekkumar Chaudhari

Every paragraph below is the verbatim Presenter View note for that slide — the deck and this file are the same script in two places. The demo section (slides 6–8) includes the exact commands to run.

## Slide 1 — Title — Fraud Detection, Measured in CAD

Welcome. Our project is credit-card fraud detection on the standard ULB dataset - but our angle is honesty. Public projects on this exact dataset routinely report 99%+ scores; our forensics experiment shows how the common evaluation shortcuts produce exactly that kind of inflation. We built a version where those shortcuts cannot happen by construction, measured performance in dollars instead of abstract scores, and then ran a forensic audit that measures exactly how much each common shortcut inflates the results. The two numbers on the right are the whole story: 0.99 if you evaluate the way most notebooks do, 0.73 if you evaluate honestly. Everything is reproducible: one command per experiment.

## Slide 2 — The Team

On camera, each member in turn: Hi, I'm [full name], and I worked on [area]. Keep it to one line each - the detailed contribution statements come at the end of the talk. This slide satisfies the submission requirement that every member's face is shown and their name is clearly stated.

## Slide 3 — The problem: a model that does nothing is 99.83% accurate

The core difficulty: fraud is one transaction in 579. That breaks accuracy as a metric - a model that never flags anything is 99.83% accurate and completely useless. So we pick scoreboards that mean something at this imbalance: PR-AUC as the primary ranking metric, precision and recall at an explicit operating point, and - our addition - dollars. The rarity has a second consequence: everyone reaches for SMOTE, which synthesizes fake fraud examples. Used correctly it is fine. Used the way most tutorials use it, it silently corrupts the test set - that is the centerpiece of our forensics experiment.

## Slide 4 — Leakage: how the standard workflow accidentally cheats

Here is the single most important idea. The common workflow applies SMOTE to the entire dataset and then splits. Two corruptions follow: synthetic near-copies of test frauds end up in training, and the test set itself becomes half fraud instead of 0.17 percent. It is like studying with the answer key. Our pipeline makes this impossible by construction: split first, then every learning step lives inside a sealed pipeline only ever fit on training folds. The decision threshold is tuned on validation, and the test set is touched exactly once.

## Slide 5 — Design: five stages, three guardrails, one verification loop

This is the whole system on one slide. Data preparation loads, deduplicates and splits first. Training happens inside the sealed pipeline. Thresholds and calibration are tuned on validation only. Evaluation touches the test set exactly once and reports both PR-AUC and dollars saved. The green badges are our guardrails - train folds only, validation only, test touched once. Above it all sits the verification loop: every experiment was adversarially reviewed, fixed, and re-run before we trusted it. And the bottom loop is repetition: five seeds, four imbalance strategies, five models, so every number carries error bars. Keep this picture in mind - every experiment that follows lives somewhere on it.

## Slide 6 — Live demo: the quick benchmark, end to end

**Demo section.** Commands, in order:

```bash
python3 -m experiments.run_benchmark --quick
cat -n src/pipeline.py
ls -la results/tables/
```

What the quick run prints (narrate along):

```
Integrity: {n_rows: 284807, n_fraud: 492, fraud_rate: 0.0017, n_exact_dups: 1081, n_feature_dups: 9144}
Dropped 9144 duplicate rows; 275663 remain.
```

This slide stays up while the screen recording plays, or the presenter switches to a live terminal. The full word-for-word script is the demo section of reports/presentation_script.md: tour the repo in one sentence, open pipeline.py and point at the sealed pipeline - the no-cheating guarantee - then run the quick benchmark and narrate while it prints: the integrity check, the duplicate removal, and finally the two tables where the leaky pipeline reports essentially perfect scores and the honest one tells the truth. Close on the results folder: every figure and table in the report regenerates from these CSVs.

## Slide 7 — Demo output 1 of 2: the finished run

*(no spoken notes — the slide is on screen while the demo output is walked through)*

## Slide 8 — Demo output 2 of 2: the code and the receipts

*(no spoken notes — the slide is on screen while the demo output is walked through)*

## Slide 9 — Experiment 1: the money benchmark

Instead of only abstract scores, we price everything. An alert costs 5 dollars to investigate; a missed fraud costs its own amount; savings is the fraction of do-nothing losses a policy prevents, net of fees. Three anchors: doing nothing is zero, blocking everything is minus 24.7 - clipped off this chart - and a perfect oracle tops out at 0.97, because 45 percent of frauds are worth less than the fee. Our best honest configurations - a plain XGBoost just ahead of the class-weighted random forest showcase, all within noise of each other - prevent about 70 percent of losses over five seeds, roughly 7,900 dollars per test slice of 55,000 transactions.

## Slide 10 — Experiment 1: the scoreboard and the bank account disagree

The punchline of the money benchmark. These two models have statistically identical ranking quality - PR-AUC 0.828 versus 0.832. But in dollars, the point estimates split 0.645 versus 0.701, with the SMOTE forest letting about 9 percent more fraud value through - it catches many frauds, but the cheap ones. Whether that money gap is real is exactly what Experiment 5 tests - spoiler: it is within noise, and that is the point. Classification metrics count frauds; they cannot see that one missed 3,000-dollar fraud outweighs a hundred missed 2-dollar frauds. And note the savings leader is a third model entirely, a plain XGBoost. Different scoreboards, different champions - which is exactly why the economic scoreboard has to be explicit.

## Slide 11 — Experiment 3: calibration, priced

Calibration usually shows up as an abstract chart. We price it. The decision rule - flag when probability times amount exceeds the 5-dollar fee - needs no tuning but trusts the probabilities literally. Class-weighted logistic regression outputs probabilities inflated about 40-fold. Fed raw into the rule, the policy loses about 3.4 times the total fraud value. One line of Platt scaling, fit only on validation, repairs the identical model to plus 0.69 - a 44,000-dollar swing per test slice. The table is the real finding: badly miscalibrated models gain tens of thousands, already-calibrated models gain nothing. Calibration has a price tag proportional to the disease.

## Slide 12 — Experiment 4: leakage forensics — one mistake does the damage

Rather than one leaky-versus-safe comparison, we made each classic mistake an independent on-off switch and ran all sixteen combinations over five seeds. Every workflow is scored twice: on its own, possibly poisoned, test split - the number a flawed notebook would publish - and on a pristine holdout carved out before any sin could touch it. The attribution is unambiguous: SMOTE-before-split alone inflates the reported PR-AUC by 0.26 - essentially the entire inflation - while the true score moves by four thousandths. The scaler leak everyone worries about: zero. Duplicates: noise. Threshold-on-test: zero on PR-AUC by mathematical construction - our built-in consistency check. And the honest workflow's reported number matches its clean-holdout truth to three thousandths: honesty verified, not assumed.

## Slide 13 — Experiment 4: two separate harms, separately measured

The forensics separated two distinct harms of the same mistake. Harm one is the lie: the reported score is fiction because the test set itself is poisoned. Harm two is worse: the decision threshold was tuned on data that is half fraud, so deployed at the real prevalence of one in 579 it drowns in false positives. Reported F1 0.95; deployed F1 0.07 - nearly fourteen times overstated. SMOTE-before-split does not just make you look better than you are - it makes you ship a worse product while believing it is excellent.

## Slide 14 — Experiment 5: which wins are real? None survived — not even ours

With only 95 frauds in the test set, how much of any ranking is luck? The paired bootstrap answers this: resample the same test rows a thousand times for all models simultaneously, so sample luck cancels and the interval lands on the difference. Result one: every pairwise PR-AUC win among our top models is statistically indistinguishable - the leaderboard is noise at this scale, and the PR-AUC leader even flips between seed sets. Result two: we pointed the same weapon at our own headline. On the discovery seeds the money gap merely looked suggestive - two of three intervals excluded zero. So we pre-registered the comparison and re-tested once on fresh seeds three and four, where both confidence intervals cross zero - the gap did not survive, and we report it as unconfirmed. Discovery, pre-register, and an honest failed confirmation - that is the protocol doing its job.

## Slide 15 — Experiments 7–8: explainability + error economics (plus the E6 and E8 asides)

We opened the box three ways. Globally, SHAP and permutation importance agree on the drivers - components like V10, V12 and V14 - with the honest caveat that these are anonymized components, not human causes. Locally, two case studies: the most confident false alarm was a two-dollar transaction - even a correct alert there loses money, which is why our per-transaction rule exists. And the most expensive miss, 829 dollars, scored zero point zero zero zero - the strongest fraud signals pointed the wrong way; a blind spot, not a near-miss. Economically, misses are wildly unequal: five tiny misses cost under ten dollars, eight big ones cost 2,918. Two more honest results: forward-in-time evaluation drops savings from 0.72 to 0.62, and a proper leakage-safe hyperparameter search changed nothing beyond noise - at 280 training frauds that is the expected truth, and we say so.

## Slide 16 — Engineering rigor: build → attack → fix → verify

A methodology slide about ourselves. Every experiment went through the same loop: build it leakage-safe, subject it to independent adversarial review, fix what is confirmed, verify the fixes, and only then run for real. This caught real flaws in our own work: our supposedly clean holdout was contaminated - the dataset contains 9,144 duplicates in the feature space the model sees, eight times the count everyone removes, itself a novel data-quality finding. A headline number was quietly cherry-picking the better calibrator on test data. One of our hypotheses was mathematically incapable of failing, so we reframed it as a consistency check. The discipline we preach in the project, we applied to the project.

## Slide 17 — Submission statement: contributions, status, assumptions

Each member on camera, one or two sentences: I'm [name], I was responsible for [area] - I wrote [files], ran [experiments], and contributed [report sections]. Then one member closes with the status statement: everything demonstrated is fully functional and reproducible - all eight experiments run end-to-end from a fresh clone with one command each. Nothing we presented is partially functional or mocked. Two extensions were scoped but deliberately not built - a second dataset and conformal prediction - and the report explains why. Our key assumptions: the 5-dollar-per-alert fee, dollar figures relabeling the native European amounts one-to-one, and post-hoc comparisons labeled exploratory - including our one pre-registered money gap, which failed its fresh-seed confirmation and is reported as unconfirmed. And the biggest lesson learned: evaluation is harder than modeling - one misplaced preprocessing step inflated scores by 25 points, and finding that required auditing our own work as aggressively as everyone else's.

## Slide 18 — Future work (report §9)

Five directions, all following from limitations we measured rather than wishes: real features for real interpretability and fairness, longer data for real drift, real cost calibration, conformal prediction once there is enough data for its guarantees to mean something, and the systems extensions - banded thresholds and production monitoring. Each one is scoped in section 9 of the report.

## Slide 19 — What we'd like you to remember

To close: four numbers carry the whole project. 0.99 versus 0.73 - the leaky score and the honest one, and we itemized which mistake causes the gap. 44,000 dollars - the measured value of honest probabilities under a decision rule that spends money. Thirteen-point-eight times - the same mistake ships a non-generalizing operating point, not just an inflated number. And 95 frauds - the reason every claim needs an interval, including our own money headline, which failed its fresh-seed confirmation - and we say so. Everything is one command to reproduce. We own the limitations explicitly. Thank you - questions welcome.

## Slide 20 — References

Nothing to read aloud - leave on screen briefly. The acknowledgement line matches the course AI-use policy; edit or remove it if the team decides otherwise.
