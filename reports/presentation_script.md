# Presentation Script — *Fraud Detection, Measured in Money*

**Deck:** `reports/fraud_presentation.pptx` (12 slides) · **Target length:** 8–12 min
**Also in the deck:** every script below is embedded in that slide's Presenter View notes.

**Suggested speaker split** (adjust freely — matches the roles in `PROJECT_PLAN.md §7`):

| Presenter | Owns | Slides |
|---|---|---|
| **P1** — Data & Leakage | the problem, forensics | 1, 2, 3, 7, 8 |
| **P2** — Benchmark & Money | E1/E2 | 4, 5 |
| **P3** — Calibration | E3 | 6 |
| **P4** — Statistics | E5 | 9 |
| **P5** — Diagnostics & Report lead | E6–E8, wrap-up | 10, 11, 12 |

Delivery tips: speak to the *one number* on each slide, pause after it, then explain. Read the **[cue]** lines as stage directions, not aloud. Total ≈ 11 min at a calm pace.

---

## Slide 1 — Title  ·  *P1*  ·  ~30 sec

"Hi, we're group [X], and our project is credit-card fraud detection — but with an unusual angle: **honesty**.

The standard dataset we use has been analysed in thousands of public projects, and most of them proudly report 99%-plus scores. The problem is that most of them are *accidentally cheating*. So we did three things: we built a fraud detector that **can't** cheat, we measured it in **dollars** instead of abstract scores, and we ran a forensic audit that quantifies exactly how much the usual shortcuts inflate the numbers.

**[cue: point to the two numbers, top-right]** These two numbers are the whole talk. Same models, same data: **0.99** if you evaluate the way most notebooks do — **0.74** if you do it honestly. Let me explain why that gap exists."

---

## Slide 2 — The 0.17% trap  ·  *P1*  ·  ~45 sec

"First, why this problem breaks normal habits. We have 284,807 transactions, and only **492 are fraud** — that's 0.17%, about **one in every 579**.

At that rarity, accuracy is a trap. A model that flags *nothing* — that does literally no work — is **99.83% accurate**. So if you report accuracy, a useless model looks almost perfect.

**[cue: gesture to the red card]** That's why we throw accuracy out as a headline and use metrics that actually mean something at 0.17%: PR-AUC for ranking quality, precision and recall at a real cutoff, and — our main contribution — dollars.

The rarity also creates the second problem: because fraud is so rare, everyone manufactures synthetic fraud examples with a tool called SMOTE — and one misplaced line of that is what corrupts most projects. I'll show you how."

---

## Slide 3 — How the workflow cheats  ·  *P1*  ·  ~60 sec

"Here's the mistake, and it's subtle. **[cue: top red flow]** The common workflow takes *all* the data, uses SMOTE to synthesise fake fraud examples, and *then* splits into training and test.

Two things go wrong. Synthetic near-copies of the test frauds leak into the training set — and the test set itself becomes roughly half fraud instead of 0.17%. It's like studying with the answer key and then being amazed you aced the exam.

**[cue: green flow]** Our pipeline makes that impossible by construction. We split **first**. Then everything that learns anything — scaling the numbers, running SMOTE — lives inside one sealed pipeline that only ever sees the training data. The decision cutoff is tuned on a separate validation set, and the test set is opened exactly **once**, at the very end.

We also drop duplicate rows before splitting, and we never let the cutoff peek at the test set. Cheating isn't avoided by discipline here — it's structurally impossible. Now [P2] will show what this honest detector is actually worth."

---

## Slide 4 — The money scoreboard  ·  *P2*  ·  ~60 sec

"Thanks. Instead of only reporting abstract scores, we price everything. In our cost model, investigating an alert costs **five dollars**, and a missed fraud costs its own transaction amount. 'Savings' is then the share of do-nothing losses a model prevents, net of those alert fees.

**[cue: walk up the bars]** Three reference points anchor the scale. Doing nothing is zero by definition. A **perfect oracle** tops out at 0.97 — not 1.0 — because 45% of frauds are worth less than the five-dollar fee, so even perfection shouldn't chase them. And our best honest model, a class-weighted random forest, prevents **69% of losses** — about **$7,800** saved per test slice of 57,000 transactions.

Not shown on the chart: 'block everything' scores **minus 24** — it pays $284,000 in fees to stop $11,000 of fraud. Off the chart, literally. But here's the finding that matters most."

---

## Slide 5 — The F1 winner loses the money  ·  *P2*  ·  ~45 sec

"Watch what happens when the metric changes. **[cue: left card]** This model — SMOTE plus random forest — ranks **second out of twenty** by F1. A metric winner.

**[cue: right card]** But measured in dollars, it finishes **dead last**. It lets $4,683 of fraud value through, versus $3,002 for the money winner — even though their classification scores are nearly identical.

Why? Because F1 counts frauds, and it can't see that one missed $3,000 fraud outweighs a hundred missed $2 frauds. The model catches lots of frauds — but the cheap ones.

The takeaway: **neither F1 nor PR-AUC identified the model a bank should actually deploy — only the money metric did.** And the PR-AUC winner is a *third* model. Three scoreboards, three different champions. Over to [P3] on how we make the probabilities trustworthy enough to act on."

---

## Slide 6 — Calibration, priced in dollars  ·  *P3*  ·  ~75 sec

"Calibration usually shows up as an abstract chart. We put a dollar figure on it.

The smart decision rule is this: flag a transaction when its **probability times its amount** exceeds the five-dollar fee. That's just expected loss — no tuning needed. But it trusts the probabilities literally, so they'd better be honest.

**[cue: reliability chart]** Here's the catch. Class-weighted logistic regression outputs probabilities calibrated to an artificial 50/50 world — they're inflated about **300-fold**. Feed those raw into the rule and it fires on everything: savings of **minus 3.56** — it loses three and a half times the total fraud value in fees.

**[cue: the table]** One line of Platt scaling — fit only on validation — repairs that *same* model to **plus 0.67**. That's a **$48,000** swing per test slice. And the table is the real result: badly miscalibrated models gain tens of thousands of dollars from calibration; already-calibrated models like LightGBM gain nothing.

So calibration isn't a checkbox — it has a price tag, proportional to how broken your probabilities were. [P1] will now show how we audited the leakage itself."

---

## Slide 7 — Leakage forensics  ·  *P1*  ·  ~75 sec

"This is the centrepiece. Instead of one 'leaky versus safe' comparison, we turned each classic mistake into an independent on/off switch — SMOTE before split, scaling on all data, keeping duplicates, tuning the threshold on test — and ran **all sixteen combinations**.

The key design: every workflow is scored **twice**. Once on its own — possibly poisoned — test set, which is the number a flawed notebook would publish. And once on a **pristine holdout** we carved out before any mistake could touch it.

**[cue: the bar chart]** The attribution is unambiguous. **SMOTE-before-split alone inflates the reported score by 0.25** — essentially the *entire* lie. Every other sin is within noise of zero. And it's a poisoned test set, not a better model: the true score moves by two ten-thousandths.

**[cue: green card]** The proof our own method works: the honest workflow's reported score, 0.744, matches its clean-holdout truth, 0.742. Honesty verified, not assumed. But that one sin does something even worse than lie."

---

## Slide 8 — The broken product  ·  *P1*  ·  ~45 sec

"The same mistake ships a broken product.

**[cue: two cards]** SMOTE-before-split reports an F1 of **0.95**. But take that exact model and threshold, and run it on clean data at the real 0.17% prevalence, and its F1 is **0.07** — a **thirteen-and-a-half-times** overstatement.

Here's why: the decision threshold was tuned in a fake, half-fraud world. Deployed at one-in-579, it drowns in false alarms. So the sin doesn't just make you *look* better than you are — it makes you *ship* a worse product while believing it's excellent.

Our dual evaluation is what separates the two harms: threshold-free metrics expose the lie, deployed-artifact metrics expose the broken product. Now [P4] on whether any of our own results are even real."

---

## Slide 9 — Which wins are real?  ·  *P4*  ·  ~75 sec

"With only 95 frauds in the test set, how much of any ranking is luck? We answer that with a paired bootstrap: resample the same test rows a thousand times for all models at once, so sampling luck cancels and the confidence interval lands on the *difference* between models.

**[cue: forest plot]** Result one: every pairwise PR-AUC 'win' among our top models is statistically indistinguishable — every interval crosses zero. LightGBM's crown over XGBoost is five thousandths, with the interval straddling zero in every seed. The single-number leaderboard everyone publishes is noise at this scale.

**[cue: green card]** Result two — and we're proud of this — we pointed the same weapon at our *own* headline. On our discovery seeds, the money gap's interval only touched zero. So we did the honest thing: we pre-registered it and re-tested on **fresh, unseen seeds**. There, both intervals exclude zero — **the gap is confirmed on held-out data**.

Discovery, pre-register, confirm. We proved our own claim the hard way, and reported exactly how. [P5] will show what's inside the model."

---

## Slide 10 — Opening the box  ·  *P5*  ·  ~60 sec

"We opened the model three ways. **[cue: SHAP chart]** Globally, SHAP and permutation importance agree on the drivers — components V14, V12, V10 — though we're honest that these features are anonymised, so they name components, not human causes.

**[cue: case cards]** Locally, two cases earn their place. The model's most confident false alarm was a **two-dollar** transaction — even a *correct* alert there loses money, which is exactly why our per-transaction rule exists. And its most expensive miss — **$1,097** — scored zero-point-zero-zero-zero: the strongest fraud signals pointed the wrong way. A blind spot, not a near-miss.

And economically, misses are wildly unequal: nine tiny misses cost fourteen dollars total; two big ones cost $1,646. Two more honest results at the bottom: evaluated forward in time, savings drop from 0.69 to 0.61 — deployment is harder than random splits admit — and a proper hyperparameter search changed nothing beyond noise, which at 280 training frauds is the expected truth, and we say so."

---

## Slide 11 — How we worked  ·  *P5*  ·  ~45 sec

"One slide on method, because it's the point of the whole project. **[cue: the four cards]** Every experiment went through the same loop: build it leakage-safe, subject it to independent adversarial review, fix what's confirmed, verify the fix — and only then run for real.

This wasn't for show. It caught real flaws in *our own* work: our supposedly clean holdout was contaminated, because the dataset has **9,144** duplicate rows in the space the model sees — eight times the 1,081 everyone knows about, which is itself a data-quality finding. It caught a metric that was quietly cherry-picking on test data. And a 'hypothesis' that was mathematically incapable of failing.

The discipline we preach in the project, we applied *to* the project."

---

## Slide 12 — Close  ·  *P5 (or all together)*  ·  ~45 sec

"To close, four numbers carry the talk. **0.99 versus 0.74** — the lie and the truth, and we itemised which mistake causes it. **$48,000** — the measured value of honest probabilities. **Thirteen-and-a-half times** — how badly one sin overstates a deployed model. And **95 frauds** — the reason every claim needs an interval, including our own headline, which we confirmed on held-out data.

Everything is reproducible — one command per experiment, seeds fixed, dependencies pinned. And we own our limitations openly: anonymised features, a two-day window, an assumed cost model, and post-hoc comparisons labelled as such.

Thank you — we're happy to take questions."

---

## Q&A prep — likely questions and crisp answers

- **"Why dollars on European data?"** — The dataset's amounts are natively in euros; we relabel 1:1 to dollars for a familiar unit. Every conclusion is a *ratio* or a comparison, so it's unit-invariant — the story doesn't change.
- **"Isn't 95 test frauds too few to conclude anything?"** — Exactly why we built the bootstrap. We *don't* over-claim on thin evidence; we report which gaps survive (the money gap, confirmed on fresh seeds) and which don't (all PR-AUC wins).
- **"Why not deep learning / a bigger dataset?"** — The data is tabular — 30 anonymised features, no spatial or sequential structure — where gradient-boosted trees are state-of-the-art; CNNs/transformers have nothing to exploit. A second dataset (IEEE-CIS) was scoped and deliberately cut to keep depth over breadth.
- **"Is SMOTE always bad?"** — No. SMOTE *inside* the pipeline (train-fold only) is fine; SMOTE *before the split* is the sin. Our forensics isolates exactly that placement.
- **"Which model would you actually deploy?"** — Class-weighted random forest with calibrated probabilities and the per-transaction decision rule — it's the money winner, and calibration makes its probabilities safe to act on.
- **"How is this reproducible?"** — `git clone`, download the dataset with one script, then one command per experiment regenerates every table and figure. Seeds are fixed; dependencies pinned.

---

## Demo script — the ~3-minute code walkthrough (word for word)

*Referenced from the "Live demo" slide. Screen-record this from the repo root,
same terminal window and font size throughout.*

**[Terminal visible, repo root]**
"This is the whole project: `src` is the pipeline library, `experiments` has
one script per experiment, and everything they produce lands in `results`."

**[Open `src/pipeline.py`, hold ~20 seconds]**
"This file is the no-cheating guarantee. Everything that learns from data —
the scaler, SMOTE, the model — is sealed into one pipeline object, and that
object is only ever fit on training folds. The evaluation can't leak, because
there's no code path where test data reaches a fitted step."

**[Run `python3 -m experiments.run_benchmark --quick`, narrate as it prints]**
"First the integrity check: 284,807 rows, 492 frauds, and it drops the 1,081
duplicate rows *before* splitting, so identical transactions can't sit on
both sides of the split."
"Now it trains the same models two ways: the safe way, and the way most
public notebooks do it — SMOTE applied before the split."
"And there's the gap, live: the leaky pipeline reports essentially perfect
scores; the honest one tells the truth. Same models, same data — the only
difference is when the split happens. Note the quick run uses one seed, so
the exact numbers are noisy; the report uses the full five-seed benchmark."

**[Run `ls results/tables/`]**
"Every number in the report and on the slides comes from these CSVs — nothing
is typed in by hand. One command per experiment regenerates all of it from
the raw data."
