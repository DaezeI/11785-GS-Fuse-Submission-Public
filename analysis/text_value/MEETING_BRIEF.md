# Nicole's meeting update: does text actually help?

## What I contributed

I built and ran a reproducible analysis of the real/shuffled/no-text results Adaeze uploaded, checked her split lists against the price bars, and added a new explicit no-text evaluation path with tests. The code and numerical results are shared in this branch.

## Finding 1: most of the observed price-error benefit survives shuffling

Real text has **5.88% lower overall error than no text**. But **98.10% of that improvement remains when the articles are shuffled**; the correctly matched article adds only the remaining 1.90% of the observed improvement.

At 35, 70, and 140 bars, shuffled articles retain roughly **94–96%** of the observed improvement. No text has higher reported direction accuracy at all three horizons, despite worse price error.

**What this means:** the existing checkpoint is weakly sensitive to which article it receives. It does not establish that text is useless or explain the mechanism. These are new calculations on Adaeze's existing experiment, not new Llama inference.

## Finding 2: a built-in “price-only” comparison already contains text

The code first lets prices attend to the article, then sends that resulting representation into its named time-series-only comparison. That is not a strict article-removal test. The same comparison helps construct the gate's training target.

I added an alternative that starts with fresh price features and supplies zero article tokens to the normal fusion path. A second option bypasses fusion entirely. Tests check that article changes cannot affect these controls and that price changes still can.

**What this means:** we have a concrete reason to make the comparison explicit before drawing conclusions about text usefulness or gate behavior. The numerical impact remains unmeasured. I have not established how Adaeze's separate zero-text script implements removal.

## Finding 3: the proposed split windows pass an independent check

All three split lists have zero shared input/forecast bars across groups under the two timestamp rules tested. C1 keeps training before validation before test. The price file also contains **162 exact duplicate rows from two repeated sessions**; I produced a separate sorted copy with those removed.

This verifies price-window separation under the tested rules. It does not verify historical report publication times or which data an existing checkpoint saw.

## What ran, and where?

On the local computer's CPU: aggregate text calculations, full price-window checks, and simple C1 price-only reference models. The no-text helper also passed real CPU PyTorch tests. No Colab GPU or new full GS-Fuse inference was used. The repository contains the results, source hashes, scripts, and a reproduction notebook.

## Proposed next experiment

Run the paired evaluation tool on one agreed checkpoint: real articles, several shuffled versions, and the new explicitly defined no-text condition. Save one prediction per example so we can compare event types, past volatility, and horizons with uncertainty estimates. For final forward-forecasting claims, use a checkpoint trained under the agreed C1 protocol.

My handoff is the evaluation code and these measured findings. The follow-up needs matching checkpoint/report inputs; there is no need to wait for teammate replies to review or reproduce the aggregate analysis now.
