# Final paper and presentation source

Git is the versioned writing source. The midterm under report/overleaf is preserved.
This is an early final-paper scaffold, not a completed final report. Confirm the
course template, exact deadlines, full authors and citations before submission.

Compile from this directory: `pdflatex -interaction=nonstopmode main.tex`.
The results table is generated from results/screened-price-v1/source_metrics.json
with `python results/export_price_result.py` from the repository root. No GPU is
needed. The current manuscript includes the measured price result and explicitly
marks the uncollected return/architecture results as pending.

For Overleaf polishing, upload a versioned repository export retaining the
report/final and results paths; set report/final/main.tex as the main document,
or flatten paths in a separately recorded export. The Overleaf shared-project
landing page supplied by Nicole does not identify a specific project. No live
Overleaf project was changed. Bring final edits back to Git and record the exact
submission commit. Generated final PDFs and slides may also be archived in Drive.
