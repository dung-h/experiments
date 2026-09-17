# Qonductor overlay

The patch applies to upstream commit 5d1ac8a. It adds a public-artifact metric
recomputation, fake-backend scheduler smoke test and report. It is checked by
scripts/bootstrap_upstreams.sh before application.

The environment used Python 3.10 with the pinned requirements file in this
directory. No IBM token, queue lookup or job submission is needed for the
committed public-artifact analysis. The result is a derived recompute of author
predictions plus a local fake-backend smoke test.
