# S56 — Wave-5 release license and provenance decision

**Decision date:** 2026-09-29  
**Owner:** expert/strong review  
**Scope:** decide whether the Wave-4 review package may become a public
redistribution package. This is a release-boundary decision; it changes no
benchmark result, method eligibility, raw evidence or scorecard value.

## Decision

**Public redistribution and `CURRENT.json` promotion are held.** The current
Wave-4 package remains a review-only, `review_required` closure.

This is not a scientific validity failure. C115 proves that the local review
closure reproduces mechanically. The hold is due to release rights and
attribution, which clean-clone validation cannot establish.

## Evidence and treatment

| Material | Observed boundary | Release decision |
| --- | --- | --- |
| This benchmark repository | No root `LICENSE` or `CITATION.cff` was present at the audit point. | The author must select the repository license and provide citation metadata; do not infer either. |
| Maestro external checkout | Its pinned local checkout has a GPL-3.0 license. | Keep the checkout/report external; do not vendor Maestro code or report into this repository without a compatible release decision. |
| Qonductor external checkout | Its pinned local checkout has an MIT license. | Keep `circuits.zip` external until its data/redistribution treatment is explicitly recorded; do not silently bundle the archive. |
| Ma--Li external checkout/corpus | No local license file was detected. Its README says its circuit source is MQTBench. MQTBench is publicly MIT-licensed, but the exact derivation and scope of the checked-out corpus have not been proved here. | Treat the checkout and corpus as non-redistributable external provenance until exact source/hash/notice mapping is recorded. |

The MQTBench upstream project states that it is MIT-licensed. That fact may
support a later source-provenance closure, but it is not a license grant for
an unidentified transformed subset in a separate repository.

## What Wave 5 does permit

1. Publish the audit, source-pin inventory and logical path-redaction plan to
   reviewers as review artifacts.
2. Keep the bounded C114/C115 package as a locally reproducible package.
3. Produce a transformed public staging tree only after it carries its own
   original-to-staged hash map and visible third-party notices.

## Conditions to unblock public release

1. The author selects a license for this repository and supplies accurate
   `CITATION.cff` metadata.
2. Record the exact Ma--Li corpus provenance: upstream revision, file/hash
   mapping to the MQTBench-origin material, and required MIT notice(s), or
   exclude those files from public staging.
3. Record the Qonductor data-archive redistribution treatment and retain its
   MIT notice if any part is distributed.
4. Apply the successor public-surface audit
   (`public_release_surface_audit_v3_20260929`) only to a new public staging
   copy, never to immutable evidence; validate staged hashes and run
   clean-clone there.
5. Obtain final author approval, create immutable commit/tag, then atomically
   update `registry/CURRENT.json` and regenerate the final allowlist.

Until all five conditions are closed, no status is promoted and no claim of
public reproducibility beyond the local review closure is allowed.
