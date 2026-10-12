> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Integrity checks

Discovered/readable manifest and receipt files: 515; fixed expected manifest paths absent: 0. Hash/pin records: 8726. Statuses: PASS 8538; MISSING 0; HASH MISMATCH 6; UNVERIFIED 182.

Each available manifest/receipt is hashed and its declared payload/source/code/output pins are recomputed where paths are available. Expected fixed manifests that are absent are listed as MISSING; no missing target path was found in this run. All 182 UNVERIFIED rows are named hashes without a resolvable file path (180 under `pins`, 2 under `output_hashes`); they are not missing files. These checks do not certify permissions or an external upstream source.

The derived `final_artifact_manifest.csv` records source pin rows and generated deliverables, but omits its own hash to avoid a recursive self-reference.

The scoped search includes the named authoritative packages, qualified/strict QPU panels, CUDA-Q MPS evidence, and only the RQ4 case roots named by the accepted source-map receipt; stale/unaccepted attempt roots are not swept in.
