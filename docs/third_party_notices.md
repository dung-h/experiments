# Third-party material and provenance

This is an inventory of source material found in the `presentation-review`
candidate at the review baseline, commit `4e42946`. It records evidence and
open decisions; it does not grant a license or establish legal clearance.
The project has no tracked root `LICENSE`, `CITATION.cff`, `NOTICE`, or
`COPYING` file. Public redistribution remains held by
[S56](../benchmark_v1/S56_WAVE5_RELEASE_LICENSE_AND_PROVENANCE_DECISION_V1.md).

## Included material

| Material in this repository | Source and evidence | Current rights / notice status |
| --- | --- | --- |
| Canonical 8,767-row archived QPU ledger, labels, source IDs and derived benchmark evidence | [`canonical_observations.csv`](../artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv); construction and limits in [methodology](methodology.md) | Combines Ma–Li (340), Qonductor (4,482) and QPack (3,945) observations. The table is derived from source data; S56 does not establish redistribution rights for these rows. Owner must review the source dataset terms and decide whether these derived labels/metadata may be publicly distributed. |
| QPack structural reconstruction evidence | [Reconstruction report](../artifacts/benchmark_v1/qpack_mcp_structural_reconstruction_20260927/REPORT.md); source pin and six-structure scope are recorded there | The bundle contains qualified structural reconstructions, not optimized angles or exact submitted circuits. Rights for redistributing the source-derived labels and reconstructed structures are not closed. Owner must approve or exclude the affected rows/material in public staging. |
| VQCSim RQ2 circuit outputs and local measurement artifacts | [`VQCSim protocol and provenance`](../experiments/simulator_papers/README.md), [`upstream lock`](../upstream/upstream.lock.json), and [run manifest](../artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_16_24_20260920/manifest.json) | The repository includes generated/normalized QASM circuit outputs and metadata; the VQCSim source checkout is external. The manifest records MQT Bench 2.1.0. The applicable notice/redistribution terms for these generated circuit artifacts have not been recorded in this candidate; owner must review before public staging. |
| DAQEC calibration CSV (378 rows) | [`DATASET_README.md`](../artifacts/validation/mali_snapshot_variability/DATASET_README.md) and [`SOURCE_MANIFEST.json`](../artifacts/validation/mali_snapshot_variability/SOURCE_MANIFEST.json) | The source record states CC-BY-4.0 and the dataset README requires attribution. Retain that attribution if distributing the CSV. This calibration series is not Ma–Li runtime data. |
| Four Qiskit-derived SQGM/SABRE files | [`sabre0330_layout.py`](../tracks/cdaa_qcre/independent/sqgm/mapper/sabre0330_layout.py), [`nassc_swap.py`](../tracks/cdaa_qcre/independent/sqgm/router/nassc_swap.py), [`sabre0330_swap.py`](../tracks/cdaa_qcre/independent/sqgm/router/sabre0330_swap.py), [`sqgm_swap.py`](../tracks/cdaa_qcre/independent/sqgm/router/sqgm_swap.py) | File headers identify IBM copyright, Apache License 2.0, and modification notices. The headers point to a `LICENSE.txt` that is not tracked here. Include the applicable Apache-2.0 license text and check required notices in any public staging package. |
| CDAA/QCRE instruction-duration snapshots | Six files under [`tracks/cdaa_qcre/data/instruction_durations/`](../tracks/cdaa_qcre/data/instruction_durations/); context in [track README](../tracks/cdaa_qcre/README.md) | README describes these as archived snapshot inputs. Their originating backend records and redistribution terms are not identified in the notice materials inspected here; owner must verify provenance/terms or exclude them from public staging. |

The repository also contains derived model inputs, predictions, timing rows,
plots and reports. They are project-generated outputs based on the sources
above; their presence does not resolve rights in the underlying source data.
The detailed file-level inclusion/exclusion inventory is
[`release_inventory.csv`](release_inventory.csv).

## External material not bundled

| External source or payload | Pinned/source evidence | Treatment and remaining decision |
| --- | --- | --- |
| Complete Ma–Li source checkout and logical-QASM corpus | [`upstream.lock.json`](../upstream/upstream.lock.json), Ma–Li commit `32c392a6ece276f1ff046d4e30052d0571ff6dc6`; [S56](../benchmark_v1/S56_WAVE5_RELEASE_LICENSE_AND_PROVENANCE_DECISION_V1.md) | Not bundled. No license file was found in the checked checkout. Its README attributes circuit source to MQTBench, but exact corpus-to-MQTBench file/hash mapping is unverified. Keep external or establish mapping and required notices before distribution. |
| Qonductor full source checkout payload (`circuits.zip` and database export) | [`upstream.lock.json`](../upstream/upstream.lock.json), Qonductor commit `5d1ac8a90cd574a23e7544e1044681641354ff67`; [S56](../benchmark_v1/S56_WAVE5_RELEASE_LICENSE_AND_PROVENANCE_DECISION_V1.md) | Full archive is not bundled. S56 records the source checkout as MIT-licensed but leaves archive/data redistribution treatment unresolved. Owner must decide whether any archive or extracted source circuits enter public staging and preserve the applicable notice. |
| QPack original execution payload and optimized-angle circuits | [Methodology](methodology.md) and [QPack reconstruction report](../artifacts/benchmark_v1/qpack_mcp_structural_reconstruction_20260927/REPORT.md) | Not recovered and not bundled. The included reconstruction is limited to the declared structural recipe. There is no pinned source payload or redistribution grant in this candidate. |
| Maestro source checkout/report | [S56](../benchmark_v1/S56_WAVE5_RELEASE_LICENSE_AND_PROVENANCE_DECISION_V1.md) | Kept external. S56 records the pinned checkout as GPL-3.0; do not vendor its code or report without checking license compatibility and required notices. |
| Local Maestro optimizer-control patch | [`native_build_provenance.json`](../artifacts/benchmark_v3/simulator/maestro_cpu_component_optimizer_off/native_build_provenance.json), patch SHA-256 `386a945dde3a503488d9abd68b1235e2785327175963f888c1296c20fb014c13` | The patch changes `python/bindings.cpp` in the GPL-3.0 Maestro checkout. It is retained only in the local worktree for audit and is not staged in this candidate pending rights review; the saved measurements do not make the patch distributable. |
| Other pinned upstream checkouts | [`upstream.lock.json`](../upstream/upstream.lock.json) and [simulator provenance notes](../experiments/simulator_papers/README.md) | `work/` checkouts are external/local and not part of the tracked package. The lock file pins revisions; it is not itself a license or permission record. Review each source and any generated data before copying material into public staging. |

## Decisions required before a public release

1. The repository owner selects the project code license and supplies accurate
   citation authors/metadata; neither is inferred here.
2. The owner approves the redistribution scope for the derived Ma–Li,
   Qonductor and QPack rows in the canonical ledger, or removes affected
   materials from public staging.
3. Resolve the exact Ma–Li-to-MQTBench circuit mapping and notices, and the
   Qonductor archive/data terms, as required by S56.
4. Confirm the notices/terms for bundled VQCSim circuit outputs,
   instruction-duration snapshots and QPack reconstructions.
5. Add the Apache-2.0 license text and required attribution for the four
   Qiskit-derived source files; retain CC-BY-4.0 attribution for the DAQEC
   calibration CSV.
6. Re-audit the transformed public staging tree and its source-to-staged hash
   map before any tagged release or `CURRENT.json` promotion.

## Evidence hashes

SHA-256 values below identify the evidence files inspected for this inventory.
The hash of this document is intentionally not listed here.

| Evidence file | SHA-256 |
| --- | --- |
| `benchmark_v1/S56_WAVE5_RELEASE_LICENSE_AND_PROVENANCE_DECISION_V1.md` | `c028895dd3346e86c29cad91fee77d9e1065a7364d775cd35a9d163229ae044d` |
| `upstream/upstream.lock.json` | `1ee1722b0109283874c2c20c431b6536c292873da2ceaf59c4fbde70b68ce976` |
| `docs/release_inventory.csv` | `2b2f24d1e8a180cf2d40a8cb4f31e1742e292292837498caedf229c9eeac2ed1` |
| `artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv` | `920d745e03dd00e8155118d9323baea18ff865e4df2620f3a059bd5503a05ba4` |
| `artifacts/validation/mali_snapshot_variability/DATASET_README.md` | `238fefffb7a220c3e5d8f7fca61f2a8470e4bcc98a2de203be14b1b8ffc44856` |
| `artifacts/validation/mali_snapshot_variability/SOURCE_MANIFEST.json` | `0ab2e4aee6b43a94c65fbf8dc7bad1df8cdc9e5021edada36510e98a2b8badf5` |
| `artifacts/validation/mali_snapshot_variability/drift_characterization.csv` | `dc31d7db266cbad65173e6942de088740d19ed3f0f6e0dadc286d51e8fa042a3` |
| `artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_16_24_20260920/manifest.json` | `9c582a3700b45e81a8a3cda396c76d84e3b59637b72c30661014b93ad6b3eb87` |
| `experiments/simulator_papers/README.md` | `9e17b6ce0372ee2ad59a7474ec23ca38c75c5504f1636ab131d7bed4ceda2510` |
| `tracks/cdaa_qcre/README.md` | `65f4d9539704381ec5f801d927c677fddda91d4ee564d345209cfa299847f9f3` |
| `artifacts/benchmark_v1/qpack_mcp_structural_reconstruction_20260927/REPORT.md` | `e46f1ba70bd8e812e8caaeba62e8fa57fe9490822ce5721a4355635efb875fbc` |

The four Qiskit-derived files are identified in the table by their preserved
file headers; their individual hashes are available from the release
inventory.
