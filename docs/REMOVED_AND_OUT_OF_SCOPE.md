# Removed and out-of-scope replications

**Date removed:** 2026-09-22.

These items needed cloud/IBM credentials or lacked a join/label that would make them a real experiment. They were deleted or marked out of scope so the capsule only keeps work that actually ran.

| Item | Why it left the capsule |
| --- | --- |
| `artifacts/validation/mali_family_ood_v1/prospective_qpu_pilot/` | Live IBM B1 plan. Credential file empty. No jobs submitted. |
| `experiments/plan_mali_experiment_b_extensions.py` | Generator for that unused pilot. |
| BlueQubit cloud MPS (Zero-Setup remote half) | Never executed. Local Pauli-propagation remains. |
| MPORA / PGTNet | Not quantum runtime. Never executed. |
| Historical 12-family Osaka/Kyoto `time_taken` | Machines retired. Recorded as a data limit in family-OOD A/C, not as a fake run. |

Family-OOD **A, C, and B2** stay. They use public Ma–Li labels and current FakeBackend structure. They do not submit QPU jobs.
