> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# B2 readiness

| Route | Raw/context coverage | Decision |
|---|---:|---|
| QPU Graph |45/45 predeclared raw-available,45/50 original scope|READY FOR B2 MEASUREMENT using declared historical preprocessing bridge|
| QPU MLP |45/45|READY; existing Phase2 timing remains a reference,not matched to the new bridge|
| Dense FP32 Graph/MLP |30/30 each|READY using the same bridge and frozen fold contracts|
| Dense FP64 Graph/MLP |30/30 each|READY using the same bridge and frozen fold contracts|
| Single CUDA-interpreter raw preprocessing |not restored on Qiskit0.44.2|BLOCKED on that path; no dependency changes authorized/performed|

READY is input compatibility,not prediction replay or measured latency. CUDA consumer constructs CPU tensors using exact typed arrays; no forward or explicit CUDA calls. Torch's configuration import attempted device discovery and warned under the address-space safety cap; this was not a model/kernel experiment. Future inference requires fresh authorized CUDA smoke/seed replay before any timing. Remeasure both Graph and MLP with identical bridge/framework/setup policy. Old MLP timing must not be spliced into a new Graph comparison.

Remaining limits:QPack5 no serialized raw inputs,competitive classical states missing,dense original preprocessing package identity incomplete,IPC/dependency split requires explicit B2 boundary,external input acquisition and cold-start remain separate. No full Pareto frontier or universal scaling claim.
