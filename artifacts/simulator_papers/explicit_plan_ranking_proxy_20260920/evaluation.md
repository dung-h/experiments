# Explicit contraction-plan ranking proxy

The corpus keeps one fixed scalar tensor network per group and compares three
named candidates: a cuTensorNet `TIME_TUNED` plan, left fold and right fold.
The measured target is warm CUDA-event contraction time only.

| Quantity | Value |
|---|---:|
| Candidate rows | 45 |
| Complete circuit groups | 15 |
| Min-FLOP chooses fastest plan | 93.3% |
| Min-FLOP median regret | 1.000× |
| Min-FLOP worst regret | 1.048× |
| Native TIME_TUNED median regret | 1.000× |
| Max scalar-output magnitude spread | 1.274e-07 |

The result tests a deliberately weak analytical baseline. It cannot be used
as a result for arXiv:2608.05819 or as evidence for a learned ranker.
