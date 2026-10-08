# Benchmark results

The real-QPU comparison uses **4,515 archived observations** and 165 conservative
groups. Every method receives the same five outer test partitions. There are
32 reported variants, including raw and separately calibrated analytical outputs,
not 32 original paper methods.

| Track | Main comparison | Supporting evidence |
| --- | --- | --- |
| Real QPU | [Method table](../results/real_qpu/method_comparison.md) · [CSV](../results/real_qpu/method_comparison.csv) | [Identical-row comparisons](../results/real_qpu/shared_row_comparisons.csv) · [Source/backend slices](../results/real_qpu/source_and_backend_metrics.csv) · [Seed metrics](../results/real_qpu/seed_metrics.csv) |
| Simulator | [Method table](../results/simulator/method_comparison.md) · [CSV](../results/simulator/method_comparison.csv) | Saved labels, attempts, quality and predictions under `data/simulator/`; comparison is within engine/clock/precision context |

## Real QPU

On all 4,515 held-out observations, graph/global MAE is 0.834 s (R² 0.468),
matched global MLP 0.893 s (R² 0.363), and inner-selected regression 0.887 s
(R² 0.411). Graph has the lowest pooled learned-method point estimate, but its
paired intervals against the MLP and selected regression both include zero.
The selector has lower source-balanced MAE: 0.879 versus graph 0.898 s.

QCRE and Qiskit duration routes cover all observations. Raw Hyb cost covers
4,175; log-cost Ridge covers 4,367; nominal Scholten covers 3,399. Missing
throughput and invalid survival/cost remain explicit failures, not zero inputs.
The original Scholten effective-depth route is unavailable because its required
template/kernel definition is not established. A partial-coverage MAE is not a
full-panel win; the pair table supplies both methods' metrics on identical rows.

Qonductor's 230 inputs are archive-supported recipes, not verified original
logical DAGs. QPack's 3,945 observations use six representative-angle structures,
not recovered historical parameters/routing. Reconstruction changes inputs,
never observed labels. Nominal calibration and heterogeneous service-time
boundaries constrain interpretation. See [dataset construction](data_preprocessing.md).

## Simulator

Transpiled Aer GNN MAE is 0.3008 s against XGBoost 0.2970 s; their paired interval
includes zero. Dense Random Forest has low MAE while work-scaled Ridge has the
highest R² among the evaluated predictors. Joint-MPS family conditioning has
no resolved runtime advantage; predicted-rung quality violations are 64/140.
The optimizer-off QCSim component model has MAE 0.0001037 s on 143/150 supported
hashes. Contraction and analog companions have distinct targets and quality
conditions; their counts do not add methods to a generic wall-clock leaderboard.

## Reading the evidence

[Scientific review](scientific_review.md) records validation scope and limitations.
[Paper comparison](paper_comparison.md) links published mechanisms to local
findings. [Analysis](analysis.md) defines metrics and grouped uncertainty.
The four [figures](../results/figures/manifest.json) show dataset composition,
paired QPU uncertainty, simulator errors and coverage/quality.
