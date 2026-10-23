# Bounded physical-circuit Transformer strategy

Status: design draft, 2026-10-23. This records the requested method-development strategy. No model or graph-extraction job is launched by this document. Any execution uses a new research amendment/namespace; the completed P1 gate and Scientific Freeze v2 remain historical records. Because existing outer results informed this draft, it cannot be described as an untouched confirmatory holdout study.

## Latest user direction: Ma–Li pilot before full Track A

The user subsequently requested a small Ma–Li Transformer study before scaling. This ordering supersedes the draft's full-panel-first implementation sketch below. The completed tabular P1 does not prohibit an independently declared graph experiment. It also does not constitute an Azizov-style source/transpiled graph benchmark.

1. **Pilot population:** restrict the frozen Track-A panel to Ma–Li: 340 observations, 148 conservative groups, fixed 1,024 shots. Inherit the original outer and inner assignments; do not create an easier random-row split. Test folds contain 68 rows each and respectively 29/30/30/30/29 groups. This is distinct from historical Study R's 170 raw-QASM grouping; do not pool those metrics or copy historical fold definitions.
2. **Representation feasibility first:** these Ma–Li rows use 300 saved compiled graph payloads, with median 257,776 and maximum 372,846 operation nodes. Fewer observations does not imply small individual graphs. Before selecting coarsening, test the resource feasibility of complete-operation sparse graph attention using small fixtures and then a guarded largest-graph canary. The existing counts alone do not establish an OOM. Dense all-pairs operation attention is a different algorithm and would scale quadratically.
3. **Choose one representation from resource/semantic evidence:** prefer complete-operation sparse dependency attention if it is feasible under the declared resource envelope. Otherwise document the failed canary and lock a bounded/coarsened representation before pilot scoring. The earlier 256-bin design below is a fallback proposal, not a completed resource decision. Never discard large graphs to improve scores.
4. **Pilot comparisons:** compare the chosen Transformer against NODE/TRUE/SHUFFLE/REVERSE controls on the identical Ma–Li rows, with matching descriptors, context, capacity and training protocol. Include source/compiled tabular controls fitted only within the Ma–Li pilot train partitions. Saved all-source P1 models were trained on a different training population and are not matched pilot baselines. Audit/reuse historical evidence before scheduling any duplicate cells.
5. **Independent benchmark question:** an Azizov-inspired comparison uses actual source versus compiled operation graphs and metadata derived from the corresponding circuit, under the same learner and context. The existing 36-number compiled P1 block does not answer that graph-representation question. Using sparse graph TransformerConv attention on operations and recomputed metadata is conceptually different from the optional 256-token region Transformer below. Do not call a compressed-region result an exact Azizov reproduction.
6. **Scale only after a locked pilot decision:** require a practical, paired gain against matched strong pilot baselines, acceptable seed/fold/tail behavior, and resource feasibility. Additional topology attribution requires TRUE to beat NODE and SHUFFLE; a good predictor need not automatically establish that claim. If the pilot supports only the representation claim, retain that narrower finding. A full 4,515-row source/transpiled graph study can follow as a separate frozen design; the original tabular P1 gate is not retroactively rewritten.

The Ma–Li pilot uses reconstructed compiled circuits and fixed shots. It cannot validate exact historical physical scheduling, cross-source behavior, or varying-shot response. Pilot success is a reason to test those broader settings, not evidence they already work.

## Why this method is worth testing

The previous P1 reduced a large operation DAG to 36 compiled/static numbers. Ridge benefited from combining source and compiled summaries; HGB did not establish a general static-summary gain. This leaves a concrete question: can attention over operation regions recover predictive information discarded by circuit-level aggregation? Correct dependency adjacency must earn an incremental benefit over the same regional descriptors without communication.

Two success claims are distinct:

1. Method value: the new predictor improves over strong source/compiled tabular models on identical observations with comparable information.
2. Connectivity value: true dependency communication improves over the same Transformer with no cross-region communication and randomized connectivity.

A model can pass the first and fail the second. A standard TransformerConv replacement alone is an architecture adaptation, not sufficient methodological novelty for a new method paper.

## Phase 0: representation and resource checks before training

Reuse the 4,515-row panel, its 165 groups, and original outer folds. Cache each of the 346 saved graph payloads once. Audit operation names/counts, executable-node semantics, barrier/measurement treatment, per-qubit dependency order, and graph-to-observation lineage. The 230 exact-physical Qonductor rows contain only 11 groups; they cannot independently support a broad claim about exact physical workloads. Most inputs are reconstructions.

Inspect graph summaries by source before accepting compression. Saved graphs have median 240,464.5 operation nodes and maximum 372,846; 76/346 exceed 300k. These counts motivate bounded attention, but do not prove an intact sparse GNN is impossible; no full-operation feasibility result is established by graph counts alone.

Primary proposal: at most 256 deterministic topological bins, stable label-free tie-breaking, no circuit deletion for size. Each token describes the operations assigned to one bin. Use the same fixed native-op vocabulary as P1, log work count, operand-arity counts and normalized bin position. Do not add calibration, new snapshots, physical-qubit embeddings or a QPU-coupling encoder. No per-node timing is assumed from the single scalar used in P1.

This is a compressed operation-region model, not an exact operation-level schedule. Before fitting, report within-bin dependency loss, retained cross-bin edges, edge multiplicity, occupancy, density and mask similarity for every treatment. Verify that deterministic binning leaves informative cross-bin connectivity. If compression reduces most examples to essentially identical chains or dense graphs, this design is not a clean topology test; stop and report that representation limitation before training. Do not choose bin count from outer accuracy.

Node descriptors, binning, positional coordinates and static features are generated once, then held identical in all treatments. Binning/position themselves encode order; the estimand is adjacency value conditional on that common order information, not removal of all topology-derived information.

## One architecture

Proposed fixed starting configuration: 3 attention blocks, embedding width 128, 4 heads, feed-forward width 256, dropout 0.1, residual connections and layer normalization. Reuse the same embedding, Q/K/V/output matrices, feed-forward layers, mean pooling, and scalar regression head in every treatment. Record actual parameter counts after implementation; do not claim matching from this sketch alone.

Each query bin attends to itself plus the predecessors permitted by its adjacency mask. This makes direction operational: node i receives information from predecessor j. A fixture must demonstrate this convention and its reversal. With 256 bins, the dense mask has 65,536 entries per graph per layer; actual memory/backward feasibility must be profiled with complete examples before fixing effective batch size. Use dynamic batches/accumulation if necessary to maintain one declared effective batch. No size-based sample removal.

The pooled graph embedding joins the same 88-dimensional H0 input (51 source descriptors + 36 compiled descriptors + log1p shots) in every treatment. Shots are a real input to the regression head. This does not assume linear scaling or establish shot extrapolation. Timing proxies are excluded from the first Transformer experiment so that an unverified graph/timing lineage does not become a model-design assumption.

A generic fully connected self-attention mask would make TRUE/REVERSE masks meaningless. The dependency mask must actually affect attention; this is a necessary implementation test.

## Four matched treatments, one budget

| Treatment | Allowed attention sources | Intended question |
|---|---|---|
| NODE | Self only | How much is explained by descriptors and common context without cross-bin communication? |
| TRUE | Self + actual predecessor bins | Does true dependency communication help? |
| REVERSE | Self + successors of the original graph | Does direction matter? |
| SHUFFLE | Self + deterministic surrogate predecessor bins | Does correct alignment between descriptors and dependency relations matter? |

For SHUFFLE, a concrete initial construction is a seeded adjacency relabeling `A_shuffle = P A P^T`, while leaving tokens and positions fixed. It preserves edge count, the global degree distribution and acyclicity, but not each token's degree. Audit unchanged-edge fraction and mask differences; the result tests connectivity-to-token alignment rather than an unrestricted topology claim. Do not call it per-node-degree-preserving. Keep this transformation fixed across epochs and model seeds; preserve its hash.

All variants retain every trainable parameter; NODE's cross-node mask is removed, not its attention module. Self edges prevent empty-softmax rows. With exactly one attention source, softmax is constant and Q/K parameters may have zero effective gradients; identical declared parameter counts do not mean identical active computation. Report this difference and verify gradients on fixtures rather than claiming perfect computation matching. Fixed global graph summaries already contain depth/edge information in every treatment: true adjacency must add value beyond them.

## Fitting and comparison contract to lock before execution

- Model seeds 42, 1234, 31415; one deterministic graph-transformation seed derived from the saved graph hash.
- Four treatments × five outer folds × three seeds = 60 final OOF cells. No fractions or additional architectures.
- Target log1p seconds; identical row-weighted MSE and training budget for all treatments. Primary selection and scoring use group-balanced MAE after expm1. Do not silently conflate fitting weights with reporting weights.
- Proposed optimizer AdamW, learning rate 3e-4, weight decay 1e-4; maximum 500 selection epochs. Lock all values in the new amendment, not retrospectively after observing test scores.
- Use inner-fold 0 for epoch selection without outer labels; then reset and refit on full outer-train for the selected number of epochs. This adds selection fits to the 60 final cells; include them in compute accounting. Fit scalers separately on each eligible training partition. Store initialization, transforms, selected epochs and fit IDs.
- Reduce complete three-seed OOF predictions by per-observation median; retain individual seeds and fold deltas.
- Primary paired contrasts: TRUE−NODE and TRUE−SHUFFLE in group-balanced MAE. Use 10,000 whole-structural-group bootstrap replicates, seed 42, and Holm correction for the two primary directional tests. REVERSE is secondary. These intervals condition on the fitted models and the previously studied dataset.
- Compare on identical rows to saved Ridge H0 (0.647324 s group-MAE) and HGB H0 (0.648004 s), which see the same global/context block. Also disclose the stronger saved HGB source-only value (0.630509 s) and timing-augmented reference (0.611596 s, different information). Report paired contrasts rather than comparing unpaired headline numbers.
- Proposed practical method threshold: at least 0.03 s or 3% group-MAE improvement over the strongest information-compatible saved comparator, with a favorable paired interval. Source/fold results and full error tails must be disclosed. Ma–Li-only improvement supports a narrower claim; it is not cross-source physical-runtime evidence.

Prior outer labels/results have already guided method choice on this dataset. Nested epoch selection does not erase that reuse. External/locked future data would be needed for a strong independent confirmation of a discovered new method.

## Decision logic

- TRUE improves tabular controls and matched NODE/SHUFFLE: evidence supports a useful dependency-aware estimator within the declared representation and dataset.
- NODE and TRUE improve tabular controls but TRUE does not beat NODE: regional representation/modeling helps; adjacency gain remains unestablished.
- TRUE beats NODE/SHUFFLE but not strong tabular controls: connectivity has a bounded effect, but practical new-method gain is unestablished.
- None improve strong controls: close this bounded design; retain its predictions and report the negative result. No automatic depth/width/attention sweep follows.

Changing connectivity is an intervention inside the estimator. It does not prove topology causes hardware runtime, reconstruct actual device scheduling, recover historical calibration, or establish universal backend transfer.

## Separate accuracy–cost measurement

The requested Graph/MLP/Ridge/RF/XGBoost cost benchmark remains a measurement task with its own common raw-QASM-to-prediction boundary. Freeze the candidate predictor before timing it; use saved fitted states and report unavailable states explicitly. Transformer training must not run concurrently with exclusive latency measurement. Method discovery and inference-cost evidence answer different questions and should keep separate receipts.
