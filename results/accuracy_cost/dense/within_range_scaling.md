> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Within-range empirical associations

FP32

| method | feature | min | max | spearman | pearson |
|---|---|---|---|---|---|
| graph | width | 2 | 9 | 0.6054882412140539 | 0.28640073841414976 |
| graph | depth | 3 | 3034 | 0.9602498975912814 | 0.9986518295187606 |
| graph | gates | 4 | 3152 | 0.9941026480216246 | 0.9977037631975288 |
| graph | nodes | 4 | 3152 | 0.9941026480216246 | 0.9977037631975288 |
| graph | edges | 3 | 6168 | 0.9919893436511342 | 0.9993982366138662 |
| mlp | width | 2 | 9 | 0.5609207052436515 | 0.28515882080050325 |
| mlp | depth | 3 | 3034 | 0.9673760563860221 | 0.9992109095244802 |
| mlp | gates | 4 | 3152 | 0.9852009005578088 | 0.9984140168846516 |
| mlp | nodes | 4 | 3152 | 0.9852009005578088 | 0.9984140168846516 |
| mlp | edges | 3 | 6168 | 0.9830885868664672 | 0.999484766064585 |

Families:{'ae': 5, 'dj': 5, 'ghz': 4, 'graphstate': 4, 'grover-noancilla': 3, 'portfolioqaoa': 3, 'grover-v-chain': 3, 'qaoa': 1, 'portfoliovqe': 2}

FP64

| method | feature | min | max | spearman | pearson |
|---|---|---|---|---|---|
| graph | width | 2 | 9 | 0.5936052561001584 | 0.3111859295988713 |
| graph | depth | 2 | 3034 | 0.971589100691027 | 0.9997553366982257 |
| graph | gates | 4 | 3152 | 0.9860910753041904 | 0.9998085535643378 |
| graph | nodes | 4 | 3152 | 0.9860910753041904 | 0.9998085535643378 |
| graph | edges | 2 | 6168 | 0.9840881821248317 | 0.9999066981360625 |
| mlp | width | 2 | 9 | 0.5744421544030397 | 0.30679046495047596 |
| mlp | depth | 2 | 3034 | 0.9682464776381908 | 0.9998290151549244 |
| mlp | gates | 4 | 3152 | 0.9636141629580555 | 0.9996306325096606 |
| mlp | nodes | 4 | 3152 | 0.9636141629580555 | 0.9996306325096606 |
| mlp | edges | 2 | 6168 | 0.9602760076591248 | 0.9998326284810395 |

Families:{'ae': 5, 'dj': 5, 'ghz': 4, 'graphstate': 4, 'grover-noancilla': 3, 'portfolioqaoa': 3, 'grover-v-chain': 3, 'qaoa': 1, 'portfoliovqe': 2}

Thirty circuits/context. Spearman and Pearson are descriptive, not causal tests.
The metadata column `gates` is the normalized operation-node count, including
terminal measurement nodes. It is not a pure unitary-gate counter. It therefore
coincides with `nodes`, rather than providing independent corroboration.

Widths cover only 2–9 active qubits. The operation range is 4–3,152 nodes, with
3–6,168 edges in FP32 and 2–6,168 in FP64. Depth spans 3–3,034 and 2–3,034.
Spearman latency associations with operation count are strong: Graph/MLP
0.994/0.985 in FP32 and 0.986/0.964 in FP64. Depth associations are about
0.960–0.972. Width associations are weaker, about 0.561–0.605.

These patterns concern estimator parsing/representation/inference work, not
simulator statevector complexity. Width, depth, family and gate type are
confounded. Extreme large-operation circuits affect Pearson correlations.
No asymptotic, exponential or universal scaling law follows from 30 circuits.
No new circuits or elaborate scaling fit were used.
