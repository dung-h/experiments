# Benchmark presentation outline V2

**Status:** approved content brief for a new deck.  
**Does not modify:** the hash-pinned historical `docs/presentation.md`, its
scorecard, or the existing exported PowerPoint.

This outline puts the construction of the benchmark ahead of individual paper
case studies. The deck must label each score as evaluated, diagnostic,
unavailable, or not evaluated. It must use the schemas in
`benchmark_v1/protocol/benchmark_result_table_schema_v1.json`.

## Ten-slide structure

1. **Question and scope**
   
   Predict runtime before scheduling in two domains: archived real-QPU
   observations and local simulator configuration cells. State `PARTIAL`, no
   live-QPU run, and no cross-clock global leaderboard.

2. **Building the unified real-QPU dataset**
   
   Show Ma–Li 340, Qonductor 4,482 and QPack MCP 3,945 as 8,767 archived
   observations. Explain label boundary, unit conversion, one-circuit filter,
   shots, and source identity.

3. **Circuit evidence and reconstruction**
   
   Contrast exact Ma–Li logical QASM, exact submitted Qonductor physical QASM,
   and QPack's six reconstruction-qualified structures. State that QPack angles
   and routing were not recovered, and reconstruction never replaces labels.

4. **Simulator panel and measurement context**
   
   Show 204 members, 191 hashes, 22 families, q2–q16, with 162 core and 42
   frontier. Include local host, engine/context, shots, first/warm clocks,
   MPS quality policy and terminal-status treatment. Do not depict a complete
   family-by-width grid.

5. **Comparison protocol and method matrix**
   
   Show frozen hash/workflow groups, outer-train-only fitting and full coverage
   denominators. List the six real-QPU and six simulator method families with
   fidelity class, output clock and status. Explain that 24 QPU variants are
   transformations, not 24 papers.

6. **Real-QPU learned methods**
   
   Present same-row graph V3-large versus Qonductor-style polynomial by source,
   alongside coverage. State that these are unified OOF models, not three
   independently trained source models; identify their mixed representation
   lifecycle as an adaptation.

7. **Real-QPU analytical methods**
   
   Present Qiskit/QCRE scheduled proxies, Scholten nominal throughput and
   Hyb-HANAS effective-cost route in a separate diagnostic table. Display
   evaluation target clock and method output clock; do not present raw proxy
   MAE as a same-clock winner.

8. **Simulator method results**
   
   Present local Aer graph/Ridge/GBR core evidence; the cuTensorNet
   same-selected-plan estimate/actual pair; CUDA-Q measurement coverage and
   MPS quality failures. Distinguish a measured engine from a predictor.

9. **Methods still unavailable**
   
   State three explicit statuses: Azizov source/hybrid/transpiled GNN pending
   a signed local run, Family-Aware missing compatible joint labels, and
   Maestro pilot gate failed with no predictor score. Keep Pasqal as an analog
   companion, not a digital-QASM row.

10. **Findings and bounded claims**
   
   Summarize supported findings, the evidence that constrains them, and the
   exact execution gates required before any missing method can be promoted.

## Required visible disclosures

- The real-QPU label is archived provider execution/service time, not pure
  pulse duration.
- Qonductor is a filtered one-circuit-job subset; QPack MCP is one circuit per
  optimizer evaluation and remains workflow-grouped.
- Error-mitigation metadata is not source-wide verified in the canonical
  ledger.
- Qiskit/QCRE raw scheduled outputs, cuTensorNet RUNTIME_EST, and observed
  local simulator wall clocks are distinct clocks.
- `unavailable` is a result status, not a missing cell to hide.

## Speaker-note sources

Each numerical slide must cite the exact source artifact, method ID, row set,
clock and configuration in its notes. The deck builder must not derive new
numbers, select a preferred seed, or use a historical result with a weaker
provenance state when the current aggregate exists.
