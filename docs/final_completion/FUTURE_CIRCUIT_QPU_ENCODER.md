# Future idea: circuit encoder + QPU encoder

Status: IDEA_ONLY — recorded at the user's request on 2026-10-22.
This note authorizes no implementation, training, data collection, or change
to Scientific Freeze v2, P1, P2, or existing artifacts.

## Proposed representation

```text
Circuit encoder ------------------+
                                  +--> interaction/fusion --> observed runtime
QPU encoder ----------------------+
    coupling graph
    native gate set
    gate durations
    physical qubit count
    connectivity descriptors
    calibration statistics
    possibly T1/T2 and gate/readout errors
```

Shot count and other relevant execution settings must enter as explicit
context inputs, rather than being assumed recoverable from circuit structure.
The runtime target must retain a precisely documented observation boundary.

The hypothesis is that an interaction between workload structure and device
properties predicts runtime better than a circuit-only model or a simple
backend-ID/context model. No benefit is established by this note.

## Proposed interaction: simple fusion first, cross-attention as a later option

This is an unexecuted architecture idea, not an existing experiment or an
authorization to open an architecture-search branch.

The two graphs represent different objects:

- Circuit graph: nodes are gates/operations, with dependency edges. Operation
  embeddings may encode gate type (for example CX/ECR), operands and local
  circuit context such as predecessor/successor information.
- QPU graph: nodes are physical qubits; edges are supported hardware couplings.
  Node attributes may include T1/T2 and readout error. Coupling attributes may
  include supported operation, duration, error and direction, subject to the
  snapshot provenance requirements below.

An operation embedding can serve as a cross-attention query, with hardware
embeddings serving as keys and values:

```text
z_op_i = CrossAttention(query=h_circuit_i,
                       keys=H_QPU,
                       values=H_QPU)
```

For coupling properties to influence attention, edge attributes must be
incorporated into QPU node embeddings by the hardware encoder, represented as
coupling tokens, or supplied through an explicitly defined edge-aware attention
mechanism. Ordinary attention over raw qubit features does not automatically
consume coupling-edge attributes.

The intended question is which hardware resources are relevant to an operation
and its surrounding workload. Logical operands such as q2/q5 do not by
themselves identify physical QPU qubits. For a submitted physical circuit, use
the actual operand mapping and optionally constrain attention to relevant
qubits/couplings. For a logical circuit without a known placement, attention
would learn an association or distribution over candidate resources; it must
not be described as recovering the actual execution placement or routing.

The first bounded comparator, if separately authorized, should be simpler:

```text
Circuit encoder --> z_C
QPU coupling GNN --> z_H

[z_C || z_H || declared interaction features || shots/context]
                         |
                         v
                      runtime
```

Only consider cross-attention after a matched evaluation identifies a limitation
of simple fusion that operation-to-resource interaction could plausibly resolve.
No extra training, hardware GNN, attention model or experiment is run by adding
this note. Fair tests must separate additional hardware information from added
capacity and computation cost.

## Conditions for a scientifically interpretable test

- Record the circuit lifecycle: logical source or exact submitted physical
  circuit. A physical circuit's operand IDs must align with the device snapshot.
- Use only device information available at prediction time. Historical targets
  require appropriately aligned historical context; present-day snapshots do
  not establish the state of a past execution.
- Preserve snapshot dates, versions, hashes, units, missingness and provenance.
  Nominal contexts may be studied only with explicit proxy qualification.
- T1/T2 and errors are candidate predictors, not gate durations or established
  mechanisms of provider service time. Their runtime relevance must be tested.
- Prefer a crossed dataset with comparable circuits observed on multiple QPUs,
  shot settings and dates; otherwise source/device/workload confounding may
  prevent interpretation of learned device effects.
- Known-backend/new-circuit prediction and unseen-backend transfer are separate
  deployment questions. The latter needs held-out backends and sufficient
  independent device support; a QPU encoder alone does not prove portability.

## Minimum comparisons if separately authorized

1. Circuit + shots/context baseline.
2. Same circuit encoder + shots/context + backend-ID representation.
3. Same circuit encoder + shots/context + QPU descriptors and fusion.
4. A matched control that mismatches QPU descriptors while preserving the
   declared circuit/context comparisons, to assess whether aligned device
   information contributes.

Control model capacity, training budget, preprocessing and splits as far as
possible. Evaluate matched held-out targets with group/hash independence and
paired uncertainty. Account separately for compilation, context acquisition,
feature preparation and inference costs. No architecture sweep is implied.

## Relationship to current evidence

The idea addresses the lack of explicit shots/device conditioning in the
frozen Ma–Li-style circuit-only transfer input. It does not repair or replace
those results retroactively. It also does not establish topology causality,
recover actual hardware schedules, or make noisy-simulator runtime equivalent
to observed QPU execution/service time. Review existing context/physical-input
studies before considering any new experiment.
