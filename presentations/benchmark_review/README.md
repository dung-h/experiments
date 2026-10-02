# Quantum runtime benchmark review deck

This is the revised 14-slide internal review deck. The editable PowerPoint is
the source deliverable; the PDF is a 14-page visual export of its rendered
slides. Speaker notes link the evidence used for each result.

The deck keeps archived real-QPU service-time prediction separate from local
simulator prediction. It covers the 8,767-row QPU ledger and its circuit
provenance, the 204-member simulator panel, methods and their status, the
source-paired QPU comparison, separate Aer and fixed-MPS predictor findings,
native diagnostics, unavailable routes, and release/reproducibility limits.
QPack inputs are marked reconstructed; Qonductor's 230 archive-resolved
logical recipes are not described as byte-exact logical QASM. Local Azizov-
style and fixed-MPS Family-Aware-style predictors are included as adaptations,
not paper-complete reproductions.

## Rebuild

Run from the repository root with the installed presentation runtime:

```bash
env \
  SKILL_DIR=/home/server/.codex/plugins/cache/openai-primary-runtime/presentations/26.923.10815/skills/presentations \
  RUNTIME_NODE_MODULES=/home/server/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
  RUNTIME_PYTHON=/home/server/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 \
  /home/server/.nvm/versions/node/v24.19.0/bin/node presentations/source/build_deck.mjs
```

The builder refuses to overwrite the PowerPoint. It reads the frozen scorecard
and figures under `presentations/figures/` and `presentations/figures_v2/`;
those inputs and the slide brief are linked in the speaker notes. It does not
train a model, run a simulator, or modify benchmark result artifacts.

To render the finished deck to slide images for visual inspection:

```bash
env RUNTIME_NODE_MODULES=/home/server/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
  /home/server/.nvm/versions/node/v24.19.0/bin/node \
  /home/server/.codex/plugins/cache/openai-primary-runtime/presentations/26.923.10815/skills/presentations/container_tools/render_presentation.mjs \
  --input presentations/benchmark_review/quantum_runtime_benchmark_vi.pptx \
  --output_dir work/presentation-deck-review/rendered_final --scale 1
```

## QA and limits

- 14 slides; 16:9 layout; editable tables and the native QPack chart.
- Presentation-package integrity and layout validators passed with no
  warnings. Every slide was rendered and visually inspected.
- Predictor figures use separate Aer and MPS target clocks. Their bootstrap
  intervals are exploratory and do not support blanket method-superiority
  claims.
- PDF: 14 pages at 640 × 360 pt, exported from 1280 × 720 rendered slides.
  Use the PPTX when editing is needed.
- The complete review package, including this deck, passed the fresh local
  clean-clone inventory/table-rebuild checks at commit `860829a`. This remains
  internal review material, not public-release or licensing approval.
