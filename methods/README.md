# Method source index

| Location | Role | Execution boundary |
| --- | --- | --- |
| [real_qpu](real_qpu/README.md) | Extraction/reconstruction/fitting/analysis | SUPPORTING / PRIVATE-BOUNDARY REFERENCE; excluded labels/partitions/model states needed |
| simulator | Context-specific implementations | SUPPORTING; historical dependencies/input contracts, not measurement authorization |
| reporting | Reporting implementation | SUPPORTING; current public checks avoid private producing paths |
| [research_support](research_support/README.md) | Exact AM1/compatibility/closure source and tests | HISTORICAL source-audit layer; original development imports preserved |

The [taxonomy](../results/thesis/tables/a_method_taxonomy.csv) distinguishes native
components, adaptations, proxies, calibrated bridges and diagnostics. Source
presence is not a portable public refit path or permission to run an old queue.
Use [reproduction scope](../docs/reproduction.md) and [public tests](../tests/README.md) first.
