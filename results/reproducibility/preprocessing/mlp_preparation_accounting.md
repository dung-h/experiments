> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# MLP preparation accounting

CONFIRMED:98.13402497% (reported98.13%). Numerator=217087022087ns. Denominator=221214835676ns. Same45 requests×30 repetitions=1350 exact ID/repeat units.

Numerator includes parsing,51-global extraction,frozen transform/tensor encoding and batch/H2D. Sum stages within each ID/repeat, then sum across1350 and divide by sum independently timed complete_B2. Denominator includes inference,host median,dispatcher/instrumentation residuals. Model loading and Graph-only DAG construction are excluded; MLP global extraction has its upstream structural traversals. This is pooled-time weighting,not equal-query mean of ratios.

Stage medians are summarized separately across query medians. Their sum need not equal median fullB2. Do not form this98.13% by dividing summed component medians by218.77019ms. Native scalar rows do not enter either sum. No historical timing record changed or new latency measured.

Source:artifacts/am1_phase2/v1/raw_stage_timings.csv, hash43a9bacc7accbfcdf81b0a82ae3c49d8104729e2b69f2f769f2d08b7018b3e8b. Independent arithmetic:mlp_accounting.json.
