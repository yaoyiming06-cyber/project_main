# Competition inference and timing repair

Goal: apply the audit fixes authorized by the user; retain unrelated working changes and do not commit.

Architecture: a shared folder-driven inference runner is called by predict and cost. It validates the image-only model bundle, scans IDs from paths without metadata CSVs, serializes each Study result, and includes fixed ensemble/TTA/patient smoothing in timed work. Model initialization and a distinct warmup folder are excluded. Study aggregation and patient dependencies are reported honestly rather than claimed as independent single-image latency.

- [ ] Add failing tests for submission probability tolerance, threshold/ranking invariance, duplicate/missing/invalid labels, folder inventory, distinct warmup, shared runner and per-round serialization.
- [ ] Extract existing prediction/loading helpers without changing legacy prediction behavior.
- [ ] Implement shared deployment bundle loading, deterministic image-only folder inference and fixed patient context.
- [ ] Replace cost CLI with official positional inputs plus independent warmup, three repeats, strict comparison and nonzero failures. Report precision, artifact sizes/hashes, actual sample counts and timing limitations.
- [ ] Add predict entry points, usage documentation and bundle template.
- [ ] Run regression suite, fixture end-to-end CLI, inventory official samples and review changes.

Acceptance: no true ViewPosition or clinical metadata read; serial image encoding; full-precision probability comparison with <=1e-3 tolerance, identical binary decisions and per-class ordering including ties; final CSV write and all final postprocessing timed; mismatch fails; incomplete 100-image sample or unresolved multi-image timing explicitly marked diagnostic. Existing checkpoints and historical submissions remain untouched.
