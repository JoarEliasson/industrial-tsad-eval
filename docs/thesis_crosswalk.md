# Thesis Crosswalk

This repo preserves the thesis result families while changing the internal
architecture from an evolved harness to explicit ports, services, and artifact
repositories.

| Thesis-era area | Productized implementation |
| --- | --- |
| Prepared data validation | Prepared Format v1 domain contracts and repositories |
| Dataset-specific raw conversion | Dataset adapter plugins behind `PrepareDataset` |
| Data localization/fetching | Dataset source plugins behind `AcquireDatasetSource` |
| Detector scoring | Detector plugins plus Score Contract v1 |
| Event detection metrics | `EvaluateScores` and `RunBenchmark` |
| Multi-run experiments | TOML benchmark orchestration |
| System/machine reporting | `RunPreflight` and `ProfileScoreEvaluate` |
| Evidence and XAI evaluation | Evidence Bundle v1, GT tag maps, `EvaluateEvidence` |
| RQ3 assistant experiment | Provider-backed replay suites, retrieval, planner artifacts, referee metrics |
| Operator-readable output | Optional deterministic operator cards |

The old `tsad.thesis` and `tsad.operator_assistant` modules are reference
material only. The new implementation keeps thesis-compatible artifact and
metric families, but places them behind cleaner interfaces.
