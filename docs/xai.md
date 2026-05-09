# XAI Evaluation

XAI evaluation scores Evidence Bundle v1 artifacts against a ground-truth tag
map and deterministic masking proxies.

```powershell
itse xai gt-map build --prepared examples/generated/OPCUA_SYNTH --out out/gt_map.json
itse xai gt-map validate --gt-map out/gt_map.json
itse xai eval --prepared examples/generated/OPCUA_SYNTH --evidence out/evidence --gt-map out/gt_map.json --out out/xai --ks 1,3,5
```

## Metrics

- `HitRate@K`: whether any top-K variable overlaps the GT tag set.
- `Recall@K`: fraction of GT tags recovered by the top-K variables.
- Masking proxy drops: robust z-score change after replacing top variables or
  top windows with train/validation medians.
- Stability: adjacent Jaccard overlap of local top-variable rankings.

The masking scorer is built in and safe. The toolkit does not execute arbitrary
scorer scripts. When evidence comes from native explainers, XAI metrics evaluate
those native rankings; masking remains the deterministic robust surrogate used
to make score-drop comparisons reproducible across detector families.

## Layout

```text
<xai_out>/
  metrics.json
  bundle_metrics.csv
  summary.csv
  skipped.json
```
