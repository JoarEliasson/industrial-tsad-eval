# Architecture

Industrial TSAD Eval uses a hexagonal architecture to keep product logic
independent from command-line rendering and filesystem details.

## Layers

- `domain` contains stable contracts and pure evaluation behavior.
- `ports` defines the interfaces that application services depend on.
- `application` coordinates use cases and owns workflow-level decisions.
- `infrastructure` implements local repositories and artifact writers.
- `plugins` provides dataset adapter and detector implementations through registries.
- `interfaces/cli` is the only layer that imports Typer or Rich.

## Dependency Rules

- Domain code imports no application, infrastructure, plugin, or interface code.
- Application code depends on domain, ports, and selected infrastructure adapters.
- Plugins implement ports and are discovered through registries.
- CLI code performs argument parsing and rendering only.
- Core code raises Python/domain exceptions; CLI code translates them to exit codes.
- Dataset preparation writes to a staging directory, validates Prepared Format v1,
  then promotes the result. Adapters never delete existing outputs directly.

## Dataset Preparation Flow

```mermaid
flowchart LR
  A["CLI or caller"] --> B["PrepareDataset use case"]
  B --> C["DatasetAdapterRegistry"]
  C --> D["DatasetAdapterPlugin"]
  D --> E["Prepared writer helpers"]
  E --> F["Staging prepared dataset"]
  B --> G["ValidatePreparedDataset"]
  G --> H["Promoted prepared dataset"]
```

The adapter owns source-specific parsing. The application service owns plugin
lookup, staging, overwrite policy, validation, and promotion.

## Benchmark Slice

Benchmark orchestration is in-process. `RunBenchmark` loads a resolved TOML
config, expands datasets x detectors x protocols, validates prepared datasets,
then calls the existing scoring and evaluation use cases.

Benchmark runs consume Prepared Format directories only. Raw-data preparation
stays explicit through `itse prepared prepare`, which keeps benchmark runs
repeatable and avoids hidden data mutation.
