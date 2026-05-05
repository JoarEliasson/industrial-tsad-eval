# Dataset Acquisition

Dataset acquisition is the raw-data boundary before Prepared Format conversion.
It helps users copy, unpack, and optionally download raw datasets into a local
cache with provenance.

Acquisition does not prepare data. After acquisition, run `itse prepared prepare`
explicitly against the returned raw directory.

## Commands

```powershell
itse data sources
itse data describe --source swat
itse data acquire --source swat --method manual --manual data/downloads/SWaT --out data/raw
itse data validate --source swat --raw data/raw/SWaT

itse prepared prepare --dataset swat --raw data/raw/SWaT --out prepared
```

The output root stores one dataset directory:

```text
data/raw/
  SWaT/
    raw_provenance.json
    ...
```

Existing dataset directories are refused unless `--overwrite` is supplied.
Acquisition writes into `<out>/.staging` first, then promotes the finished raw
directory.

## Supported Sources

- `tep`: `manual`, `mathworks-http`, optional `kaggle`.
- `swat`: `manual`, optional `kaggle`.
- `hai`: `manual`, optional `kaggle`, optional `git`.
- `hai-cpps`: `manual`.

Manual acquisition accepts a directory, supported archive, or single file. Safe
archive unpacking rejects path traversal and archive links.

Optional Kaggle support is installed with:

```powershell
python -m pip install -e ".[acquisition]"
```

Kaggle credentials are handled by Kaggle tooling, not by this package. The
toolkit does not load `.env` files, store tokens, or manage credentials.

## Provenance

Every acquisition writes `raw_provenance.json`:

```json
{
  "contract_version": "raw-provenance-v1",
  "source_name": "swat",
  "dataset_name": "SWaT",
  "method": "manual",
  "file_count": 2,
  "files": [
    {
      "path": "SWaT_Dataset_Normal.csv",
      "size_bytes": 1234,
      "sha256": "..."
    }
  ]
}
```

The file inventory is relative to the raw root and excludes the provenance file
itself.

## Dataset Notes

TEP can use local CSV folders or user-provided HTTP(S) archives. `mathworks-http`
requires `--ref` to point at the resource to download.

SWaT official access is request-gated through iTrust. Use `manual` for files you
have already obtained through approved channels. Kaggle mirrors are optional
user-provided refs.

HAI supports local files, optional Kaggle refs, and optional public git mirrors.
Git acquisition clones into a `repository/` subdirectory.

HAI-CPPS is manual-only in this phase. Simulator execution, Docker workflows,
and IEEE/request-gated downloads stay outside the productized acquisition layer.
