# Cloud Workload Forecasting Challenge

Forecast the last two hours (24 five-minute points) of `active_vm_count` and `aggregate_cpu_utilization` from the first 110 points of the Microsoft Azure Public Dataset V2 VM trace, and turn the forecast into a capacity recommendation.

AI assistance: Claude Code was used as an AI assistant for planning and coding.

## Summary

- `active_vm_count` is constant at 10,000 in the whole trace, so the forecast is the last observed value.
- `aggregate_cpu_utilization` is forecast with ARIMA(1,0,0) plus the hourly profile of the training data. A random forest on the correction of that baseline is the machine learning comparison.
- Model choice uses rolling-origin cross-validation on the 110 training points only. The final 24 values are never used for fitting, selection or tuning, and are scored once after `predictions.csv` is written.
- Cross-validation MAE (percentage points of CPU): baseline 0.194, ARIMA 0.179, random forest 0.177. The gain over the baseline is about 8% and the two models are not separable. There are only about 2.6 independent 24-step windows in 110 points.
- Holdout MAE, reported once: ARIMA 0.091, random forest 0.123, baseline 0.161. The count forecast is exact.
- Capacity policy: forecast plus a margin taken from past forecast errors, applied only when recent load is in the top 20% of the history so far. The margin is the smallest one that keeps at most 5% of high-load values above capacity in cross-validation.

The full write-up is in [`reports/final_report.md`](reports/final_report.md), with a PDF copy in `reports/final_report.pdf`.

## Repository layout

```text
forecasting-challenge/
├── README.md
├── pyproject.toml            # dependencies and package metadata
├── environment.yml           # conda wrapper around pyproject.toml
├── predictions.csv           # submitted forecast, 24 rows
├── ASSIGNMENT.md             # task documents, unchanged
├── DATASET.md
├── DATA_CARD_TEMPLATE.md
├── DATA_RELEASE.md
├── SOURCES.md
├── SUBMISSION.md
├── CODE_OF_CONDUCT.md
├── LICENSE
├── challenge_data/           # data release, git-ignored (README only is tracked)
├── src/cwfc/
│   ├── download.py           # download and verify the data release
│   ├── data.py               # load readings, build target series, split the holdout
│   ├── analysis.py           # shared statistics and plot helpers
│   ├── features.py           # past-only features and the hourly profile
│   ├── models.py             # baseline, ARIMA, random forest, reference forecasters
│   ├── evaluate.py           # rolling-origin cross-validation and metrics
│   ├── policy.py             # error analysis, intervals, headroom policy
│   ├── predict.py            # command line that writes predictions.csv
│   ├── baselines.py          # starter file: persistence and seasonal naive
│   ├── metrics.py            # starter file: MAE, RMSE, MAE ratio
│   └── validation.py         # starter file: rolling-origin helpers
├── notebooks/
│   ├── 01_eda.ipynb          # data checks and workload patterns
│   ├── 02_modeling.ipynb     # validation design, models, comparison
│   └── 03_policy.ipynb       # error analysis, intervals, headroom
├── reports/
│   ├── final_report.md       # final report (markdown)
│   ├── final_report.pdf      # final report (PDF, 7 page body plus appendices)
│   ├── figures/              # figures saved by the notebooks and predict.py
│   └── tables/               # tables saved by the notebooks and predict.py
└── tests/                    # pytest suite, one file per module
```

## Environment setup

Python **3.11 or newer** is required. Both options below install the same packages: all dependencies are declared in `pyproject.toml`, and `environment.yml` only pins Python and delegates to pip.

### Option A: conda (Miniconda / Anaconda)

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate atomity
```

### Option B: venv

Create and activate a virtual environment:

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Verify the installation

With either option active:

```bash
python -m pytest -q
```

The suite has 414 tests covering data checks, leakage, models, evaluation, the policy and the prediction command. It takes about one minute. One test uses the real data and is skipped until `python -m cwfc.download` has been run.

## Dataset download

The dataset is the fixed `data-v1` release of this repository.

**Download:** [Azure Dataset v1 Release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)

### Automatic download (recommended)

With the environment active, run from the repository root:

```bash
python -m cwfc.download
```

This downloads `data-v1.tar.gz`, extracts it into `challenge_data/`, and verifies `SHA256SUMS`. It does nothing if the verified files are already there. From Python: `from cwfc.download import download_data`.

### Manual download

Download `data-v1.tar.gz` from the release and extract it into the repository as `challenge_data/`:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

The resulting layout is:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

Verify the dataset before use:

```bash
cd challenge_data
sha256sum -c SHA256SUMS
```

You should see:

```text
vm_cpu_readings.csv.gz: OK
vm_metadata.csv.gz: OK
DATA_CARD.md: OK
```

The dataset contains 10,000 VMs with 134 CPU measurements per VM at five-minute intervals, for a total of 1,340,000 CPU observations.

## Generate the predictions

With the environment active and the data downloaded, run from the repository root:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This fits the models on everything before the last 24 timestamps only, writes `predictions.csv` (24 rows with `timestamp`, `active_vm_count` and `aggregate_cpu_utilization`), and only then scores the holdout once, for reporting. Add `--save-report` to also save the holdout table and figure under `reports/`.

The timestamps in the file are the raw holdout timestamps. The run stops with an error message, and writes nothing, if the training series has a gap, a duplicate timestamp or a timestamp without any valid reading. It warns if ARIMA falls back to the baseline. The command does not depend on the supplied files: it works on any readings file with the same columns.

## Run the analysis notebooks

With either environment active, from the repository root:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_policy.ipynb
```

The EDA takes about half a minute, the modelling notebook about five minutes and the policy notebook about two. Model and policy logic lives in `src/cwfc/`; the notebooks call it and save their tables and figures under `reports/`. A fresh conda environment (Python 3.11) reproduced `predictions.csv` and every notebook result exactly. A fresh venv on Python 3.15 did the same except for floating-point noise and a few ARIMA tuning rows, which are explained in Appendix E of the report.

## Method

- **Targets:** both series are built per timestamp from the VM readings. CPU is the aggregate over VMs, and invalid readings are excluded.
- **Validation:** an expanding window with 39 rolling origins (48 to 86 history points) and a 24-step direct horizon. Features use only data available at the forecast origin, and tests check that changing future values never changes an earlier forecast.
- **Models:** a baseline (history mean plus hourly profile), ARIMA(1,0,0) with the profile as an exogenous regressor, and a random forest per forecast step that learns the baseline correction. Any failed ARIMA fit falls back to the baseline.
- **Metrics:** MAE, RMSE, MAE ratio to the baseline, error by horizon step, and error on high-load values.
- **Policy:** prediction intervals from absolute forecast errors, and a headroom rule that adds a margin only when a peak-risk trigger is on. At the end of the training data the trigger is off, so the plan is the forecast itself.

## Notes on the starter code

`baselines.py`, `metrics.py` and `validation.py` come from the starter. They are kept and used, with docstrings, input checks and an `origin_positions` helper added. In `validation.py`, `rolling_origin_splits` and `TimeSplit` are kept as provided but are not used by the pipeline, which splits by index position instead. The unused starter `smape` function was removed.

## Data source and attribution

The data is a curated subset of the Microsoft Azure Public Dataset V2 (https://github.com/Azure/AzurePublicDataset). Attribution to the Azure Public Dataset is retained, and its current license and usage terms apply. See [`SOURCES.md`](SOURCES.md) and the `DATA_CARD.md` in the release. No external dataset is used.

The release is a stratified sample, so category and core-bucket shares reflect the sampling design and not the real Azure population.

## Submission

The submission follows [`SUBMISSION.md`](SUBMISSION.md): `predictions.csv` has exactly the columns `timestamp,active_vm_count,aggregate_cpu_utilization` and 24 rows, and the command above reproduces it from the release data.
