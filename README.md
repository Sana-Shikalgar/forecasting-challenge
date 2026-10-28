# Cloud Workload Forecasting Challenge

AI/ML take-home challenge built from the **Microsoft Azure Public Dataset V2** VM trace.

## Start here

1. Read [`ASSIGNMENT.md`](ASSIGNMENT.md).
2. Read [`DATASET.md`](DATASET.md).
3. Download the **`data-v1` GitHub Release** dataset and extract it as `challenge_data/`.
4. Read [`SUBMISSION.md`](SUBMISSION.md).
5. Implement your analysis and forecasting pipeline.

The challenge dataset is distributed as a versioned GitHub Release rather than committed to the repository. The `data-v1` release contains the fixed curated dataset used for this assignment.

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

The repository includes a small test suite covering the starter forecasting utilities. Run it before beginning the assignment.

## Dataset download

The hiring dataset is distributed as the fixed `data-v1` release of this repository.

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

Verify the dataset before starting the assignment:

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

The release is a real, curated subset of the Microsoft Azure Public Dataset V2. It is not synthetic data.

The dataset contains 10,000 VMs with 134 CPU measurements per VM at five-minute intervals, for a total of 1,340,000 CPU observations.

Use the exact `data-v1` release provided for this assignment. Do not substitute another dataset or a different version of the Azure Public Dataset.

## Generate the predictions

With the environment from [Environment setup](#environment-setup) active (conda or venv, the commands are the same) and the data downloaded, run from the repository root:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This fits the models on everything before the last 24 timestamps only, writes `predictions.csv` (24 rows with `timestamp`, `active_vm_count` and `aggregate_cpu_utilization`), and only then scores the holdout once, for reporting. Add `--save-report` to also save the holdout table and figure under `reports/`. The run stops with an error message, and writes nothing, if the training series has a gap, a duplicate timestamp or a timestamp without any valid reading. It warns if ARIMA falls back to the baseline.

## Run the analysis notebooks

With either environment active, from the repository root:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_policy.ipynb
```

The EDA takes about half a minute, the modelling notebook about five minutes and the policy notebook about two.

## What you are solving

The underlying data contains anonymized VM workload telemetry sampled every five minutes. Your job is to understand the workload and build a defensible forecasting pipeline for near-term aggregate demand.

The challenge is intentionally open-ended. We evaluate reasoning, time-series methodology, leakage prevention, error analysis, reproducibility, and the quality of your operational conclusions—not just model complexity.

## Expected environment

Python 3.11+ is recommended. See [Environment setup](#environment-setup) for the conda and venv instructions.

Model dependencies are intentionally not prescribed. Add only what you need to `pyproject.toml` (the single source of truth for both setups) and document them.

## Source

Microsoft Azure Public Dataset V2:
https://github.com/Azure/AzurePublicDataset

The official V2 documentation describes a 30-day VM trace with five-minute CPU readings, VM metadata, and encrypted subscription/deployment/VM identifiers. The full upstream trace is much larger than this challenge release.

## Submission

Submit your work as a repository or archive according to [`SUBMISSION.md`](SUBMISSION.md).
