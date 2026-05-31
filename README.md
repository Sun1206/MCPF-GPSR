# MCPF-GPSR ICDM 2026 Reproducibility Package

This folder contains the code, data snapshots, saved parent predictions, available trained checkpoints, and result CSV files used by the paper:

**MCPF-GPSR: Graph-Periodic Surplus Residual Calibration for Public Graph Time-Series Forecasting**

All commands below are written relative to this package root:

```powershell
cd artifacts\mcpf_gpsr_icdm2026_repro_package
```

## 1. Package Layout

```text
code/
  bints_parent/                         BINTS parent source snapshot
  rast_parent/                          RAST parent source snapshot
  bints_mcpf_extended_calibrate.py       MCPF-GPSR calibration and residual-basis selection
  analyze_bints_extended_results.py      BINTS aggregation, paired tests, grouped-view summaries
  analyze_bints_graph_ablation.py        graph-control aggregation
  rast_train_sanitized.py                RAST training launcher
  rast_export_mcpf_npz.py                RAST prediction exporter

data/
  raw/bints/                             BINTS raw folders and adjacency metadata used in the paper
  raw/rast/                              PEMS04 and PEMS08 snapshots used by the RAST transfer check
  parent_predictions/bints/              saved BINTS validation/test predictions consumed by MCPF-GPSR
  results/                               CSV files used by all paper tables

models/
  bints/                                 available trained BINTS parent checkpoints
  rast_logs/                             retained RAST train/export/calibration logs

paper_resources/figures/                 editable draw.io workflow and exported PDF/PNG
```

## 2. Environment

The MCPF-GPSR calibration layer is CPU-friendly because it consumes saved parent predictions. Parent training uses GPU when available.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas scikit-learn torch tqdm
```

For RAST parent training, install the dependencies required by the included `code\rast_parent` snapshot.

## 3. Fast Path: Reproduce Paper Tables from Saved Predictions

The paper tables can be regenerated from the included saved BINTS prediction files under `data\parent_predictions\bints` without retraining the parent model.

### 3.1 Run MCPF-GPSR on BINTS saved predictions

```powershell
New-Item -ItemType Directory -Force data\results\calibrated_bints | Out-Null

$jobs = @(
  @{npz="covid_seq4_pred7_khop5.npz";       dataset="covid";     cycle=1;  nodes=16; out="covid_pred7_extended.csv"},
  @{npz="covid_seq4_pred14_khop5.npz";      dataset="covid";     cycle=1;  nodes=16; out="covid_pred14_extended.csv"},
  @{npz="covid_seq4_pred30_khop5.npz";      dataset="covid";     cycle=1;  nodes=16; out="covid_pred30_extended.csv"},
  @{npz="nyc_seq4_pred7_khop5.npz";         dataset="nyc";       cycle=24; nodes=10; out="nyc_pred7_extended.csv"},
  @{npz="nyc_seq4_pred14_khop5.npz";        dataset="nyc";       cycle=24; nodes=10; out="nyc_pred14_extended.csv"},
  @{npz="nyc_seq4_pred30_khop5.npz";        dataset="nyc";       cycle=24; nodes=10; out="nyc_pred30_extended.csv"},
  @{npz="nyc_covid_seq4_pred7_khop5.npz";   dataset="nyc_covid"; cycle=1;  nodes=5;  out="nyc_covid_pred7_extended.csv"},
  @{npz="nyc_covid_seq4_pred14_khop5.npz";  dataset="nyc_covid"; cycle=1;  nodes=5;  out="nyc_covid_pred14_extended.csv"},
  @{npz="nyc_covid_seq4_pred30_khop5.npz";  dataset="nyc_covid"; cycle=1;  nodes=5;  out="nyc_covid_pred30_extended.csv"}
)

foreach ($j in $jobs) {
  python code\bints_mcpf_extended_calibrate.py `
    --npz "data\parent_predictions\bints\$($j.npz)" `
    --dataset $j.dataset `
    --cycle $j.cycle `
    --nodes $j.nodes `
    --adj-root data\raw\bints\adj_matrix `
    --out "data\results\calibrated_bints\$($j.out)"
}
```

Aggregate the calibrated rows into the paper-level BINTS tables:

```powershell
python code\analyze_bints_extended_results.py `
  --root data\results\calibrated_bints `
  --out data\results
```

### 3.2 Run graph-control ablations

The graph-control ablation compares the declared graph-periodic residual basis against identity, permuted, and random graph controls.

```powershell
$graphJobs = @(
  @{npz="covid_seq4_pred7_khop5.npz";  dataset="covid"; cycle=1;  nodes=16; tag="covid_pred7"},
  @{npz="covid_seq4_pred14_khop5.npz"; dataset="covid"; cycle=1;  nodes=16; tag="covid_pred14"},
  @{npz="covid_seq4_pred30_khop5.npz"; dataset="covid"; cycle=1;  nodes=16; tag="covid_pred30"},
  @{npz="nyc_seq4_pred7_khop5.npz";    dataset="nyc";   cycle=24; nodes=10; tag="nyc_pred7"},
  @{npz="nyc_seq4_pred14_khop5.npz";   dataset="nyc";   cycle=24; nodes=10; tag="nyc_pred14"},
  @{npz="nyc_seq4_pred30_khop5.npz";   dataset="nyc";   cycle=24; nodes=10; tag="nyc_pred30"}
)

foreach ($j in $graphJobs) {
  foreach ($mode in @("identity","permute","random")) {
    New-Item -ItemType Directory -Force "data\results\graph_ablation_$mode" | Out-Null
    python code\bints_mcpf_extended_calibrate.py `
      --npz "data\parent_predictions\bints\$($j.npz)" `
      --dataset $j.dataset `
      --cycle $j.cycle `
      --nodes $j.nodes `
      --adj-root data\raw\bints\adj_matrix `
      --graph-mode $mode `
      --graph-seed 0 `
      --out "data\results\graph_ablation_$mode\$($j.tag)_$mode.csv"
  }
}

python code\analyze_bints_graph_ablation.py `
  --root data\results `
  --out data\results
```

## 4. Reproduce Parent Predictions

### 4.1 BINTS parent training and prediction export

The included NPZ files are sufficient for table reproduction. To regenerate them from the BINTS parent:

```powershell
Copy-Item data\raw\bints\* code\bints_parent\datasets -Recurse -Force
cd code\bints_parent
$env:BINTS_SAVE_PRED_DIR = "..\..\data\parent_predictions\bints"

foreach ($dataset in @("covid","nyc","nyc_covid")) {
  foreach ($pred in @(7,14,30)) {
    foreach ($seed in @(0,1,2)) {
      python main.py `
        --gpu_id 0 `
        --batch_size 64 `
        --dataset $dataset `
        --seq_day 4 `
        --pred_day $pred `
        --khop 5 `
        --seed $seed `
        --num_epochs 10 `
        --model_save_path ..\..\models\bints
    }
  }
}

cd ..\..
```

The stronger-parent check in the paper uses the same command with `--num_epochs 30` for the selected COVID/NYC-COVID cells.

### 4.2 RAST parent training, prediction export, and calibration

Train RAST parent models:

```powershell
python code\rast_train_sanitized.py `
  --rast-root code\rast_parent `
  --config src\scripts\main\RAST_PEMS04.py `
  --gpus 0

python code\rast_train_sanitized.py `
  --rast-root code\rast_parent `
  --config src\scripts\main\RAST_PEMS08.py `
  --gpus 0
```

Export validation/test predictions to MCPF-GPSR NPZ format:

```powershell
New-Item -ItemType Directory -Force data\parent_predictions\rast | Out-Null

python code\rast_export_mcpf_npz.py `
  --rast-root code\rast_parent `
  --config src\scripts\main\RAST_PEMS04.py `
  --out data\parent_predictions\rast\pems04_seed0.npz `
  --gpus 0 `
  --device-type gpu `
  --batch-size 128

python code\rast_export_mcpf_npz.py `
  --rast-root code\rast_parent `
  --config src\scripts\main\RAST_PEMS08.py `
  --out data\parent_predictions\rast\pems08_seed0.npz `
  --gpus 0 `
  --device-type gpu `
  --batch-size 128
```

Calibrate RAST forecasts:

```powershell
python code\bints_mcpf_extended_calibrate.py `
  --npz data\parent_predictions\rast\pems04_seed0.npz `
  --parent-name RAST `
  --dataset pems04 `
  --nodes 307 `
  --out data\results\rast_pems04_seed0_mcpf.csv

python code\bints_mcpf_extended_calibrate.py `
  --npz data\parent_predictions\rast\pems08_seed0.npz `
  --parent-name RAST `
  --dataset pems08 `
  --nodes 170 `
  --out data\results\rast_pems08_seed0_mcpf.csv
```

The final package includes the RAST result CSVs and retained run logs under `models\rast_logs`.

## 5. Figure and Table Provenance

| Paper item | Source artifact(s) | Reproduction command |
|---|---|---|
| Fig. 1 workflow | `paper_resources\figures\mcpf_gpsr_workflow.drawio`, `paper_resources\figures\mcpf_gpsr_workflow.pdf` | Edit/export with draw.io; the submission TeX includes `figures\mcpf_gpsr_workflow.pdf`. |
| Algorithm 1 | `submission_package\paper\mcpf_bints_icdm2026.tex` | Compile the paper from the submission package. |
| Table I: frozen-parent graph forecasting views | `data\raw\bints\*`, `data\raw\rast\*` | Dataset metadata and view definitions are reported directly in the paper. |
| Table II: public BINTS graph forecasting results | `data\results\combined_three_view_rows.csv` | Section 3.1 commands, then `python code\analyze_bints_extended_results.py --root data\results\calibrated_bints --out data\results`. |
| Table III: matched-cell BINTS consistency | `data\results\extended_pairwise_tests.csv` | Same aggregation command as Table II. |
| Table IV: large-view absolute metric means | `data\results\combined_three_view_rows.csv` | Same aggregation command as Table II. |
| Table V: residual-source decomposition | `data\results\residual_source_decomposition.csv` | Same aggregation command as Table II; rows are grouped by residual family. |
| Table VI: single-forecast and grouped-view checks | `data\results\single_forecast_deployment.csv`, `data\results\grouped_view_bootstrap.csv` | Same aggregation command as Table II. |
| Table VII: GPSR graph ablation | `data\results\graph_ablation_gpsr_by_mode_dataset.csv` | Section 3.2 graph-control commands, then `python code\analyze_bints_graph_ablation.py --root data\results --out data\results`. |
| Table VIII: paired graph-control tests | `data\results\graph_ablation_pairwise_tests.csv` | Same graph-ablation aggregation command as Table VII. |
| Table IX: transfer across parent families | `data\results\metricaware_gate_summary.csv`, `data\results\rast_mcpf_summary.csv`, `data\results\rast_pems04_e30_mcpf.csv`, `data\results\rast_pems08_e30_mcpf.csv` | RAST commands in Section 4.2 and stronger-parent BINTS commands in Section 4.1. |

## 6. Included Result CSVs

The checked-in result CSVs are the exact table inputs used by the final paper:

- `combined_three_view_rows.csv`
- `extended_best_structured_vs_generic.csv`
- `extended_pairwise_tests.csv`
- `extended_rows_nyc_covid.csv`
- `graph_ablation_gpsr_by_mode_dataset.csv`
- `graph_ablation_pairwise_tests.csv`
- `grouped_view_bootstrap.csv`
- `metricaware_gate_summary.csv`
- `rast_mcpf_summary.csv`
- `rast_pems04_e30_mcpf.csv`
- `rast_pems08_e30_mcpf.csv`
- `residual_source_decomposition.csv`
- `single_forecast_deployment.csv`

## 7. Notes on Models and Data

- `models\bints\*.pth` contains the available trained BINTS parent checkpoints.
- BINTS saved prediction NPZ files are the direct inputs to MCPF-GPSR and let the BINTS tables be regenerated without retraining the parent.
- RAST checkpoints are regenerated by the training command because the RAST/BasicTS runner writes them inside its run directory; this package retains the RAST logs and final calibrated result CSVs.
- The raw data folders included here are the data snapshots used by the submitted paper.
