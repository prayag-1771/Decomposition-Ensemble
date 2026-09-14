# Decomposition-Ensemble

Code, data and results for the manuscript *Causal Evaluation of Decomposition
Ensemble Stock Index Forecasting: Reported Accuracy Depends on the Decomposition
Seeing the Future* (Sharma, Saxena, Srivastava and Tiwari, Vellore Institute of
Technology, Chennai).

The study reproduces a published ICEEMDAN + PSO-VMD + BiLSTM-attention
forecaster with a TCN head and a Tiny Recursive Model head, and evaluates it on
the daily highest and lowest prices of the S&P 500 and the SSEC under two
protocols that differ only in whether the decomposition may see the future:

* **Protocol A, full series decomposition**: the whole series is decomposed once
  and then split into training and test windows (the usual practice).
* **Protocol B, walk forward decomposition**: the training portion is decomposed
  once; for every test day the history up to the previous day is decomposed
  afresh, so no input ever contains a value at or after the forecast day.

Every number, table and figure in the paper is produced by the scripts here from
the files in `data/` and `results/`.

## Layout

| Path | Contents |
|---|---|
| `data/series_*.npy` | The four price series (S&P 500 and SSEC, highest and lowest), unadjusted daily values from Yahoo Finance, missing rows removed jointly |
| `data/onetime_*.npz`, `data/walkforward_*.npz` | Per-component training and test windows under each protocol, with `test_index` giving the day of every test row; `*_meta.json` records K, alpha, component count and timings |
| `data/_pre_parity_fix/` | The walk forward files as first produced, before the odd-length trimming defect described in the paper was corrected |
| `data/validation.json`, `data/_*.txt` | Self-test output and logs of the decomposition runs |
| `shared/decomp.py` | EMD, ICEEMDAN, VMD and the PSO search |
| `shared/walkforward.py` | The two protocols and the component-count reconciliation |
| `shared/models.py` | Encoder, TCN head, TRM head, bias-corrected weight averaging, training loop |
| `shared/evaluate.py` | Benchmarks, error measures, modified Diebold and Mariano test |
| `shared/parts.py` | Component-level work units used by the scheduler |
| `veer/run_decompositions.py` | Builds `data/` from the cached series |
| `prayag/run_training.py` | Trains one network per component (worker mode shares one GPU between processes) and assembles forecast files |
| `ayush/run_evaluation.py` | Benchmarks, model-versus-benchmark comparison, leakage inflation, summary figures |
| `run_all.py` | Checks the data files, runs a synthetic smoke test, trains everything and evaluates, resumable |
| `tools/fix_walkforward_parity.py` | Recomputes the walk forward windows that ended one day early |
| `results/pred_*.npz` | Forecast, truth, per-component forecasts and `test_index` for each protocol, series and head |
| `results/parts/` | Every individual component model's forecasts (262 files) |
| `results/minmax_scaling_run/` | The archived run under the source model's min-max scaling, referred to in the paper's audit table |
| `results/*.json`, `results/fig_*.png` | Evaluation outputs |
| `paper/` | Manuscript sources, the scripts that build the tables and figures, the reference records and the checks |

## Reproducing the study

Python 3.11 or later with `numpy`, `scipy`, `matplotlib`, `yfinance` and a CUDA
build of PyTorch (the runs reported used torch 2.6.0 with CUDA 12.4 on a single
NVIDIA GeForce GTX 1650).

```bash
# 1. decompositions (CPU; about 25 to 40 minutes per walk forward series)
cd veer
python run_decompositions.py --task validate
python run_decompositions.py --task onetime
python run_decompositions.py --task walkforward --realizations 20 --stride 1
cd ..
python tools/fix_walkforward_parity.py          # windows must end at t-1

# 2. check, smoke test, then train all 262 networks and evaluate (GPU; about 3 hours)
python run_all.py --check
python run_all.py --smoke --fused --workers 5
python run_all.py --fused --workers 5

# 3. tables, figures and numbers for the paper
cd paper
python leakage_diagnostic.py     # input diagnostic, Table 3, Fig. 1
python analyze.py                # Tables 4 to 8, Figs. 2 and 3, results_numbers.json
python repro_numbers.py          # counts quoted in the reproducibility subsection
python check_numbers.py          # every number in the prose against the result files
python make_refs.py main.tex     # reference list from publisher records
python build_main.py             # assembles main.tex and runs the structural checks
```

`data/` and `results/` already contain the outputs of every step, so any step can
be run on its own. `run_all.py` skips component models that are already saved.

## Compiling the manuscript

`paper/build_main.py` writes `paper/main.tex`, which cites `paper/refs.bib`
through the Springer Nature class `sn-jnl` (option `sn-mathphys-num`). The class
and bibliography style are not included here; download the official template
package from the link in `paper/template/SOURCE.txt` and place `sn-jnl.cls` and
`sn-mathphys-num.bst` next to `main.tex`, together with `paper/figs/`.
`paper/main_selfcontained.tex` is the same manuscript with the reference list
written out.

## Data source

Daily highest and lowest prices of the S&P 500 (`^GSPC`) and the Shanghai Stock
Exchange Composite Index (`000001.SS`) over ten years, downloaded once from Yahoo
Finance and cached in `data/series_*.npy`. The cached arrays are the ones used in
every experiment; a fresh download would extend the ten-year window and change
the split.
