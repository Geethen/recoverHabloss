# Analysis script map

Use these versioned entry points for reproducible results:

| Script | Purpose | Main output |
|---|---|---|
| `extract_ppi_gee.py` | Extract labelled and predicted-only Zander values from Earth Engine | `data/ppi_gee/*.csv` |
| `compare_ppi_subsets.py` | Compare Olofsson, fixed PPI/PTD, and tuned PPI++ | `data/ppi_gee/comparison.csv` |
| `bootstrap_ppi_subsets.py` | Run the stratified PPI++ bootstrap | `data/ppi_gee/bootstrap_comparison.csv` |
| `analyse_ppi_variance.py` | Decompose analytic PPI++ variance by design stratum | `data/analysis_results/ppi_variance_*.csv` |
| `analyse_transition_composition.py` | Estimate weighted RECOVER/HABLOSS ratios and compositions | `data/analysis_results/transition_composition*.csv` |
| `extract_embeddings_gee.py` | Extract AlphaEarth 2018/2024 embeddings per spatial block | `data/embeddings/*.parquet` |
| `model_transitions.py` | Spatially blocked CV of the direct transition classifier | `data/analysis_results/transition_cv_*.csv` |

The embedding workflow runs in two stages. `extract_embeddings_gee.py` splits
the globally distributed plots into fixed degree blocks and extracts each block
to its own resumable Parquet shard; check the partition with `--dry-run` and
smoke test with `--max-blocks 2` before the full run. `model_transitions.py`
then predicts the `lc_2018 -> lc_2024` transition, holding out whole spatial
blocks so clustered plots cannot leak between train and test.

Shared estimator formulas are in `estimators.py`; shared design-based ratio and
area-shard readers are in `design_analysis.py`. All entry points expose their
input paths through `--help` and default to the project-drive locations listed
in the repository README.

The following files are historical exploratory scripts and are not canonical
reproduction entry points: `apply_weights_habloss.py`, `build_weights.py`,
`compare_recover.py`, `ppi_on_recover.py`, and `recover_weights.py`. They retain
old Linux paths and, in some cases, temporary scratch paths. Do not use their
printed values in reporting unless they have first been migrated to the
versioned workflow above.
