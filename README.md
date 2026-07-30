# Tunisia Wildfire Risk Dashboard

This repository contains a ready-to-deploy Streamlit application for wildfire risk analysis in Tunisia. The app combines two complementary modelling layers:

1. **Strategic annual fire-severity prediction at governorate level**
2. **Operational daily 500 m pixel-risk ranking and alerting**

A validation bridge connects the annual governorate-level severity layer with the daily pixel-level alert layer by aggregating daily pixel alerts to governorate-year summaries.

## Application overview

The dashboard provides:

- Annual fire-severity predictions by governorate
- Daily 500 m pixel-level fire-risk ranking
- Critical, High, and Watch alert tiers
- Interactive map-based alert exploration
- Annual-to-daily validation bridge summaries
- Model performance and feature-importance views
- Downloadable alert tables and prediction outputs

## Repository layout

The app supports two layouts.

### Option A — flat repository layout

Place all files directly in the repository root:

```text
app.py
requirements.txt
refined_tunisia_wildfire_dataset.csv
annual_governorate_predictions.csv
annual_governorate_metrics.json
annual_governorate_model_bundle.joblib
final_trainable_500m_wildfire_dataset_model_aligned.csv
operational_pixel_scores_all.csv
operational_pixel_proxy_feature_importance.csv
operational_pixel_proxy_metrics.json
operational_pixel_proxy_model_bundle.joblib
bridge_annual_pixel_alerts.csv
```

The app also supports the filename variant:

```text
refined_tunisia_wildfire_dataset(1).csv
```

### Option B — organized repository layout

You may also organize files into folders:

```text
app.py
requirements.txt

data/
  refined_tunisia_wildfire_dataset.csv
  annual_governorate_predictions.csv
  annual_governorate_metrics.json
  final_trainable_500m_wildfire_dataset_model_aligned.csv
  operational_pixel_scores_all.csv
  operational_pixel_proxy_feature_importance.csv
  operational_pixel_proxy_metrics.json
  bridge_annual_pixel_alerts.csv

models/
  annual_governorate_model_bundle.joblib
  operational_pixel_proxy_model_bundle.joblib
```

The application automatically searches both the repository root and the `data/` and `models/` folders.

## Required files

At minimum, the following files are needed for deployment:

```text
app.py
requirements.txt
refined_tunisia_wildfire_dataset.csv
final_trainable_500m_wildfire_dataset_model_aligned.csv
operational_pixel_proxy_model_bundle.joblib
operational_pixel_proxy_feature_importance.csv
operational_pixel_proxy_metrics.json
```

The following files are recommended because they make startup faster:

```text
annual_governorate_predictions.csv
annual_governorate_metrics.json
annual_governorate_model_bundle.joblib
operational_pixel_scores_all.csv
bridge_annual_pixel_alerts.csv
```

If the recommended annual artifacts are missing, the app can rebuild them from the annual source dataset. If `operational_pixel_scores_all.csv` is missing, the app can generate it from the pixel model bundle and the model-aligned pixel feature table.

## Local installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Run the app locally:

```bash
streamlit run app.py
```

## Streamlit Cloud deployment

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy the app.

If large files are used, commit them with Git LFS and confirm that the actual model and data files are available in the deployed repository, not only LFS pointer files.

## Dashboard pages

### 1. Strategic annual severity

This page displays annual fire-severity predictions at governorate level. It keeps the strategic annual model focused on long-term severity assessment, including risk class, predicted annual fire count, and predicted burned area.

### 2. Operational pixel alerts

This page displays daily 500 m pixel-risk scores and ranking-based alert tiers. Alert tiers are assigned by daily score percentile:

| Tier | Rule | Intended use |
|---|---|---|
| Critical | Top 1% of scored pixels | Highest-priority alerts |
| High | Top 5% of scored pixels | Operational alert layer |
| Watch | Top 10% of scored pixels | Wider monitoring layer |
| Normal | Remaining scored pixels | Routine monitoring |

The pixel model is designed for relative risk ranking. Scores should be interpreted as alert-ranking values rather than fully calibrated national fire probabilities.

### 3. Annual–pixel bridge

This page aggregates daily pixel alerts to governorate-year level and compares them with the strategic annual severity predictions. The bridge does not change either model; it provides an integrated validation and interpretation layer.

### 4. Model diagnostics

This page summarizes model metrics, score distributions, and feature importance for monitoring and interpretation.

## Model scope and interpretation

The system includes two model scopes:

- The annual model supports strategic governorate-level planning.
- The daily pixel model supports short-term alert prioritization over the currently available 500 m pixel proxy universe.

The daily pixel model currently scores a historical-fire-cell proxy dataset rather than a complete nationwide burnable-land grid. For this reason, daily pixel scores should be used for ranking and alert prioritization, not as absolute nationwide probability estimates.

## Data and model notes

- The operational pixel model uses `target_fire_1d` as the target.
- The model-aligned pixel dataset uses `case_control_weight` during model development.
- Wind direction was excluded due to inconsistent availability.
- Precipitation gaps were handled through training-period seasonal imputation with missingness indicators.
- Prior-fire history was treated as censored historical information rather than ordinary missing data.
- Raw FIRMS detection metadata is not used as a predictor in the operational pixel model.

## Updating the data

To update the app with new scoring data:

1. Prepare a new daily feature file with the same schema as `final_trainable_500m_wildfire_dataset_model_aligned.csv`.
2. Place the file in the repository root or the `data/` folder.
3. Use the scoring page in the app to generate daily scores, or replace `operational_pixel_scores_all.csv` with the updated scored output.
4. Redeploy the app.

## Troubleshooting

### FileNotFoundError during deployment

Check that all required files are committed to the repository and that filenames match the expected names. The app can read files from either the repository root or the `data/` and `models/` folders.

### ModuleNotFoundError

Confirm that `requirements.txt` is present in the repository root and that all dependencies are listed.

### Large file problems

Use Git LFS for large `.csv` or `.joblib` files. After pushing, verify that the deployed repository contains the actual file contents.

### Empty map or missing alerts

Confirm that the scored pixel file contains latitude and longitude columns:

```text
pixel_centroid_lat_proxy
pixel_centroid_lon_proxy
```

Also confirm that the selected date and governorate contain scored pixel records.

## Limitations

The current app is suitable for demonstration, validation, and operational prototyping. A fully calibrated nationwide pixel-level fire probability system requires a complete burnable-land grid, pixel-specific live vegetation features, operational weather forecasts, and daily scoring over all eligible 500 m cells.


## Strategic model v2 update

The annual governorate severity model has been updated to improve the 2025 strategic audit. The update keeps the original annual governorate purpose but applies a more stable severity definition and more conservative outbreak-aware prediction logic.

Key changes:

- Replaces the unstable governorate-min/max risk label with a global robust log severity score: 60% fire count and 40% burned area.
- Preserves Low/Medium/High severity classes using fixed operational thresholds on the robust score.
- Removes satellite-derived leakage fields from predictors.
- Adds causal lag, rolling, and historical-capacity features by governorate.
- Uses a log-scale ensemble for burned-area prediction.
- Uses a conservative historical-capacity floor for annual fire-count prediction to reduce systematic under-alerting in outbreak years.

The 2025 rows are used as an audit set in the metrics file. The deployable model bundle is retrained on all available annual records through 2025 for future forecasts.
