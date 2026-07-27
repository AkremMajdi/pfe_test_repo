# Tunisia Wildfire Risk Dashboard

This package is a ready-to-deploy Streamlit app that aggregates the two wildfire models developed in the notebooks:

1. **Strategic annual fire severity prediction at governorate level**
   - Source notebook: `First_model (1).ipynb`
   - Source dataset: `refined_tunisia_wildfire_dataset.csv`
   - Purpose preserved: annual governorate-level Low / Medium / High severity classification, plus annual fire-count and burned-area regression.

2. **Operational 500 m pixel-level daily fire-risk ranking**
   - Source workflow: the operational pixel model developed earlier in this chat.
   - Purpose preserved: daily 500 m proxy-pixel risk ranking and alert allocation using `target_fire_1d` and `case_control_weight`.

3. **Annual–pixel validation bridge**
   - The bridge does not change either model. It aggregates daily pixel alert pressure to governorate-year level and compares it with the annual strategic severity model.
   - This was implemented directly in the Streamlit app because only one new annual notebook and one annual dataset were uploaded with the latest request.

## What is included

```text
streamlit_wildfire_app/
  app.py
  README.md
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

## Important modelling scope

The app intentionally keeps the two model purposes separate:

- The **annual model** is strategic and governorate-level. It is suitable for annual planning, severity review, and governorate comparison.
- The **operational pixel model** is tactical and daily. It is suitable for ranking candidate 500 m proxy pixels and generating alert lists.
- The current pixel model is still a **historical-fire-cell proxy model**, not a full national burnable-landscape probability model. Treat outputs as relative alert rankings unless the full 500 m burnable grid is later substituted.

## Minimum changes made

The app was built with minimum necessary changes to make the notebooks deployable:

- Converted notebook logic into reusable Python functions.
- Preserved the annual model's risk-score formula: `0.6 * normalized_fire_count + 0.4 * normalized_burned_area`.
- Preserved the annual model's random-forest classifier and random-forest regressors.
- Preserved the operational model's target `target_fire_1d`, sample weighting, feature set, calibrated proxy probability, and alert-budget approach.
- Added a small quality fix to the annual severity binning: the lower severity threshold uses `<= q33` rather than `< q33`. In this dataset `q33 = 0`, so the original strict comparison would remove the Low class from training.
- Added a bridge table joining annual severity with daily pixel alert pressure by governorate and year.

## Run locally

From inside the app folder:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

You can deploy the whole folder to Streamlit Community Cloud, a VPS, Docker, or any internal server that can run Python.

For Streamlit Community Cloud:

1. Push the folder contents to a GitHub repository.
2. Set `app.py` as the Streamlit entry point.
3. Make sure `requirements.txt` is in the repository root or app folder used by Streamlit.
4. Keep the `data/` and `models/` folders in the same directory as `app.py`.

## Optional email alert system

The dashboard can send the current alert table by email if Streamlit secrets are configured.

Create `.streamlit/secrets.toml` with:

```toml
[alerts]
email_enabled = true
smtp_host = "smtp.example.com"
smtp_port = 587
smtp_user = "your_user@example.com"
smtp_password = "your_password_or_app_password"
sender = "alerts@example.com"
recipients = ["recipient@example.com"]
```

Do not hardcode credentials in `app.py`.

## Dashboard pages

### Strategic annual severity

Shows:

- Annual severity classification by governorate.
- Predicted fire count.
- Predicted burned area.
- Annual model metrics.

### Operational pixel alerts

Shows:

- Daily 500 m proxy-pixel risk map.
- Top 1%, 5%, or 10% alert budgets.
- Downloadable alert list.
- Optional email alert delivery.
- Optional upload-and-score workflow for a new daily feature table matching the model schema.

### Annual–pixel bridge

Shows:

- Governorate-year comparison between annual severity and daily alert pressure.
- Critical / high / watch pixel-day counts aggregated by year.
- Observed fire pixel-days in the proxy dataset.

### Model monitoring

Shows:

- Annual and pixel model metrics.
- Pixel feature importance.
- Score distributions by year.
- Scope and data-quality warnings.

## Production next step

To make the operational page fully production-grade, replace the current proxy pixel universe with a daily feature table generated from the full national 500 m burnable grid. The app already supports scoring an uploaded feature table as long as it contains the same model feature columns.
