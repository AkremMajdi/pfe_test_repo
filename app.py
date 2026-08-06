"""
Streamlit dashboard for Tunisia wildfire modelling.

This single app aggregates two modelling workflows while preserving their original goals:
1. Strategic annual fire-severity prediction at governorate level.
2. Operational daily 500 m pixel-risk ranking with an alert system.

The bridge layer links annual governorate severity with daily pixel alert pressure and reports proxy coverage explicitly.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Iterable
import json
import smtplib
from email.message import EmailMessage

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    import pydeck as pdk
except ImportError:  # Streamlit can still display a simpler st.map fallback.
    pdk = None
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

# The original package used /data and /models folders. Streamlit Cloud users often
# flatten files into the repository root. These helpers make the app work in both
# layouts without changing model behavior.
DATA_DIR = APP_DIR / "data" if (APP_DIR / "data").exists() else APP_DIR
MODELS_DIR = APP_DIR / "models" if (APP_DIR / "models").exists() else APP_DIR


def find_artifact(names: str | Iterable[str], artifact_type: str = "data") -> Path:
    """Return the first existing artifact across supported repository layouts.

    Search order supports:
    - original package layout: data/<file>, models/<file>
    - flattened repo layout: <file> at repository root
    - common upload suffixes such as refined_tunisia_wildfire_dataset(1).csv

    If no file exists yet, return the preferred output path so derived artifacts
    can be created beside the other files.
    """
    if isinstance(names, str):
        names = [names]
    names = list(names)
    preferred_dir = MODELS_DIR if artifact_type == "model" else DATA_DIR
    search_dirs = []
    for directory in [preferred_dir, APP_DIR, APP_DIR / "data", APP_DIR / "models"]:
        if directory not in search_dirs:
            search_dirs.append(directory)

    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return preferred_dir / names[0]


ANNUAL_DATA_PATH = find_artifact(
    [
        "refined_tunisia_wildfire_dataset.csv",
        "refined_tunisia_wildfire_dataset(1).csv",
        "refined_tunisia_wildfire_dataset (1).csv",
    ],
    "data",
)
ANNUAL_BUNDLE_PATH = find_artifact("annual_governorate_model_bundle.joblib", "model")
ANNUAL_PREDICTIONS_PATH = find_artifact("annual_governorate_predictions.csv", "data")
ANNUAL_METRICS_PATH = find_artifact("annual_governorate_metrics.json", "data")

PIXEL_DATA_PATH = find_artifact(
    [
        "final_trainable_500m_wildfire_dataset_model_aligned.csv",
        "final_trainable_500m_wildfire_dataset.csv",
        "wildfire_pixel_proxy_500m.csv",
    ],
    "data",
)
PIXEL_BUNDLE_PATH = find_artifact("operational_pixel_proxy_model_bundle.joblib", "model")
PIXEL_SCORES_PATH = find_artifact("operational_pixel_scores_all.csv", "data")
PIXEL_METRICS_PATH = find_artifact("operational_pixel_proxy_metrics.json", "data")
PIXEL_IMPORTANCE_PATH = find_artifact("operational_pixel_proxy_feature_importance.csv", "data")
BRIDGE_PATH = find_artifact("bridge_annual_pixel_alerts.csv", "data")

RANDOM_SEED = 42
RISK_CLASS_ORDER = ["Low", "Medium", "High"]
ALERT_TIERS = ["Critical", "High", "Watch", "Monitor"]
ALERT_FRACTIONS = {"Critical — top 1%": 0.01, "High — top 5%": 0.05, "Watch — top 10%": 0.10}

BRIDGE_PIXEL_METRIC_COLUMNS = [
    "scored_pixel_days",
    "observed_fire_pixel_days",
    "mean_raw_risk_score",
    "max_raw_risk_score",
    "critical_alert_pixel_days",
    "high_or_critical_alert_pixel_days",
    "watch_or_above_alert_pixel_days",
]

LIGHT_RISK_SCALE = [
    [0.00, "#EAF4FF"],  # very light blue
    [0.35, "#FFF4C2"],  # light yellow
    [0.70, "#FFD7A8"],  # light orange
    [1.00, "#F7B7B7"],  # light red
]


@dataclass
class AnnualStudyConfig:
    """Same purpose as the uploaded annual notebook, with deployable paths."""

    study_name: str = "tunisia_fire_thesis"
    train_years: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)
    test_years: tuple[int, ...] = (2023, 2024, 2025)
    target_fire_count: str = "fire_count"
    target_burned_area: str = "area_burned_ha"
    severity_target: str = "risk_class"
    weight_fire_count: float = 0.6
    weight_burned_area: float = 0.4
    excluded_vars: tuple[str, ...] = ("brightness", "total_frp", "confidence", "satellite_fire_count")


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def load_json(path: Path) -> dict:
    require_file(path, "JSON artifact")
    return json.loads(path.read_text(encoding="utf-8"))


def format_percent(x: float | int | None, decimals: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{100 * float(x):.{decimals}f}%"


def format_number(x: float | int | None, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):,.{decimals}f}"


def class_to_level(value: str) -> int:
    mapping = {"Low": 1, "Medium": 2, "High": 3}
    return mapping.get(str(value), np.nan)


def add_alert_columns(frame: pd.DataFrame, alert_fraction: float) -> pd.DataFrame:
    """Flag the top fraction of rows as alerts within the current filtered universe."""
    out = frame.copy()
    if out.empty:
        out["is_selected_alert"] = False
        return out
    n_alerts = max(1, int(np.ceil(len(out) * alert_fraction)))
    out = out.sort_values("raw_risk_score", ascending=False).reset_index(drop=True)
    out["rank_for_selected_budget"] = np.arange(1, len(out) + 1)
    out["is_selected_alert"] = out["rank_for_selected_budget"] <= n_alerts
    return out


def normalize_bridge_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Make a bridge table safe for charts and explicit about proxy coverage.

    The annual model covers 24 governorates, whereas the current pixel proxy covers
    only governorates represented by historical FIRMS cells. A left join therefore
    produces missing pixel metrics for some governorate-years. Those are coverage
    gaps, not chart-ready numeric values.
    """
    out = frame.copy()
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")

    for column in BRIDGE_PIXEL_METRIC_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["has_pixel_proxy_coverage"] = out["scored_pixel_days"].fillna(0).gt(0)
    out["pixel_coverage_status"] = np.where(
        out["has_pixel_proxy_coverage"],
        "Pixel proxy available",
        "Not represented in current pixel proxy",
    )

    count_columns = [
        "scored_pixel_days",
        "observed_fire_pixel_days",
        "critical_alert_pixel_days",
        "high_or_critical_alert_pixel_days",
        "watch_or_above_alert_pixel_days",
    ]
    for column in count_columns:
        out[column] = out[column].fillna(0).clip(lower=0)

    for column in ["mean_raw_risk_score", "max_raw_risk_score"]:
        out[column] = out[column].fillna(0.0).clip(lower=0.0)

    # Plotly marker sizes must be finite and non-negative. Keep uncovered rows
    # visible with a small marker while preserving the true observed count in hover.
    out["bridge_marker_size"] = np.maximum(out["observed_fire_pixel_days"], 1.0)
    return out


def prepare_pixel_map_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate map coordinates and prepare visible risk/alert markers."""
    out = frame.copy()
    numeric_columns = [
        "pixel_centroid_lat_proxy",
        "pixel_centroid_lon_proxy",
        "raw_risk_score",
        "calibrated_proxy_probability",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(
        subset=["pixel_centroid_lat_proxy", "pixel_centroid_lon_proxy", "raw_risk_score"]
    ).copy()
    out = out[
        out["pixel_centroid_lat_proxy"].between(30.0, 38.5)
        & out["pixel_centroid_lon_proxy"].between(7.0, 12.5)
    ].copy()

    if out.empty:
        return out

    out["raw_risk_score"] = out["raw_risk_score"].clip(lower=0.0, upper=1.0)
    out["selected_alert_label"] = np.where(
        out["is_selected_alert"].fillna(False),
        "Selected alert",
        "Scored pixel",
    )
    # Alerts are visibly larger; all non-alert pixels remain visible after filters.
    out["map_marker_size"] = np.where(out["is_selected_alert"].fillna(False), 18.0, 8.0)
    return out


def map_zoom_for_frame(frame: pd.DataFrame) -> float:
    """Choose a useful zoom for country, governorate, and small filtered views."""
    if frame.empty:
        return 6.0
    lat_span = float(frame["pixel_centroid_lat_proxy"].max() - frame["pixel_centroid_lat_proxy"].min())
    lon_span = float(frame["pixel_centroid_lon_proxy"].max() - frame["pixel_centroid_lon_proxy"].min())
    span = max(lat_span, lon_span)
    if span < 0.15:
        return 10.0
    if span < 0.40:
        return 8.5
    if span < 1.00:
        return 7.5
    if span < 2.50:
        return 6.5
    return 5.6


def render_pixel_risk_map(map_df: pd.DataFrame, selected_date, selected_gov: str) -> None:
    """Render filtered risk scores on a reliable, light Plotly basemap."""
    if map_df.empty:
        st.warning("No valid coordinates are available for the selected filters.")
        return

    center = {
        "lat": float(map_df["pixel_centroid_lat_proxy"].mean()),
        "lon": float(map_df["pixel_centroid_lon_proxy"].mean()),
    }
    zoom = map_zoom_for_frame(map_df)

    hover_data = {
        "pixel_centroid_lat_proxy": ":.4f",
        "pixel_centroid_lon_proxy": ":.4f",
        "raw_risk_score": ":.3f",
        "calibrated_proxy_probability": ":.5f",
        "map_marker_size": False,
        "is_selected_alert": True,
        "target_fire_1d": True,
    }

    try:
        fig = px.scatter_mapbox(
            map_df,
            lat="pixel_centroid_lat_proxy",
            lon="pixel_centroid_lon_proxy",
            color="raw_risk_score",
            size="map_marker_size",
            size_max=18,
            color_continuous_scale=LIGHT_RISK_SCALE,
            range_color=(0.0, 1.0),
            hover_name="gouvernorat",
            hover_data=hover_data,
            custom_data=[
                "pixel_id_500m",
                "alert_tier",
                "selected_alert_label",
                "rank_for_selected_budget",
            ],
            center=center,
            zoom=zoom,
            opacity=0.82,
            title=f"Predicted wildfire risk and selected alerts — {selected_date} — {selected_gov}",
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Pixel: %{customdata[0]}<br>"
                "Raw risk score: %{marker.color:.3f}<br>"
                "Model tier: %{customdata[1]}<br>"
                "Budget status: %{customdata[2]}<br>"
                "Budget rank: %{customdata[3]}<extra></extra>"
            )
        )
        fig.update_layout(
            mapbox_style="carto-positron",
            height=620,
            margin={"r": 0, "t": 55, "l": 0, "b": 0},
            coloraxis_colorbar={"title": "Predicted risk", "tickformat": ".2f"},
            legend_title_text="Map status",
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
    except Exception as exc:
        # Do not let a map-provider/rendering issue take down the page.
        st.warning(f"Interactive map rendering fell back to the basic map: {exc}")
        fallback = map_df.rename(
            columns={
                "pixel_centroid_lat_proxy": "lat",
                "pixel_centroid_lon_proxy": "lon",
            }
        )
        st.map(fallback[["lat", "lon"]], use_container_width=True)


# -----------------------------------------------------------------------------
# Annual governorate model: preserved from uploaded notebook with one quality fix
# -----------------------------------------------------------------------------
def build_annual_model_bundle(
    source_path: Path = ANNUAL_DATA_PATH,
    bundle_path: Path = ANNUAL_BUNDLE_PATH,
    predictions_path: Path = ANNUAL_PREDICTIONS_PATH,
    metrics_path: Path = ANNUAL_METRICS_PATH,
) -> dict:
    """
    Train the strategic annual governorate model.

    This follows the uploaded notebook's intent and model family:
    - annual governorate panel,
    - risk score = 60% normalized fire count + 40% normalized burned area,
    - Low/Medium/High severity classes,
    - random-forest classifier plus count/area random-forest regressors.

    Minimum quality adjustment:
    The original threshold used `score < q33` for Low. In this dataset q33 can be 0,
    which removes the Low class from training. The deployable app uses `score <= q33`
    so zero-severity years remain Low rather than being shifted into Medium.
    """
    require_file(source_path, "annual source dataset")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cfg = AnnualStudyConfig()
    wf = pd.read_csv(source_path)
    wf = wf.drop(columns=[c for c in cfg.excluded_vars if c in wf.columns])

    def compute_risk_score(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        train_mask = group["year"].isin(cfg.train_years)
        c_min = group.loc[train_mask, cfg.target_fire_count].min()
        c_max = group.loc[train_mask, cfg.target_fire_count].max()
        a_min = group.loc[train_mask, cfg.target_burned_area].min()
        a_max = group.loc[train_mask, cfg.target_burned_area].max()
        c_norm = (group[cfg.target_fire_count] - c_min) / (c_max - c_min) if c_max > c_min else 0.0
        a_norm = (group[cfg.target_burned_area] - a_min) / (a_max - a_min) if a_max > a_min else 0.0
        group["risk_score"] = cfg.weight_fire_count * c_norm + cfg.weight_burned_area * a_norm
        return group

    panel = wf.groupby("gouvernorat", group_keys=False).apply(compute_risk_score)
    train_scores = panel.loc[panel["year"].isin(cfg.train_years), "risk_score"]
    q33 = float(train_scores.quantile(0.33))
    q66 = float(train_scores.quantile(0.66))

    def classify_risk(score: float) -> str:
        if pd.isna(score):
            return "Unknown"
        if score <= q33:
            return "Low"
        if score < q66:
            return "Medium"
        return "High"

    panel[cfg.severity_target] = panel["risk_score"].apply(classify_risk)
    panel = panel.sort_values(["gouvernorat", "year"]).reset_index(drop=True)
    panel["fire_count_lag1"] = panel.groupby("gouvernorat")[cfg.target_fire_count].shift(1)
    panel["area_burned_lag1"] = panel.groupby("gouvernorat")[cfg.target_burned_area].shift(1)
    panel["heat_drought_index"] = panel["tmax"] / (panel["prcp"] + 1)

    feature_columns = [
        "forest_area_ha",
        "other_forest_area_ha",
        "tmax",
        "prcp",
        "wspd",
        "NDVI",
        "NDMI",
        "NBR",
        "fire_count_lag1",
        "area_burned_lag1",
        "heat_drought_index",
    ]
    feature_columns = [c for c in feature_columns if c in panel.columns]

    train_df = panel[panel["year"].isin(cfg.train_years)].copy()
    test_df = panel[panel["year"].isin(cfg.test_years)].copy()

    label_encoder = LabelEncoder()
    label_encoder.fit(panel[cfg.severity_target].astype(str))

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train_cls = label_encoder.transform(train_df[cfg.severity_target].astype(str))
    y_test_cls = label_encoder.transform(test_df[cfg.severity_target].astype(str))

    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    fire_count_regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_leaf=5,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    area_burned_regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_leaf=5,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    classifier.fit(X_train, y_train_cls)
    fire_count_regressor.fit(X_train, train_df[cfg.target_fire_count])
    area_burned_regressor.fit(X_train, train_df[cfg.target_burned_area])

    cls_pred = classifier.predict(X_test)
    count_pred = fire_count_regressor.predict(X_test)
    area_pred = area_burned_regressor.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test_cls, cls_pred)),
        "macro_f1": float(f1_score(y_test_cls, cls_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test_cls, cls_pred, average="weighted", zero_division=0)),
        "fire_count_r2": float(r2_score(test_df[cfg.target_fire_count], count_pred)),
        "fire_count_mae": float(mean_absolute_error(test_df[cfg.target_fire_count], count_pred)),
        "area_burned_r2": float(r2_score(test_df[cfg.target_burned_area], area_pred)),
        "area_burned_mae": float(mean_absolute_error(test_df[cfg.target_burned_area], area_pred)),
        "risk_score_q33": q33,
        "risk_score_q66": q66,
        "train_years": list(cfg.train_years),
        "test_years": list(cfg.test_years),
        "features": feature_columns,
        "label_encoder_classes": list(label_encoder.classes_),
        "classifier_training_classes": list(map(int, classifier.named_steps["model"].classes_)),
        "labeling_adjustment": "Lower severity bin uses <= q33 so zero-risk years remain Low when q33 is 0.",
    }

    probabilities = classifier.predict_proba(panel[feature_columns])
    panel["predicted_risk_class"] = label_encoder.inverse_transform(classifier.predict(panel[feature_columns]))
    for class_name in label_encoder.classes_:
        panel[f"prob_{class_name}"] = 0.0
    for column_idx, encoded_class in enumerate(classifier.named_steps["model"].classes_):
        class_name = label_encoder.inverse_transform([encoded_class])[0]
        panel[f"prob_{class_name}"] = probabilities[:, column_idx]
    panel["predicted_fire_count"] = fire_count_regressor.predict(panel[feature_columns])
    panel["predicted_area_burned_ha"] = area_burned_regressor.predict(panel[feature_columns])
    panel.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bundle = {
        "config": asdict(cfg),
        "classifier": classifier,
        "fire_count_regressor": fire_count_regressor,
        "area_burned_regressor": area_burned_regressor,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "risk_quantiles": {"q33": q33, "q66": q66},
        "notes": "Derived from the uploaded annual notebook with minimum deployability changes.",
    }
    joblib.dump(bundle, bundle_path)
    return bundle


def assign_alert_tiers(scores: pd.Series) -> pd.Series:
    """Assign global risk tiers using score percentiles.

    This preserves the original dashboard concept: alerts are relative rankings,
    not absolute calibrated nationwide probabilities.
    """
    if scores.empty:
        return pd.Series([], dtype=object)
    q99 = scores.quantile(0.99)
    q95 = scores.quantile(0.95)
    q90 = scores.quantile(0.90)
    return pd.Series(
        np.select(
            [scores >= q99, scores >= q95, scores >= q90],
            ["Critical", "High", "Watch"],
            default="Monitor",
        ),
        index=scores.index,
    )


def build_pixel_scores(
    source_path: Path = PIXEL_DATA_PATH,
    bundle_path: Path = PIXEL_BUNDLE_PATH,
    scores_path: Path = PIXEL_SCORES_PATH,
) -> pd.DataFrame:
    """Create precomputed pixel scores if they were not uploaded.

    This is a deployment convenience only; it does not retrain the pixel model.
    It uses the trained bundle and the model-aligned pixel feature table.
    """
    require_file(source_path, "model-aligned pixel dataset")
    require_file(bundle_path, "operational pixel model bundle")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(bundle_path)
    feature_columns = bundle["feature_columns"]
    frame = pd.read_csv(source_path, parse_dates=["prediction_date"])
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise FileNotFoundError(
            "The pixel dataset exists, but it is not model-aligned. "
            f"Missing required feature columns: {missing[:20]}"
        )

    raw = bundle["model"].predict_proba(frame[feature_columns])[:, 1]
    scored = frame.copy()
    scored["raw_risk_score"] = raw
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        scored["calibrated_proxy_probability"] = calibrator.predict(raw)
    else:
        scored["calibrated_proxy_probability"] = raw

    scored["year"] = pd.to_datetime(scored["prediction_date"]).dt.year
    scored["alert_tier"] = assign_alert_tiers(scored["raw_risk_score"])

    keep = [
        "pixel_id_500m",
        "prediction_date",
        "target_fire_1d",
        "case_control_weight",
        "selection_probability",
        "sampling_stratum",
        "data_split",
        "spatial_fold",
        "candidate_mask_scope",
        "operational_landscape_ready",
        "gouvernorat",
        "pixel_centroid_lat_proxy",
        "pixel_centroid_lon_proxy",
        "year",
        "raw_risk_score",
        "calibrated_proxy_probability",
        "alert_tier",
    ]
    keep = [c for c in keep if c in scored.columns]
    scored[keep].to_csv(scores_path, index=False)
    return scored[keep]


def build_bridge_dataset(
    bridge_path: Path = BRIDGE_PATH,
) -> pd.DataFrame:
    """Aggregate daily pixel alert pressure to governorate-year level.

    This is the validation bridge requested by the user: it links the strategic
    annual severity model with operational daily pixel alerts without changing
    either model's purpose.
    """
    annual = load_annual_predictions()
    pixel = load_pixel_scores()

    agg = (
        pixel.groupby(["gouvernorat", "year"], dropna=False)
        .agg(
            scored_pixel_days=("pixel_id_500m", "count"),
            observed_fire_pixel_days=("target_fire_1d", "sum"),
            mean_raw_risk_score=("raw_risk_score", "mean"),
            max_raw_risk_score=("raw_risk_score", "max"),
            critical_alert_pixel_days=("alert_tier", lambda s: int((s == "Critical").sum())),
            high_or_critical_alert_pixel_days=("alert_tier", lambda s: int(s.isin(["Critical", "High"]).sum())),
            watch_or_above_alert_pixel_days=("alert_tier", lambda s: int(s.isin(["Critical", "High", "Watch"]).sum())),
        )
        .reset_index()
    )

    annual_columns = [
        "gouvernorat",
        "year",
        "risk_class",
        "predicted_risk_class",
        "risk_score",
        "predicted_fire_count",
        "predicted_area_burned_ha",
    ]
    annual_columns = [c for c in annual_columns if c in annual.columns]
    bridge = annual[annual_columns].merge(agg, on=["gouvernorat", "year"], how="left")
    bridge = normalize_bridge_frame(bridge)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bridge.to_csv(bridge_path, index=False)
    return bridge


# -----------------------------------------------------------------------------
# Data/model loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_annual_bundle() -> dict:
    if not ANNUAL_BUNDLE_PATH.exists():
        return build_annual_model_bundle()
    return joblib.load(ANNUAL_BUNDLE_PATH)


@st.cache_resource(show_spinner=False)
def load_pixel_bundle() -> dict:
    require_file(PIXEL_BUNDLE_PATH, "operational pixel model bundle")
    return joblib.load(PIXEL_BUNDLE_PATH)


@st.cache_data(show_spinner=False)
def load_annual_predictions() -> pd.DataFrame:
    if not ANNUAL_PREDICTIONS_PATH.exists():
        build_annual_model_bundle()
    annual = pd.read_csv(ANNUAL_PREDICTIONS_PATH)
    annual["severity_level"] = annual["risk_class"].map(class_to_level)
    annual["predicted_severity_level"] = annual["predicted_risk_class"].map(class_to_level)
    return annual


@st.cache_data(show_spinner=False)
def load_pixel_scores() -> pd.DataFrame:
    if not PIXEL_SCORES_PATH.exists():
        build_pixel_scores()
    scores = pd.read_csv(PIXEL_SCORES_PATH, parse_dates=["prediction_date"])
    scores["date"] = scores["prediction_date"].dt.date
    return scores


@st.cache_data(show_spinner=False)
def load_bridge() -> pd.DataFrame:
    required_columns = {
        "gouvernorat",
        "year",
        "predicted_risk_class",
        "high_or_critical_alert_pixel_days",
        "observed_fire_pixel_days",
    }
    if not BRIDGE_PATH.exists():
        return normalize_bridge_frame(build_bridge_dataset())

    bridge = pd.read_csv(BRIDGE_PATH)
    if not required_columns.issubset(bridge.columns):
        return normalize_bridge_frame(build_bridge_dataset())
    return normalize_bridge_frame(bridge)


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    require_file(PIXEL_IMPORTANCE_PATH, "pixel feature importance")
    return pd.read_csv(PIXEL_IMPORTANCE_PATH)


@st.cache_data(show_spinner=False)
def load_model_metrics() -> tuple[dict, dict]:
    if not ANNUAL_METRICS_PATH.exists():
        build_annual_model_bundle()
    annual_metrics = load_json(ANNUAL_METRICS_PATH)
    pixel_metrics = load_json(PIXEL_METRICS_PATH)
    return annual_metrics, pixel_metrics


# -----------------------------------------------------------------------------
# Optional external alert delivery
# -----------------------------------------------------------------------------
def send_email_alert(subject: str, body: str, alert_df: pd.DataFrame) -> tuple[bool, str]:
    """
    Optional email sender using Streamlit secrets.

    Expected .streamlit/secrets.toml:
    [alerts]
    email_enabled = true
    smtp_host = "smtp.example.com"
    smtp_port = 587
    smtp_user = "user@example.com"
    smtp_password = "..."
    sender = "alerts@example.com"
    recipients = ["recipient@example.com"]
    """
    try:
        alerts = st.secrets.get("alerts", {})
    except Exception:
        alerts = {}

    if not alerts or not bool(alerts.get("email_enabled", False)):
        return False, "Email sending is not configured. Add [alerts] settings to Streamlit secrets."

    required = ["smtp_host", "smtp_port", "smtp_user", "smtp_password", "sender", "recipients"]
    missing = [key for key in required if key not in alerts]
    if missing:
        return False, f"Email secrets are incomplete. Missing: {', '.join(missing)}"

    recipients = alerts["recipients"]
    if isinstance(recipients, str):
        recipients = [r.strip() for r in recipients.split(",") if r.strip()]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = alerts["sender"]
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)
    msg.add_attachment(
        alert_df.to_csv(index=False).encode("utf-8"),
        maintype="text",
        subtype="csv",
        filename="wildfire_alerts.csv",
    )

    try:
        with smtplib.SMTP(alerts["smtp_host"], int(alerts["smtp_port"])) as smtp:
            smtp.starttls()
            smtp.login(alerts["smtp_user"], alerts["smtp_password"])
            smtp.send_message(msg)
        return True, f"Email alert sent to {len(recipients)} recipient(s)."
    except Exception as exc:
        return False, f"Email failed: {exc}"


# -----------------------------------------------------------------------------
# UI rendering
# -----------------------------------------------------------------------------
def render_header() -> None:
    st.set_page_config(
        page_title="Tunisia Wildfire Risk Dashboard",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .small-note {font-size: 0.86rem; color: #666;}
        .metric-caption {font-size: 0.82rem; color: #666; margin-top: -0.75rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🔥 Tunisia Wildfire Risk Dashboard")
    st.caption(
        "Strategic annual governorate severity + operational 500 m daily pixel-risk ranking. "
        "The pixel model currently ranks historically observed fire cells, not the full national burnable landscape."
    )


def render_status_cards() -> None:
    annual = load_annual_predictions()
    pixel = load_pixel_scores()
    annual_metrics, pixel_metrics = load_model_metrics()
    test_raw = next((m for m in pixel_metrics.get("test_metrics", []) if m.get("prediction") == "raw_risk_score"), {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annual records", f"{len(annual):,}")
    c2.metric("Pixel-date scores", f"{len(pixel):,}")
    c3.metric("Annual accuracy", format_percent(annual_metrics.get("accuracy"), 1))
    c4.metric("Pixel test ROC-AUC", format_number(test_raw.get("roc_auc"), 3))


def render_annual_page() -> None:
    annual = load_annual_predictions()
    annual_metrics, _ = load_model_metrics()

    st.header("Strategic annual fire severity — governorate level")
    st.write(
        "This view preserves the uploaded annual model purpose: predict Low/Medium/High annual severity by governorate, "
        "plus annual fire-count and burned-area regressions."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", format_percent(annual_metrics.get("accuracy"), 1))
    c2.metric("Macro F1", format_number(annual_metrics.get("macro_f1"), 3))
    c3.metric("Fire-count MAE", format_number(annual_metrics.get("fire_count_mae"), 1))
    c4.metric("Burned-area MAE", format_number(annual_metrics.get("area_burned_mae"), 1))

    with st.expander("Annual model notes", expanded=False):
        st.json(annual_metrics)
        st.info(
            "A minimal quality fix was applied: the lower severity threshold uses <= q33. "
            "In the uploaded dataset q33 is 0, so the original strict '< q33' rule would remove the Low class from training."
        )

    selected_year = st.selectbox("Year", sorted(annual["year"].unique()), index=len(sorted(annual["year"].unique())) - 1)
    year_df = annual[annual["year"] == selected_year].copy()

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(
            year_df.sort_values("risk_score", ascending=False),
            x="gouvernorat",
            y="risk_score",
            color="predicted_risk_class",
            category_orders={"predicted_risk_class": RISK_CLASS_ORDER},
            title=f"Predicted annual severity — {selected_year}",
            labels={"risk_score": "Ground-truth severity score", "gouvernorat": "Governorate"},
        )
        fig.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            year_df,
            x="predicted_fire_count",
            y="predicted_area_burned_ha",
            color="predicted_risk_class",
            hover_name="gouvernorat",
            category_orders={"predicted_risk_class": RISK_CLASS_ORDER},
            title="Predicted annual count vs burned area",
            labels={"predicted_fire_count": "Predicted fires", "predicted_area_burned_ha": "Predicted burned area (ha)"},
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    display_columns = [
        "gouvernorat",
        "year",
        "risk_class",
        "predicted_risk_class",
        "risk_score",
        "fire_count",
        "predicted_fire_count",
        "area_burned_ha",
        "predicted_area_burned_ha",
    ]
    st.dataframe(year_df[display_columns].sort_values("risk_score", ascending=False), use_container_width=True)
    st.download_button(
        "Download annual severity table",
        data=year_df[display_columns].to_csv(index=False).encode("utf-8"),
        file_name=f"annual_governorate_severity_{selected_year}.csv",
        mime="text/csv",
    )


def render_pixel_alert_page() -> None:
    pixel = load_pixel_scores()
    st.header("Operational daily 500 m pixel-risk alerts")
    st.write(
        "This page uses the daily pixel proxy model. Alerts are ranking-based: top 1%, 5%, or 10% of scored pixels "
        "for the selected date."
    )

    all_dates = sorted(pixel["date"].unique())
    default_date = all_dates[-1]
    selected_date = st.selectbox("Prediction date", all_dates, index=len(all_dates) - 1)

    date_df = pixel[pixel["date"] == selected_date].copy()
    gov_options = ["All governorates"] + sorted(date_df["gouvernorat"].dropna().unique().tolist())
    selected_gov = st.selectbox("Governorate", gov_options)
    budget_name = st.radio("Alert budget", list(ALERT_FRACTIONS.keys()), horizontal=True)
    alert_fraction = ALERT_FRACTIONS[budget_name]

    # Alert budget is set globally for the selected date, then optionally filtered for display.
    global_alerted = add_alert_columns(date_df, alert_fraction)
    view_df = global_alerted if selected_gov == "All governorates" else global_alerted[global_alerted["gouvernorat"] == selected_gov].copy()
    alerts_df = view_df[view_df["is_selected_alert"]].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scored pixels", f"{len(view_df):,}")
    c2.metric("Selected alerts", f"{len(alerts_df):,}")
    c3.metric("Max raw risk", format_number(view_df["raw_risk_score"].max() if len(view_df) else np.nan, 3))
    c4.metric("Observed fire labels", f"{int(view_df['target_fire_1d'].sum()) if len(view_df) else 0:,}")

    map_df = prepare_pixel_map_frame(view_df)
    if alerts_df.empty and not view_df.empty:
        st.info(
            "No pixels in the selected governorate fall inside the globally selected alert budget for this date. "
            "The map still displays all scored pixels, colored by predicted risk."
        )
    render_pixel_risk_map(map_df, selected_date, selected_gov)

    st.subheader("Alert table")
    alert_cols = [
        "prediction_date",
        "gouvernorat",
        "pixel_id_500m",
        "raw_risk_score",
        "calibrated_proxy_probability",
        "alert_tier",
        "rank_for_selected_budget",
        "pixel_centroid_lat_proxy",
        "pixel_centroid_lon_proxy",
        "target_fire_1d",
    ]
    st.dataframe(alerts_df[alert_cols].sort_values("raw_risk_score", ascending=False), use_container_width=True)

    csv = alerts_df[alert_cols].sort_values("raw_risk_score", ascending=False).to_csv(index=False).encode("utf-8")
    st.download_button("Download current alert list", csv, file_name=f"wildfire_alerts_{selected_date}.csv", mime="text/csv")

    with st.expander("Send optional email alert", expanded=False):
        st.write("Email delivery is optional and requires Streamlit secrets. The alert CSV is attached.")
        if st.button("Send email alert for current list"):
            subject = f"Wildfire alerts — {selected_date} — {budget_name}"
            body = (
                f"Wildfire alert export for {selected_date}.\n"
                f"Governorate filter: {selected_gov}.\n"
                f"Alert budget: {budget_name}.\n"
                f"Number of displayed alerts: {len(alerts_df)}.\n\n"
                "Reminder: these are proxy-model ranking alerts, not full-landscape calibrated probabilities."
            )
            ok, message = send_email_alert(subject, body, alerts_df[alert_cols])
            if ok:
                st.success(message)
            else:
                st.warning(message)

    with st.expander("Score an uploaded daily feature table", expanded=False):
        st.write("Upload a CSV with the same feature columns as the model-aligned pixel dataset.")
        uploaded = st.file_uploader("Daily feature CSV", type=["csv"])
        if uploaded is not None:
            score_uploaded_features(uploaded)


def score_uploaded_features(uploaded_file) -> None:
    bundle = load_pixel_bundle()
    feature_columns = bundle["feature_columns"]
    incoming = pd.read_csv(uploaded_file)
    missing = [c for c in feature_columns if c not in incoming.columns]
    if missing:
        st.error(f"Uploaded file is missing {len(missing)} required model feature columns: {missing[:10]}")
        return
    scored = incoming.copy()
    raw = bundle["model"].predict_proba(scored[feature_columns])[:, 1]
    scored["raw_risk_score"] = raw
    if "calibrator" in bundle and bundle["calibrator"] is not None:
        scored["calibrated_proxy_probability"] = bundle["calibrator"].predict(raw)
    scored = add_alert_columns(scored, 0.05)
    st.success(f"Scored {len(scored):,} rows. Default alert budget: top 5%.")
    st.dataframe(scored.sort_values("raw_risk_score", ascending=False).head(100), use_container_width=True)
    st.download_button(
        "Download scored upload",
        scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_daily_pixel_features.csv",
        mime="text/csv",
    )


def render_bridge_page() -> None:
    bridge = load_bridge()
    st.header("Validation bridge — annual severity × daily pixel alerts")
    st.write(
        "The bridge does not change either model. It aggregates daily pixel alert pressure to governorate-year level "
        "and compares it with the strategic annual severity model. Governorates outside the current pixel-proxy "
        "universe are retained and explicitly marked as not covered."
    )

    if bridge.empty or "year" not in bridge.columns:
        st.error("The validation bridge is empty or has no year column. Rebuild bridge_annual_pixel_alerts.csv.")
        return

    years = sorted(int(y) for y in bridge["year"].dropna().unique())
    if not years:
        st.error("No valid years are available in the validation bridge.")
        return

    covered_by_year = (
        bridge.groupby("year")["scored_pixel_days"]
        .sum(min_count=1)
        .fillna(0)
    )
    overlapping_years = [int(y) for y, value in covered_by_year.items() if value > 0]
    default_year = overlapping_years[-1] if overlapping_years else years[-1]
    selected_year = st.selectbox(
        "Bridge year",
        years,
        index=years.index(default_year),
    )

    view = normalize_bridge_frame(bridge[bridge["year"] == selected_year].copy())
    if view.empty:
        st.warning("No bridge rows are available for the selected year.")
        return

    covered_governorates = int(view["has_pixel_proxy_coverage"].sum())
    total_governorates = int(view["gouvernorat"].nunique())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Governorates", f"{total_governorates:,}", help=f"{covered_governorates} have pixel-proxy coverage")
    c2.metric("Critical pixel-days", f"{int(view['critical_alert_pixel_days'].sum()):,}")
    c3.metric("High+critical pixel-days", f"{int(view['high_or_critical_alert_pixel_days'].sum()):,}")
    c4.metric("Observed fire pixel-days", f"{int(view['observed_fire_pixel_days'].sum()):,}")

    if covered_governorates < total_governorates:
        st.info(
            f"Pixel-proxy coverage is available for {covered_governorates} of {total_governorates} governorates in "
            f"{selected_year}. Missing operational values are shown as zero and labelled as coverage gaps; they are "
            "not interpreted as zero fire risk."
        )

    x_column = "predicted_severity_score" if "predicted_severity_score" in view.columns else "risk_score"
    x_label = "Predicted annual severity score" if x_column == "predicted_severity_score" else "Annual severity score"

    fig = px.scatter(
        view,
        x=x_column,
        y="high_or_critical_alert_pixel_days",
        size="bridge_marker_size",
        size_max=34,
        color="predicted_risk_class",
        symbol="pixel_coverage_status",
        hover_name="gouvernorat",
        hover_data={
            "observed_fire_pixel_days": True,
            "scored_pixel_days": True,
            "mean_raw_risk_score": ":.3f",
            "max_raw_risk_score": ":.3f",
            "bridge_marker_size": False,
            "pixel_coverage_status": True,
        },
        category_orders={"predicted_risk_class": RISK_CLASS_ORDER},
        color_discrete_map={
            "Low": "#B8E0D2",
            "Medium": "#FFE5A5",
            "High": "#F6B3B3",
        },
        title=f"Annual severity versus daily alert pressure — {selected_year}",
        labels={
            x_column: x_label,
            "high_or_critical_alert_pixel_days": "High + Critical pixel-days",
            "predicted_risk_class": "Predicted annual class",
            "pixel_coverage_status": "Pixel coverage",
        },
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    columns = [
        "gouvernorat",
        "year",
        "risk_class",
        "predicted_risk_class",
        "risk_score",
        "predicted_severity_score",
        "predicted_fire_count",
        "predicted_area_burned_ha",
        "pixel_coverage_status",
        "scored_pixel_days",
        "observed_fire_pixel_days",
        "critical_alert_pixel_days",
        "high_or_critical_alert_pixel_days",
        "watch_or_above_alert_pixel_days",
        "mean_raw_risk_score",
        "max_raw_risk_score",
    ]
    columns = [column for column in columns if column in view.columns]
    if "has_pixel_proxy_coverage" in view.columns:
        ordered_view = view.sort_values(
            ["has_pixel_proxy_coverage", "high_or_critical_alert_pixel_days"],
            ascending=[False, False],
        )
    else:
        ordered_view = view
    table = ordered_view[columns]
    st.dataframe(table, use_container_width=True)
    st.download_button(
        "Download bridge table",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"annual_pixel_bridge_{selected_year}.csv",
        mime="text/csv",
    )


def render_monitoring_page() -> None:
    pixel = load_pixel_scores()
    importance = load_feature_importance()
    annual_metrics, pixel_metrics = load_model_metrics()

    st.header("Model monitoring and data quality")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Pixel model test metrics")
        st.json(pixel_metrics)
    with c2:
        st.subheader("Annual model metrics")
        st.json(annual_metrics)

    st.subheader("Pixel feature importance")
    top_n = st.slider("Top features", 5, min(30, len(importance)), 15)
    fig = px.bar(
        importance.head(top_n).sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Operational pixel model feature importance",
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score distributions")
    score_year = st.selectbox("Year for score distribution", sorted(pixel["year"].unique()), index=len(sorted(pixel["year"].unique())) - 1)
    plot_df = pixel[pixel["year"] == score_year]
    fig = px.histogram(plot_df, x="raw_risk_score", color="target_fire_1d", nbins=50, title=f"Raw risk score distribution — {score_year}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset cautions")
    st.warning(
        "The daily pixel model currently scores a historical-fire-cell proxy universe. Treat scores as relative ranking values. "
        "For a nationwide operational dashboard, replace the proxy universe with the full burnable 500 m grid and a live daily feature pipeline."
    )


def main() -> None:
    render_header()
    render_status_cards()

    page = st.sidebar.radio(
        "Navigation",
        [
            "Strategic annual severity",
            "Operational pixel alerts",
            "Annual–pixel bridge",
            "Model monitoring",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model scopes**")
    st.sidebar.markdown(
        "- Annual model: governorate-year strategic severity.\n"
        "- Pixel model: daily 500 m proxy-cell alert ranking.\n"
        "- Bridge: governorate-year aggregation of daily pixel alerts."
    )

    if page == "Strategic annual severity":
        render_annual_page()
    elif page == "Operational pixel alerts":
        render_pixel_alert_page()
    elif page == "Annual–pixel bridge":
        render_bridge_page()
    else:
        render_monitoring_page()


if __name__ == "__main__":
    main()
