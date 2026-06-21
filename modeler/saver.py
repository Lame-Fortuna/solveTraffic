# %% Part 1 - feature engineering and cascade XGBoost training
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from catboost import CatBoostRegressor
from xgboost import XGBClassifier


RANDOM_STATE = 42
CENTER_LAT = 12.9716
CENTER_LON = 77.5946

DATA_PATH = Path("dataset.csv")
ARTIFACT_DIR = Path("traffic_cascade_artifacts")
BUNDLE_PATH = ARTIFACT_DIR / "traffic_cascade_bundle.pkl"
MODEL_PATH = ARTIFACT_DIR / "cascade_xgboost_model.json"
DURATION_MODEL_PATH = ARTIFACT_DIR / "duration_catboost_model.cbm"
METADATA_PATH = ARTIFACT_DIR / "traffic_cascade_metadata.json"

HISTORICAL_KEYS = ["corridor", "police_station", "zone", "junction"]
CATEGORICAL_COLUMNS = [
    "event_type",
    "event_cause",
    "direction",
    "veh_type",
    "corridor",
    "cargo_material",
    "reason_breakdown",
    "police_station",
    "zone",
    "junction",
]

GENERAL_FEATURES = [
    "latitude",
    "longitude",
    "age_of_truck",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "is_peak_hour",
    "is_night",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "distance_from_center",
    "location_cluster",
    "has_vehicle",
    "has_cargo",
    "truck_breakdown",
    "is_planned",
    "event_type_code",
    "direction_code",
    "veh_type_code",
    "cargo_material_code",
    "reason_breakdown_code",
]
HISTORICAL_CODE_FEATURES = [
    "corridor_code",
    "police_station_code",
    "zone_code",
    "junction_code",
    "event_cause_code",
]
CASCADE_LOGIC_FEATURES = [
    "is_high_impact_cause",
    "is_surge_hour",
    "priority_closure_interaction",
    "critical_cause_flag",
]
HISTORY_FEATURES = [
    f"{key}_{suffix}"
    for key in HISTORICAL_KEYS
    for suffix in ["closure_risk", "impact_mean", "freq"]
]

FEATURE_COLUMNS = list(
    dict.fromkeys(
        GENERAL_FEATURES
        + CASCADE_LOGIC_FEATURES
        + HISTORICAL_CODE_FEATURES
        + HISTORY_FEATURES
    )
)

DURATION_FEATURES = [
    "corridor",
    "event_cause",
    "is_weekend",
    "event_type",
    "priority",
    "latitude",
    "longitude",
    "dist_center",
    "hour_sin",
    "hour_cos",
    "Impact_Level",
    "cascade_prob_impact_level_2",
]
DURATION_CAT_FEATURES = ["corridor", "event_cause", "event_type", "priority"]
DURATION_TARGET = "refined_duration"
DURATION_TARGET_CAP_MINUTES = 1440.0
DURATION_POINT_CAP_MINUTES = 720.0
DURATION_RANGE_EDGES = [0, 30, 60, 120, 240, 480, 720, 1440]
DURATION_DISPLAY_BUCKETS = [
    (0.0, 30.0, "0-30 min"),
    (30.0, 60.0, "30-60 min"),
    (60.0, 120.0, "60-120 min"),
    (120.0, 240.0, "120-240 min"),
    (240.0, 480.0, "240-480 min"),
    (480.0, None, ">480 min"),
]


def normalize_bool(value: Any) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    text = str(value).strip().lower()
    return int(text in {"true", "yes", "1", "planned", "high"})


def normalize_bool_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_bool).fillna(0).astype(int)


def priority_to_num(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().lower()
    return {"low": 0.0, "medium": 0.5, "moderate": 0.5, "high": 1.0}.get(text, 0.0)


def has_real_value_series(s: pd.Series) -> pd.Series:
    text = s.fillna("").astype(str).str.strip().str.lower()
    return s.notna() & ~text.isin({"", "null", "none", "nan", "__missing__"})


def safe_datetime_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
    return pd.Series(pd.NaT, index=df.index)


def build_category_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_COLUMNS:
        values = (
            df[col]
            if col in df.columns
            else pd.Series("__missing__", index=df.index, dtype="object")
        )
        labels = sorted(values.fillna("__missing__").astype(str).unique())
        maps[col] = {label: code for code, label in enumerate(labels)}
    return maps


def apply_category_maps(df: pd.DataFrame, maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        values = (
            df[col]
            if col in df.columns
            else pd.Series("__missing__", index=df.index, dtype="object")
        )
        df[f"{col}_code"] = values.fillna("__missing__").astype(str).map(maps[col]).fillna(-1).astype(int)
    return df


def build_numeric_defaults(df: pd.DataFrame) -> dict[str, float]:
    defaults: dict[str, float] = {}
    for col, fallback in {
        "latitude": CENTER_LAT,
        "longitude": CENTER_LON,
        "endlatitude": CENTER_LAT,
        "endlongitude": CENTER_LON,
        "age_of_truck": 0.0,
    }.items():
        if col in df.columns:
            value = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).median()
            defaults[col] = float(value) if pd.notna(value) else fallback
        else:
            defaults[col] = fallback
    return defaults


def make_historical_lookup(df: pd.DataFrame) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    global_closure = float(df["requires_road_closure"].mean())
    global_impact = float(df["Impact_Level"].mean())
    for key in HISTORICAL_KEYS:
        if key in df.columns:
            key_values = df[key].fillna("__missing__").astype(str)
            temp = df.assign(**{key: key_values})
            lookup[key] = {
                "closure_risk": temp.groupby(key)["requires_road_closure"].mean().to_dict(),
                "impact_mean": temp.groupby(key)["Impact_Level"].mean().to_dict(),
                "freq": temp[key].value_counts().to_dict(),
            }
        else:
            lookup[key] = {"closure_risk": {}, "impact_mean": {}, "freq": {}}
    return {
        "by_key": lookup,
        "global": {
            "closure_risk": global_closure,
            "impact_mean": global_impact,
            "total_count": int(len(df)),
        },
    }


def add_historical_features(df: pd.DataFrame, lookup: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    global_stats = lookup["global"]
    for key in HISTORICAL_KEYS:
        values = (
            df[key].fillna("__missing__").astype(str)
            if key in df.columns
            else pd.Series("__missing__", index=df.index, dtype="object")
        )
        key_lookup = lookup["by_key"][key]
        df[f"{key}_closure_risk"] = values.map(key_lookup["closure_risk"]).fillna(global_stats["closure_risk"])
        df[f"{key}_impact_mean"] = values.map(key_lookup["impact_mean"]).fillna(global_stats["impact_mean"])
        df[f"{key}_freq"] = values.map(key_lookup["freq"]).fillna(0)
    return df


def engineer_base_features(
    raw_df: pd.DataFrame,
    *,
    clusterer: KMeans | None,
    category_maps: dict[str, dict[str, int]],
    numeric_defaults: dict[str, float],
    fit_clusterer: bool = False,
    require_target: bool = False,
) -> tuple[pd.DataFrame, KMeans | None]:
    df = raw_df.copy()
    df.columns = df.columns.str.strip()

    if require_target and "requires_road_closure" not in df.columns:
        raise ValueError("requires_road_closure column is required for training")

    if "requires_road_closure" in df.columns:
        df["requires_road_closure"] = normalize_bool_series(df["requires_road_closure"])
    else:
        df["requires_road_closure"] = 0

    if "priority_num" in df.columns:
        df["priority_num"] = pd.to_numeric(df["priority_num"], errors="coerce").fillna(0)
    elif "priority" in df.columns:
        df["priority_num"] = df["priority"].map(priority_to_num).astype(float)
    elif require_target:
        raise ValueError("priority or priority_num column is required for training")
    else:
        df["priority_num"] = 0.0

    start_dt = safe_datetime_series(df, "start_datetime")
    df["start_datetime"] = start_dt
    df["created_date"] = safe_datetime_series(df, "created_date")
    df["modified_datetime"] = safe_datetime_series(df, "modified_datetime")

    df["year"] = start_dt.dt.year.fillna(0).astype(int)
    df["month"] = start_dt.dt.month.fillna(0).astype(int)
    df["day"] = start_dt.dt.day.fillna(0).astype(int)
    df["hour"] = start_dt.dt.hour.fillna(0).astype(int)
    df["minute"] = start_dt.dt.minute.fillna(0).astype(int)
    df["day_of_week"] = start_dt.dt.dayofweek.fillna(0).astype(int)
    week = start_dt.dt.isocalendar().week.astype("Float64")
    df["week_of_year"] = week.fillna(0).astype(int)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = (df["hour"].between(7, 10) | df["hour"].between(17, 20)).astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    for col in ["latitude", "longitude", "endlatitude", "endlongitude", "age_of_truck"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(numeric_defaults[col])

    df["endlatitude"] = df["endlatitude"].replace(0, np.nan).fillna(df["latitude"])
    df["endlongitude"] = df["endlongitude"].replace(0, np.nan).fillna(df["longitude"])
    df["distance_from_center"] = np.sqrt(
        (df["latitude"] - CENTER_LAT) ** 2 + (df["longitude"] - CENTER_LON) ** 2
    )

    coords = df[["latitude", "longitude"]].replace([np.inf, -np.inf], np.nan).fillna(
        {"latitude": numeric_defaults["latitude"], "longitude": numeric_defaults["longitude"]}
    )
    if fit_clusterer:
        n_clusters = min(20, max(1, len(df)))
        clusterer = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        df["location_cluster"] = clusterer.fit_predict(coords)
    elif clusterer is not None:
        df["location_cluster"] = clusterer.predict(coords)
    else:
        df["location_cluster"] = 0

    if "event_type" in df.columns:
        df["is_planned"] = normalize_bool_series(df["event_type"])
    else:
        df["is_planned"] = 0

    vehicle_source = df["veh_type"] if "veh_type" in df.columns else pd.Series(None, index=df.index)
    cargo_source = df["cargo_material"] if "cargo_material" in df.columns else pd.Series(None, index=df.index)
    cause_source = df["event_cause"] if "event_cause" in df.columns else pd.Series("", index=df.index)
    cause_text = cause_source.fillna("").astype(str).str.lower()

    df["has_vehicle"] = has_real_value_series(vehicle_source).astype(int)
    df["has_cargo"] = has_real_value_series(cargo_source).astype(int)
    df["truck_breakdown"] = cause_text.str.contains("breakdown", regex=False).astype(int)

    df = apply_category_maps(df, category_maps)

    high_impact_causes = ["debris", "vip", "vip_movement"]
    critical_causes = ["vip", "debris", "closure"]
    df["is_high_impact_cause"] = cause_text.apply(lambda x: int(any(c in x for c in high_impact_causes)))
    df["is_surge_hour"] = ((df["hour"] >= 21) | (df["hour"] <= 6)).astype(int)
    df["priority_closure_interaction"] = df["priority_num"] * df["requires_road_closure"]
    df["critical_cause_flag"] = cause_text.apply(lambda x: int(any(c in x for c in critical_causes)))

    if require_target:
        df["Impact_Level"] = np.select(
            [df["requires_road_closure"].eq(1), df["priority_num"].eq(df["priority_num"].max())],
            [2, 1],
            default=0,
        ).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, clusterer


def add_fold_history(train_part: pd.DataFrame, valid_part: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_lookup = make_historical_lookup(train_part)
    return add_historical_features(train_part, fold_lookup), add_historical_features(valid_part, fold_lookup)


def make_cascade_xgb() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        learning_rate=0.02,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def run_cascade_cv(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    y = df["Impact_Level"]
    min_class = int(y.value_counts().min())
    n_splits = min(n_splits, min_class)
    if n_splits < 2:
        raise ValueError("Need at least 2 rows in every Impact_Level class for stratified CV")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float]] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        train_part = df.iloc[train_idx].copy()
        valid_part = df.iloc[val_idx].copy()
        train_part, valid_part = add_fold_history(train_part, valid_part)

        X_train = train_part[FEATURE_COLUMNS].fillna(0)
        y_train = train_part["Impact_Level"]
        X_val = valid_part[FEATURE_COLUMNS].fillna(0)
        y_val = valid_part["Impact_Level"]

        sample_w = compute_sample_weight(class_weight={0: 1, 1: 1, 2: 50}, y=y_train)
        model = make_cascade_xgb()
        model.fit(X_train, y_train, sample_weight=sample_w)
        pred = model.predict(X_val)
        f1_each = f1_score(y_val, pred, average=None, labels=[0, 1, 2], zero_division=0)
        rows.append(
            {
                "fold": fold,
                "f1_0": float(f1_each[0]),
                "f1_1": float(f1_each[1]),
                "f1_2": float(f1_each[2]),
                "macro": float(f1_score(y_val, pred, average="macro", zero_division=0)),
            }
        )
        print(f"Fold {fold}: label-2 F1={f1_each[2]:.4f}")
    return pd.DataFrame(rows)


def train_cascade_pipeline(data_path: str | Path = DATA_PATH, run_cv: bool = True) -> dict[str, Any]:
    raw_df = pd.read_csv(data_path)
    raw_df.columns = raw_df.columns.str.strip()

    numeric_defaults = build_numeric_defaults(raw_df)
    category_maps = build_category_maps(raw_df)
    featured_df, clusterer = engineer_base_features(
        raw_df,
        clusterer=None,
        category_maps=category_maps,
        numeric_defaults=numeric_defaults,
        fit_clusterer=True,
        require_target=True,
    )
    assert clusterer is not None

    historical_lookup = make_historical_lookup(featured_df)
    train_df = add_historical_features(featured_df, historical_lookup)

    cv_results = run_cascade_cv(train_df) if run_cv else pd.DataFrame()
    X = train_df[FEATURE_COLUMNS].fillna(0)
    y = train_df["Impact_Level"]
    sample_w = compute_sample_weight(class_weight={0: 1, 1: 1, 2: 50}, y=y)

    model = make_cascade_xgb()
    model.fit(X, y, sample_weight=sample_w)

    return {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "clusterer": clusterer,
        "category_maps": category_maps,
        "numeric_defaults": numeric_defaults,
        "historical_lookup": historical_lookup,
        "training_summary": {
            "rows": int(len(train_df)),
            "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "cv_mean": cv_results.drop(columns="fold").mean().to_dict() if not cv_results.empty else {},
        },
        "cv_results": cv_results,
    }


def get_cascade_oof_signals(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    y = df["Impact_Level"]
    min_class = int(y.value_counts().min())
    n_splits = min(n_splits, min_class)
    if n_splits < 2:
        full_lookup = make_historical_lookup(df)
        full_df = add_historical_features(df, full_lookup)
        model = make_cascade_xgb()
        sample_w = compute_sample_weight(class_weight={0: 1, 1: 1, 2: 50}, y=y)
        model.fit(full_df[FEATURE_COLUMNS].fillna(0), y, sample_weight=sample_w)
        probs = model.predict_proba(full_df[FEATURE_COLUMNS].fillna(0))
        labels = model.predict(full_df[FEATURE_COLUMNS].fillna(0)).astype(int)
        return pd.DataFrame(
            {
                "Impact_Level": labels,
                "cascade_prob_impact_level_0": probs[:, 0],
                "cascade_prob_impact_level_1": probs[:, 1],
                "cascade_prob_impact_level_2": probs[:, 2],
            },
            index=df.index,
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros((len(df), 3), dtype=float)
    oof_labels = np.zeros(len(df), dtype=int)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        train_part = df.iloc[train_idx].copy()
        valid_part = df.iloc[val_idx].copy()
        train_part, valid_part = add_fold_history(train_part, valid_part)

        X_train = train_part[FEATURE_COLUMNS].fillna(0)
        y_train = train_part["Impact_Level"]
        X_val = valid_part[FEATURE_COLUMNS].fillna(0)
        sample_w = compute_sample_weight(class_weight={0: 1, 1: 1, 2: 50}, y=y_train)

        model = make_cascade_xgb()
        model.fit(X_train, y_train, sample_weight=sample_w)
        oof_probs[val_idx] = model.predict_proba(X_val)
        oof_labels[val_idx] = model.predict(X_val).astype(int)
        print(f"Classifier OOF fold {fold} complete for duration signals")

    return pd.DataFrame(
        {
            "Impact_Level": oof_labels,
            "cascade_prob_impact_level_0": oof_probs[:, 0],
            "cascade_prob_impact_level_1": oof_probs[:, 1],
            "cascade_prob_impact_level_2": oof_probs[:, 2],
        },
        index=df.index,
    )


def build_refined_duration_target(raw_df: pd.DataFrame) -> pd.Series:
    start = safe_datetime_series(raw_df, "start_datetime")
    end = pd.Series(pd.NaT, index=raw_df.index)
    for col in ["resolved_datetime", "end_datetime", "closed_datetime", "modified_datetime"]:
        end = end.fillna(safe_datetime_series(raw_df, col))
    duration = (end - start).dt.total_seconds().div(60.0)
    return duration


def clean_duration_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DURATION_CAT_FEATURES:
        if col not in out.columns:
            out[col] = "Unknown"
        out[col] = out[col].fillna("Unknown").astype(str)
        out.loc[out[col].str.strip().str.lower().isin(["", "null", "none", "nan"]), col] = "Unknown"

    for col in DURATION_FEATURES:
        if col not in out.columns:
            out[col] = 0
    numeric_cols = [c for c in DURATION_FEATURES if c not in DURATION_CAT_FEATURES]
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return out[DURATION_FEATURES]


def build_duration_frame(
    raw_df: pd.DataFrame,
    classifier_artifacts: dict[str, Any],
    classifier_signals: pd.DataFrame | None = None,
    include_target: bool = False,
) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df.columns = raw_df.columns.str.strip()

    featured_df, _ = engineer_base_features(
        raw_df,
        clusterer=classifier_artifacts["clusterer"],
        category_maps=classifier_artifacts["category_maps"],
        numeric_defaults=classifier_artifacts["numeric_defaults"],
        fit_clusterer=False,
        require_target=False,
    )
    duration_df = pd.DataFrame(index=raw_df.index)
    for col in ["corridor", "event_cause", "event_type", "priority"]:
        duration_df[col] = raw_df[col] if col in raw_df.columns else "Unknown"

    duration_df["latitude"] = featured_df["latitude"]
    duration_df["longitude"] = featured_df["longitude"]
    duration_df["dist_center"] = featured_df["distance_from_center"]
    duration_df["hour_sin"] = featured_df["hour_sin"]
    duration_df["hour_cos"] = featured_df["hour_cos"]
    duration_df["is_weekend"] = featured_df["is_weekend"]

    if classifier_signals is None:
        full_df = add_historical_features(featured_df, classifier_artifacts["historical_lookup"])
        X = full_df[classifier_artifacts["feature_columns"]].fillna(0)
        probs = classifier_artifacts["model"].predict_proba(X)
        labels = classifier_artifacts["model"].predict(X).astype(int)
        classifier_signals = pd.DataFrame(
            {
                "Impact_Level": labels,
                "cascade_prob_impact_level_2": probs[:, 2],
            },
            index=raw_df.index,
        )

    duration_df["Impact_Level"] = classifier_signals["Impact_Level"].astype(int).values
    duration_df["cascade_prob_impact_level_2"] = classifier_signals["cascade_prob_impact_level_2"].astype(float).values

    if include_target:
        duration_df[DURATION_TARGET] = build_refined_duration_target(raw_df)

    return duration_df


def make_duration_regressor() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        verbose=0,
        random_seed=RANDOM_STATE,
    )


def make_duration_range_calibration(actual: pd.Series, pred: np.ndarray) -> dict[str, Any]:
    results = pd.DataFrame({"actual": actual.to_numpy(dtype=float), "pred": pred.astype(float)})
    results["abs_error"] = (results["actual"] - results["pred"]).abs()
    results["pred_bucket"] = pd.cut(
        results["pred"],
        bins=DURATION_RANGE_EDGES,
        labels=[f"{DURATION_RANGE_EDGES[i]}-{DURATION_RANGE_EDGES[i + 1]}" for i in range(len(DURATION_RANGE_EDGES) - 1)],
        right=False,
        include_lowest=True,
    )

    fallback_half_width = {
        "0-30": 8.0,
        "30-60": 38.0,
        "60-120": 84.0,
        "120-240": 145.0,
        "240-480": 318.0,
        "480-720": 856.0,
        "720-1440": 856.0,
    }
    by_bucket: dict[str, Any] = {}
    for label, group in results.groupby("pred_bucket", observed=False):
        label_text = str(label)
        if group.empty:
            by_bucket[label_text] = {
                "count": 0,
                "q80_abs_error": fallback_half_width[label_text],
                "median_abs_error": fallback_half_width[label_text],
            }
            continue
        by_bucket[label_text] = {
            "count": int(len(group)),
            "q80_abs_error": float(group["abs_error"].quantile(0.80)),
            "median_abs_error": float(group["abs_error"].median()),
        }

    return {
        "edges": DURATION_RANGE_EDGES,
        "by_pred_bucket": by_bucket,
        "fallback_half_width": fallback_half_width,
    }


def train_duration_pipeline(raw_df: pd.DataFrame, classifier_artifacts: dict[str, Any]) -> dict[str, Any]:
    classifier_featured_df, _ = engineer_base_features(
        raw_df,
        clusterer=classifier_artifacts["clusterer"],
        category_maps=classifier_artifacts["category_maps"],
        numeric_defaults=classifier_artifacts["numeric_defaults"],
        fit_clusterer=False,
        require_target=True,
    )
    classifier_signals = get_cascade_oof_signals(classifier_featured_df)
    duration_df = build_duration_frame(
        raw_df,
        classifier_artifacts,
        classifier_signals=classifier_signals,
        include_target=True,
    )
    mask = (
        duration_df[DURATION_TARGET].notna()
        & (duration_df[DURATION_TARGET] > 0)
        & (duration_df[DURATION_TARGET] <= DURATION_TARGET_CAP_MINUTES)
    )
    duration_df = duration_df.loc[mask].copy()
    if len(duration_df) < 20:
        raise ValueError("Need at least 20 valid duration rows to train duration regression")

    train_df = duration_df.sample(frac=0.8, random_state=RANDOM_STATE)
    test_df = duration_df.drop(train_df.index)
    if test_df.empty:
        test_df = train_df

    X_train = clean_duration_feature_frame(train_df)
    y_train_log = np.log1p(train_df[DURATION_TARGET])
    X_test = clean_duration_feature_frame(test_df)
    y_test = test_df[DURATION_TARGET]

    calibration_model = make_duration_regressor()
    calibration_model.fit(X_train, y_train_log, cat_features=DURATION_CAT_FEATURES)
    pred_log = calibration_model.predict(X_test)
    pred = np.expm1(np.clip(pred_log, 0, np.log1p(DURATION_POINT_CAP_MINUTES)))
    calibration = make_duration_range_calibration(y_test, pred)

    final_model = make_duration_regressor()
    final_model.fit(
        clean_duration_feature_frame(duration_df),
        np.log1p(duration_df[DURATION_TARGET]),
        cat_features=DURATION_CAT_FEATURES,
    )

    mae = float(np.mean(np.abs(y_test.to_numpy(dtype=float) - pred)))
    return {
        "duration_model": final_model,
        "duration_features": DURATION_FEATURES,
        "duration_cat_features": DURATION_CAT_FEATURES,
        "duration_calibration": calibration,
        "duration_training_summary": {
            "rows": int(len(duration_df)),
            "mae_minutes": mae,
            "target_mean_minutes": float(duration_df[DURATION_TARGET].mean()),
            "target_median_minutes": float(duration_df[DURATION_TARGET].median()),
        },
    }


def train_full_traffic_pipeline(data_path: str | Path = DATA_PATH, run_cv: bool = True) -> dict[str, Any]:
    classifier_artifacts = train_cascade_pipeline(data_path, run_cv=run_cv)
    raw_df = pd.read_csv(data_path)
    raw_df.columns = raw_df.columns.str.strip()
    duration_artifacts = train_duration_pipeline(raw_df, classifier_artifacts)
    classifier_artifacts.update(duration_artifacts)
    return classifier_artifacts


if __name__ == "__main__":
    artifacts = train_full_traffic_pipeline(DATA_PATH, run_cv=True)
    print("Training complete")
    print(artifacts["training_summary"])
    print(artifacts["duration_training_summary"])


# %% Part 2 - save all deployable files
def save_cascade_artifacts(artifacts: dict[str, Any], artifact_dir: str | Path = ARTIFACT_DIR) -> dict[str, Path]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = artifact_dir / "traffic_cascade_bundle.pkl"
    model_path = artifact_dir / "cascade_xgboost_model.json"
    duration_model_path = artifact_dir / "duration_catboost_model.cbm"
    metadata_path = artifact_dir / "traffic_cascade_metadata.json"
    cv_path = artifact_dir / "cascade_cv_results.csv"

    artifacts["model"].save_model(model_path)
    if "duration_model" in artifacts:
        artifacts["duration_model"].save_model(duration_model_path)

    bundle = {
        "model": artifacts["model"],
        "feature_columns": artifacts["feature_columns"],
        "clusterer": artifacts["clusterer"],
        "category_maps": artifacts["category_maps"],
        "numeric_defaults": artifacts["numeric_defaults"],
        "historical_lookup": artifacts["historical_lookup"],
        "training_summary": artifacts["training_summary"],
        "model_path": str(model_path),
    }
    if "duration_model" in artifacts:
        bundle.update(
            {
                "duration_model": artifacts["duration_model"],
                "duration_features": artifacts["duration_features"],
                "duration_cat_features": artifacts["duration_cat_features"],
                "duration_calibration": artifacts["duration_calibration"],
                "duration_training_summary": artifacts["duration_training_summary"],
                "duration_model_path": str(duration_model_path),
            }
        )
    with bundle_path.open("wb") as f:
        pickle.dump(bundle, f)

    metadata = {
        "feature_columns": artifacts["feature_columns"],
        "numeric_defaults": artifacts["numeric_defaults"],
        "category_maps": artifacts["category_maps"],
        "historical_lookup": artifacts["historical_lookup"],
        "training_summary": artifacts["training_summary"],
        "model_path": str(model_path),
        "bundle_path": str(bundle_path),
    }
    if "duration_model" in artifacts:
        metadata.update(
            {
                "duration_features": artifacts["duration_features"],
                "duration_cat_features": artifacts["duration_cat_features"],
                "duration_calibration": artifacts["duration_calibration"],
                "duration_training_summary": artifacts["duration_training_summary"],
                "duration_model_path": str(duration_model_path),
            }
        )
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    if not artifacts["cv_results"].empty:
        artifacts["cv_results"].to_csv(cv_path, index=False)

    print(f"Saved deployment bundle: {bundle_path}")
    print(f"Saved standalone XGBoost model: {model_path}")
    if "duration_model" in artifacts:
        print(f"Saved standalone CatBoost duration model: {duration_model_path}")
    print(f"Saved readable metadata: {metadata_path}")
    return {
        "bundle": bundle_path,
        "model": model_path,
        "duration_model": duration_model_path,
        "metadata": metadata_path,
        "cv_results": cv_path,
    }


if __name__ == "__main__":
    saved_paths = save_cascade_artifacts(artifacts, ARTIFACT_DIR)


# %% Part 3 - load saved files and predict with the same logic
def load_cascade_bundle(bundle_path: str | Path = BUNDLE_PATH) -> dict[str, Any]:
    with Path(bundle_path).open("rb") as f:
        return pickle.load(f)


def prepare_prediction_frame(raw_events: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    if isinstance(raw_events, pd.DataFrame):
        raw_df = raw_events.copy()
    elif isinstance(raw_events, dict):
        raw_df = pd.DataFrame([raw_events])
    else:
        raw_df = pd.DataFrame(raw_events)

    df, _ = engineer_base_features(
        raw_df,
        clusterer=bundle["clusterer"],
        category_maps=bundle["category_maps"],
        numeric_defaults=bundle["numeric_defaults"],
        fit_clusterer=False,
        require_target=False,
    )
    df = add_historical_features(df, bundle["historical_lookup"])
    return df[bundle["feature_columns"]].replace([np.inf, -np.inf], np.nan).fillna(0)


def predict_impact(
    raw_events: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    bundle: dict[str, Any] | None = None,
    bundle_path: str | Path = BUNDLE_PATH,
    return_probabilities: bool = False,
) -> int | list[int] | pd.DataFrame:
    bundle = load_cascade_bundle(bundle_path) if bundle is None else bundle
    X = prepare_prediction_frame(raw_events, bundle)
    preds = bundle["model"].predict(X).astype(int)

    if return_probabilities:
        probs = bundle["model"].predict_proba(X)
        output = pd.DataFrame(
            {
                "Impact_Level": preds,
                "Impact_Description": [
                    {0: "Low", 1: "High priority", 2: "Road closure / critical"}.get(int(p), "Unknown")
                    for p in preds
                ],
                "prob_level_0": probs[:, 0],
                "prob_level_1": probs[:, 1],
                "prob_level_2": probs[:, 2],
            }
        )
        return output

    return int(preds[0]) if len(preds) == 1 else preds.tolist()


def _duration_bucket_label(point_minutes: float, calibration: dict[str, Any]) -> str:
    edges = calibration.get("edges", DURATION_RANGE_EDGES)
    point = float(np.clip(point_minutes, 0, DURATION_TARGET_CAP_MINUTES))
    for i in range(len(edges) - 1):
        if edges[i] <= point < edges[i + 1]:
            return f"{edges[i]}-{edges[i + 1]}"
    return f"{edges[-2]}-{edges[-1]}"


def duration_display_bucket(point_minutes: float) -> dict[str, Any]:
    point = float(np.clip(point_minutes, 0, DURATION_TARGET_CAP_MINUTES))
    for lower, upper, label in DURATION_DISPLAY_BUCKETS:
        if upper is None:
            if point >= lower:
                return {
                    "lower_minutes": int(lower),
                    "upper_minutes": None,
                    "label": label,
                    "point_estimate_minutes": round(point, 1),
                }
        elif lower <= point < upper:
            return {
                "lower_minutes": int(lower),
                "upper_minutes": int(upper),
                "label": label,
                "point_estimate_minutes": round(point, 1),
            }
    lower, upper, label = DURATION_DISPLAY_BUCKETS[-1]
    return {
        "lower_minutes": int(lower),
        "upper_minutes": None if upper is None else int(upper),
        "label": label,
        "point_estimate_minutes": round(point, 1),
    }


def duration_range_from_point(
    point_minutes: float,
    impact_label: int,
    prob_level_2: float,
    closure_hint: Any = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    display = duration_display_bucket(point_minutes)
    display["basis"] = "non-uniform duration bucket from regression point estimate"
    return display


def predict_traffic_impact_and_duration(
    raw_events: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    bundle: dict[str, Any] | None = None,
    bundle_path: str | Path = BUNDLE_PATH,
) -> dict[str, Any] | list[dict[str, Any]]:
    bundle = load_cascade_bundle(bundle_path) if bundle is None else bundle
    if "duration_model" not in bundle:
        raise ValueError("Bundle does not contain a duration_model. Re-run the full training/saving sections.")

    raw_df = raw_events.copy() if isinstance(raw_events, pd.DataFrame) else pd.DataFrame([raw_events] if isinstance(raw_events, dict) else raw_events)
    classifier_X = prepare_prediction_frame(raw_df, bundle)
    probs = bundle["model"].predict_proba(classifier_X)
    labels = bundle["model"].predict(classifier_X).astype(int)

    classifier_signals = pd.DataFrame(
        {
            "Impact_Level": labels,
            "cascade_prob_impact_level_2": probs[:, 2],
        },
        index=raw_df.index,
    )
    duration_frame = build_duration_frame(raw_df, bundle, classifier_signals=classifier_signals, include_target=False)
    duration_X = clean_duration_feature_frame(duration_frame)
    duration_log = bundle["duration_model"].predict(duration_X)
    duration_points = np.expm1(np.clip(duration_log, 0, np.log1p(DURATION_POINT_CAP_MINUTES)))

    outputs: list[dict[str, Any]] = []
    for i, idx in enumerate(raw_df.index):
        prob_dict = {str(label): round(float(probs[i, label]), 4) for label in range(probs.shape[1])}
        duration_range = duration_range_from_point(
            duration_points[i],
            impact_label=int(labels[i]),
            prob_level_2=float(probs[i, 2]),
            closure_hint=raw_df.at[idx, "requires_road_closure"] if "requires_road_closure" in raw_df.columns else None,
            calibration=bundle.get("duration_calibration"),
        )
        outputs.append(
            {
                "impact_label": int(labels[i]),
                "impact_probabilities": prob_dict,
                "duration_range": {
                    "lower_minutes": duration_range["lower_minutes"],
                    "upper_minutes": duration_range["upper_minutes"],
                    "label": duration_range["label"],
                },
                "duration_point_estimate_minutes": duration_range["point_estimate_minutes"],
            }
        )

    return outputs[0] if isinstance(raw_events, dict) else outputs


if __name__ == "__main__":
    loaded_bundle = load_cascade_bundle(BUNDLE_PATH)
    example_event = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "start_datetime": "2024-12-01 08:30:00",
        "event_type": "unplanned",
        "event_cause": "heavy rain and water logging",
        "priority": "High",
        "requires_road_closure": True,
        "age_of_truck": 5,
        "corridor": "MG Road",
        "police_station": "Central",
        "zone": "East",
        "junction": "Trinity Circle",
    }

    print(predict_traffic_impact_and_duration(example_event, loaded_bundle))
