import os
import re
import json
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


warnings.filterwarnings("ignore")

# Paths are kept relative to the project root so the old command
# `python advanced_kaggle_solution.py` continues to work.
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
MAIN_SUBMISSION_PATH = "submission.csv"
REPORTS_DIR = "reports"
SUBMISSIONS_DIR = "submissions"

SEEDS = [42, 777, 2025, 3407, 1001]
TARGET_CANDIDATES = ["Прогрессия", "Интракраниальная прогрессия"]
LEAKAGE_COLUMNS = TARGET_CANDIDATES + ["Локальный рецидив", "Дистантные метастазы"]
ID_CANDIDATES = ["ID", "Id", "id"]
MISSING_TOKENS = {"", "-", "—", "--", "nan", "none", "null", "na", "n/a", "#ref!"}
YES_TOKENS = {"есть", "да", "1", "true", "yes", "y", "+"}
NO_TOKENS = {"нет", "0", "false", "no", "n", "не удален", "не удалён", "отсутствует"}
MALE_TOKENS = {"м", "муж", "мужской", "male"}
FEMALE_TOKENS = {"ж", "жен", "женский", "female"}


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class FeatureStats:
    interval_clip: Dict[str, Tuple[float, float]]
    lesion_bin_edges: List[float]
    total_volume_bin_edges: List[float]
    max_volume_bin_edges: List[float]


@dataclass
class ModelResult:
    name: str
    model_family: str
    feature_set: str
    cv_strategy: str
    n_folds: int
    oof_proba: np.ndarray
    test_proba: np.ndarray
    best_threshold: float
    oof_bal_acc: float
    test_pos_rate: float
    fold_scores: List[float]
    fold_thresholds: List[float]
    notes: str


# ---------------------------------------------------------------------------
# Project setup and data loading
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def read_csv_robust(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1251", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[load] {path} loaded with {enc}, shape={df.shape}")
            return df
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed reading {path}. Last error: {last_error}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in out.columns]
    return out


def normalize_text_value(v):
    if pd.isna(v):
        return np.nan
    s = str(v).replace("\xa0", " ").strip().lower().replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    if s in MISSING_TOKENS:
        return np.nan
    return s


def parse_numeric_value(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).replace("\xa0", " ").strip().lower()
    s = s.replace(" ", "").replace(",", ".")
    s = re.sub(r"[^0-9\.\-]+", "", s)
    if s in {"", ".", "-", "--"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def norm_key(s: str) -> str:
    s = normalize_text_value(s)
    if pd.isna(s):
        return ""
    return re.sub(r"[^a-zа-я0-9]+", "", s)


def find_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lookup = {norm_key(c): c for c in columns}
    for cand in candidates:
        k = norm_key(cand)
        if k in lookup:
            return lookup[k]
    for cand in candidates:
        k = norm_key(cand)
        for ek, col in lookup.items():
            if k and (k in ek or ek in k):
                return col
    return None


def prepare_target(train_df: pd.DataFrame) -> Tuple[pd.Series, str]:
    if "Прогрессия" in train_df.columns:
        raw = train_df["Прогрессия"]
        if is_numeric_dtype(raw):
            return pd.to_numeric(raw, errors="coerce"), "Прогрессия"
        z = raw.map(normalize_text_value)
        return z.map(lambda x: 1.0 if x in YES_TOKENS else (0.0 if x in NO_TOKENS else np.nan)), "Прогрессия"

    if "Интракраниальная прогрессия" in train_df.columns:
        z = train_df["Интракраниальная прогрессия"].map(normalize_text_value)
        # As requested: everything non-"нет" becomes 1
        y = z.map(lambda x: np.nan if pd.isna(x) else (0.0 if x == "нет" else 1.0))
        return y.astype(float), "Интракраниальная прогрессия"

    raise ValueError("Target not found: expected 'Прогрессия' or 'Интракраниальная прогрессия'.")


def get_leakage_columns_to_drop(columns: Iterable[str]) -> List[str]:
    """Return outcome columns that must never become model features."""
    column_set = set(columns)
    return [column for column in LEAKAGE_COLUMNS if column in column_set]


def detect_leakage_columns(train_cols: List[str], test_cols: List[str]) -> Dict[str, List[str]]:
    train_set = set(train_cols)
    test_set = set(test_cols)
    explicit = get_leakage_columns_to_drop(train_cols)

    suspicious = []
    for c in train_cols:
        nk = norm_key(c)
        if any(tok in nk for tok in ["прогресс", "рецидив", "дистант", "метастаз", "followup", "исход"]):
            suspicious.append(c)

    train_only = sorted(list(train_set - test_set))
    test_only = sorted(list(test_set - train_set))
    return {
        "explicit_drop": explicit,
        "suspicious": sorted(list(set(suspicious))),
        "train_only": train_only,
        "test_only": test_only,
    }


def save_leakage_report(leakage_info: Dict[str, List[str]]) -> None:
    rows = []
    for category, columns in leakage_info.items():
        for column in columns:
            rows.append(
                {
                    "category": category,
                    "column": column,
                    "excluded_from_features": column in leakage_info["explicit_drop"],
                }
            )
    pd.DataFrame(rows, columns=["category", "column", "excluded_from_features"]).to_csv(
        f"{REPORTS_DIR}/leakage_columns.csv", index=False, encoding="utf-8"
    )
    print(f"[leakage] excluded outcome columns: {leakage_info['explicit_drop']}")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_column_names(df)
    for c in out.columns:
        if is_string_dtype(out[c]) or out[c].dtype == object:
            out[c] = out[c].map(normalize_text_value)
    return out


def _parse_date_series(s: pd.Series) -> pd.Series:
    first = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")
    unresolved = first.isna() & s.notna()
    if unresolved.any():
        second = pd.to_datetime(s[unresolved], dayfirst=True, errors="coerce")
        first.loc[unresolved] = second
    return first


def add_date_features(
    df: pd.DataFrame, stats: Optional[FeatureStats] = None
) -> Tuple[pd.DataFrame, FeatureStats, Dict[str, float]]:
    out = df.copy()
    cols = list(out.columns)
    date_cols = [c for c in cols if "дата" in norm_key(c)]
    parse_success = {}
    parsed = {}

    for c in date_cols:
        dt_col = f"{c}__dt"
        out[dt_col] = _parse_date_series(out[c])
        parsed[c] = dt_col
        parse_success[c] = float(out[dt_col].notna().mean())
        out[f"{c}__is_valid_date"] = out[dt_col].notna().astype(float)
        out[f"{c}__year"] = out[dt_col].dt.year.astype(float)
        out[f"{c}__month"] = out[dt_col].dt.month.astype(float)

    c_rx = find_col(cols, ["Дата 1-ой РХ", "Дата 1 ой РХ", "Дата 1-ои РХ"])
    rx_dt = parsed.get(c_rx) if c_rx else None
    if rx_dt is not None:
        for c in date_cols:
            out[f"{c}__days_from_rx"] = (out[parsed[c]] - out[rx_dt]).dt.days.astype(float)

    c_birth = find_col(cols, ["Дата рождения"])
    c_diag = find_col(cols, ["Дата постановки онкологического диагноза / начала первичного лечения"])
    c_mgm = find_col(cols, ["Дата развития МГМ"])
    c_rem = find_col(cols, ["Дата удаления первичного очага"])
    c_ovgm = find_col(cols, ["Дата проведения ОВГМ"])
    c_op = find_col(cols, ["Дата операции на ГМ"])

    def dt(c: Optional[str]) -> Optional[str]:
        return parsed.get(c) if c else None

    def diff_days(end_dt: Optional[str], start_dt: Optional[str]) -> pd.Series:
        if end_dt is None or start_dt is None:
            return pd.Series(np.nan, index=out.index, dtype=float)
        return (out[end_dt] - out[start_dt]).dt.days.astype(float)

    out["primary_removed_flag"] = out[dt(c_rem)].notna().astype(float) if dt(c_rem) else 0.0
    out["ovgm_flag"] = out[dt(c_ovgm)].notna().astype(float) if dt(c_ovgm) else 0.0
    out["brain_operation_flag"] = out[dt(c_op)].notna().astype(float) if dt(c_op) else 0.0

    out["age_at_rx_years"] = diff_days(dt(c_rx), dt(c_birth)) / 365.25
    out["age_at_diagnosis_years"] = diff_days(dt(c_diag), dt(c_birth)) / 365.25
    out["age_at_mgm_years"] = diff_days(dt(c_mgm), dt(c_birth)) / 365.25
    out["diagnosis_to_mgm_days"] = diff_days(dt(c_mgm), dt(c_diag))
    out["mgm_to_rx_days"] = diff_days(dt(c_rx), dt(c_mgm))
    out["diagnosis_to_rx_days"] = diff_days(dt(c_rx), dt(c_diag))
    out["primary_removal_to_rx_days"] = diff_days(dt(c_rx), dt(c_rem))
    out["ovgm_to_rx_days"] = diff_days(dt(c_rx), dt(c_ovgm))
    out["brain_operation_to_rx_days"] = diff_days(dt(c_rx), dt(c_op))

    interval_cols = [
        "diagnosis_to_mgm_days",
        "mgm_to_rx_days",
        "diagnosis_to_rx_days",
        "primary_removal_to_rx_days",
        "ovgm_to_rx_days",
        "brain_operation_to_rx_days",
    ]

    if stats is None:
        interval_clip = {}
        for c in interval_cols:
            v = out[c].dropna()
            if len(v) >= 10:
                interval_clip[c] = (float(v.quantile(0.01)), float(v.quantile(0.99)))
            else:
                interval_clip[c] = (-36500.0, 36500.0)

        stats = FeatureStats(
            interval_clip=interval_clip,
            lesion_bin_edges=[],
            total_volume_bin_edges=[],
            max_volume_bin_edges=[],
        )

    for c in interval_cols:
        lo, hi = stats.interval_clip[c]
        out[f"{c}_raw"] = out[c]
        out[f"{c}_clipped"] = out[c].clip(lo, hi)
        out[f"{c}_log1p_pos"] = np.log1p(out[c].clip(lower=0))
        out[f"{c}_negative_flag"] = (out[c] < 0).astype(float)
        out[f"{c}_missing_flag"] = out[c].isna().astype(float)

    # Drop raw datetime columns from model features
    out = out.drop(columns=[c for c in out.columns if c.endswith("__dt")], errors="ignore")
    return out, stats, parse_success


def add_clinical_features(df: pd.DataFrame, stats: Optional[FeatureStats] = None) -> Tuple[pd.DataFrame, FeatureStats]:
    out = df.copy()
    cols = list(out.columns)

    c_rh = find_col(cols, ["Число РХ процедур на ГН"])
    c_karn = find_col(cols, ["Индекс Карновского"])
    c_lesions = find_col(cols, ["Число очагов в ГМ"])
    c_total = find_col(cols, ["Суммарный объём очагов", "Суммарный объем очагов"])
    c_max = find_col(cols, ["Объём максимального очага", "Объем максимального очага"])
    c_sex = find_col(cols, ["Пол"])
    c_diag = find_col(cols, ["Онкологический диагноз"])
    c_mut = find_col(cols, ["Активирующие мутации"])
    c_ext = find_col(cols, ["Экстракраниальные метастазы"])
    c_treat = find_col(cols, ["Лекарственное лечение"])

    numeric_sources = [c_rh, c_karn, c_lesions, c_total, c_max]
    for c in numeric_sources:
        if c and c in out.columns:
            out[c] = out[c].map(parse_numeric_value)

    out["lesion_count"] = out[c_lesions] if c_lesions else np.nan
    out["total_volume"] = out[c_total] if c_total else np.nan
    out["max_volume"] = out[c_max] if c_max else np.nan
    out["karnovsky"] = out[c_karn] if c_karn else np.nan
    out["rh_procedures"] = out[c_rh] if c_rh else np.nan

    out["has_activating_mutations"] = (
        out[c_mut].map(lambda x: 1.0 if x in YES_TOKENS else (0.0 if x in NO_TOKENS else np.nan))
        if c_mut
        else np.nan
    )
    out["has_extracranial_metastases"] = (
        out[c_ext].map(lambda x: 1.0 if x in YES_TOKENS else (0.0 if x in NO_TOKENS else np.nan))
        if c_ext
        else np.nan
    )
    out["sex_bin"] = (
        out[c_sex].map(lambda x: 0.0 if x in MALE_TOKENS else (1.0 if x in FEMALE_TOKENS else np.nan))
        if c_sex
        else np.nan
    )

    out["avg_volume_per_lesion"] = np.where(out["lesion_count"] > 0, out["total_volume"] / out["lesion_count"], np.nan)
    out["max_to_total_volume_ratio"] = np.where(out["total_volume"] > 0, out["max_volume"] / out["total_volume"], np.nan)
    out["residual_volume"] = out["total_volume"] - out["max_volume"]
    # Volumes are strongly skewed; log1p keeps zero values valid and reduces outlier influence.
    out["log1p_lesion_count"] = np.log1p(out["lesion_count"].clip(lower=0))
    out["log1p_total_volume"] = np.log1p(out["total_volume"].clip(lower=0))
    out["log1p_max_volume"] = np.log1p(out["max_volume"].clip(lower=0))
    out["karnovsky_deficit"] = 100 - out["karnovsky"]

    # Binning
    if stats is None:
        def q_edges(s: pd.Series) -> List[float]:
            if s.dropna().empty:
                return [0.0, 1.0]
            qs = s.quantile([0.0, 0.33, 0.66, 1.0]).values
            qs = np.unique(np.round(qs, 6))
            if len(qs) < 2:
                qs = np.array([float(s.min()), float(s.max()) + 1e-6])
            return qs.tolist()

        if stats is None:
            stats = FeatureStats(interval_clip={}, lesion_bin_edges=[], total_volume_bin_edges=[], max_volume_bin_edges=[])
        stats.lesion_bin_edges = q_edges(out["lesion_count"])
        stats.total_volume_bin_edges = q_edges(out["total_volume"])
        stats.max_volume_bin_edges = q_edges(out["max_volume"])

    def bin_by_edges(values: pd.Series, edges: List[float], prefix: str) -> pd.Series:
        if not edges or len(edges) < 2:
            return pd.Series("bin_unknown", index=values.index)
        edges = sorted(list(set(edges)))
        if len(edges) < 2:
            return pd.Series("bin_unknown", index=values.index)
        edges[0] = -np.inf
        edges[-1] = np.inf
        labels = [f"{prefix}_{i}" for i in range(len(edges) - 1)]
        return pd.cut(values, bins=edges, labels=labels, include_lowest=True).astype("object")

    out["lesion_count_bin"] = bin_by_edges(out["lesion_count"], stats.lesion_bin_edges, "lesion")
    out["total_volume_bin"] = bin_by_edges(out["total_volume"], stats.total_volume_bin_edges, "totalvol")
    out["max_volume_bin"] = bin_by_edges(out["max_volume"], stats.max_volume_bin_edges, "maxvol")

    # Karnovsky bins
    def karn_bin(v):
        if pd.isna(v):
            return np.nan
        v = float(v)
        if v <= 50:
            return "karn_<=50"
        if v <= 60:
            return "karn_60"
        if v <= 70:
            return "karn_70"
        if v <= 80:
            return "karn_80"
        if v <= 90:
            return "karn_90"
        return "karn_100"

    out["karnovsky_bin"] = out["karnovsky"].map(karn_bin).astype("object")

    # Missing flags for key numerics
    for c in ["lesion_count", "total_volume", "max_volume", "karnovsky", "age_at_rx_years"]:
        if c in out.columns:
            out[f"{c}_missing"] = out[c].isna().astype(float)

    # Keep normalized core categoricals explicitly
    out["diag_cat"] = out[c_diag].astype("object") if c_diag else "missing"
    out["treatment_cat"] = out[c_treat].astype("object") if c_treat else "missing"
    out["mutation_cat"] = out[c_mut].astype("object") if c_mut else "missing"
    out["extracranial_cat"] = out[c_ext].astype("object") if c_ext else "missing"
    out["sex_cat"] = out[c_sex].astype("object") if c_sex else "missing"

    return out, stats


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["lesion_x_total"] = out["lesion_count"] * out["total_volume"]
    out["lesion_x_max"] = out["lesion_count"] * out["max_volume"]
    out["lesion_x_kdef"] = out["lesion_count"] * out["karnovsky_deficit"]
    out["total_x_kdef"] = out["total_volume"] * out["karnovsky_deficit"]
    out["ovgm_x_lesion"] = out.get("ovgm_flag", 0.0) * out["lesion_count"]
    out["ovgm_x_total"] = out.get("ovgm_flag", 0.0) * out["total_volume"]
    out["operation_x_max"] = out.get("brain_operation_flag", 0.0) * out["max_volume"]

    out["mutations_x_treatment"] = out["mutation_cat"].fillna("missing").astype(str) + "__" + out["treatment_cat"].fillna("missing").astype(str)
    out["diagnosis_x_treatment"] = out["diag_cat"].fillna("missing").astype(str) + "__" + out["treatment_cat"].fillna("missing").astype(str)
    out["diagnosis_x_lesion_bin"] = out["diag_cat"].fillna("missing").astype(str) + "__" + out["lesion_count_bin"].astype(str)
    out["diagnosis_x_volume_bin"] = out["diag_cat"].fillna("missing").astype(str) + "__" + out["total_volume_bin"].astype(str)
    out["extracranial_x_diagnosis"] = out["extracranial_cat"].fillna("missing").astype(str) + "__" + out["diag_cat"].fillna("missing").astype(str)
    out["extracranial_x_treatment"] = out["extracranial_cat"].fillna("missing").astype(str) + "__" + out["treatment_cat"].fillna("missing").astype(str)
    return out


def _drop_raw_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_like = [c for c in out.columns if "дата" in norm_key(c)]
    return out.drop(columns=date_like, errors="ignore")


def _drop_leakage_and_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Outcome columns are removed explicitly to avoid target leakage from follow-up labels.
    drop_cols = get_leakage_columns_to_drop(out.columns)
    for c in ID_CANDIDATES:
        if c in out.columns:
            drop_cols.append(c)
    return out.drop(columns=drop_cols, errors="ignore")


def assert_no_leakage_features(df: pd.DataFrame, context: str) -> None:
    leaked = get_leakage_columns_to_drop(df.columns)
    if leaked:
        raise RuntimeError(f"Leakage columns still present in {context}: {leaked}")


def build_feature_frame(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureStats]:
    tr = build_base_features(train_df)
    va = build_base_features(valid_df)
    te = build_base_features(test_df)

    tr, stats, _ = add_date_features(tr, None)
    va, _, _ = add_date_features(va, stats)
    te, _, _ = add_date_features(te, stats)

    tr, stats = add_clinical_features(tr, stats)
    va, _ = add_clinical_features(va, stats)
    te, _ = add_clinical_features(te, stats)

    tr = add_interactions(tr)
    va = add_interactions(va)
    te = add_interactions(te)

    tr = _drop_raw_date_columns(tr)
    va = _drop_raw_date_columns(va)
    te = _drop_raw_date_columns(te)

    tr = _drop_leakage_and_id(tr)
    va = _drop_leakage_and_id(va)
    te = _drop_leakage_and_id(te)
    assert_no_leakage_features(tr, "train features")
    assert_no_leakage_features(va, "validation features")
    assert_no_leakage_features(te, "test features")

    # Ensure same columns
    all_cols = sorted(list(set(tr.columns) | set(va.columns) | set(te.columns)))
    tr = tr.reindex(columns=all_cols)
    va = va.reindex(columns=all_cols)
    te = te.reindex(columns=all_cols)
    return tr, va, te, stats


def make_feature_sets(df: pd.DataFrame, shifted_features: Optional[List[str]] = None) -> Dict[str, List[str]]:
    cols = list(df.columns)
    shifted = set(shifted_features or [])

    compact = [
        "age_at_rx_years",
        "age_at_diagnosis_years",
        "age_at_mgm_years",
        "diagnosis_to_mgm_days_raw",
        "mgm_to_rx_days_raw",
        "diagnosis_to_rx_days_raw",
        "primary_removal_to_rx_days_raw",
        "ovgm_to_rx_days_raw",
        "brain_operation_to_rx_days_raw",
        "primary_removed_flag",
        "ovgm_flag",
        "brain_operation_flag",
        "has_activating_mutations",
        "has_extracranial_metastases",
        "lesion_count",
        "total_volume",
        "max_volume",
        "avg_volume_per_lesion",
        "max_to_total_volume_ratio",
        "residual_volume",
        "karnovsky",
        "karnovsky_deficit",
        "diag_cat",
        "treatment_cat",
        "mutation_cat",
        "extracranial_cat",
        "sex_cat",
    ]
    compact = [c for c in compact if c in cols]

    full = cols.copy()
    no_shift = [c for c in cols if c not in shifted]

    return {
        "full": full,
        "compact": compact if compact else full,
        "no_shift": no_shift if no_shift else full,
    }


def fit_oof_target_encoder(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    cat_cols: List[str],
    n_splits: int = 5,
    seed: int = 42,
    smoothing: float = 20.0,
    min_samples_leaf: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global_mean = float(np.mean(y))
    oof = pd.DataFrame(index=X_train.index)
    test_encoded = pd.DataFrame(index=X_test.index)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in cat_cols:
        train_col = X_train[col].fillna("__nan__").astype(str)
        test_col = X_test[col].fillna("__nan__").astype(str)

        oof_vals = np.zeros(len(X_train), dtype=float)
        test_fold_vals = []

        for tr_idx, va_idx in splitter.split(X_train, y):
            tr_c = train_col.iloc[tr_idx]
            va_c = train_col.iloc[va_idx]
            tr_y = y[tr_idx]

            grp = pd.DataFrame({"c": tr_c.values, "y": tr_y}).groupby("c")["y"].agg(["mean", "count"])
            counts = grp["count"].values
            means = grp["mean"].values
            smooth = (means * counts + global_mean * smoothing) / (counts + smoothing)
            # min_samples_leaf damping
            damp = 1 / (1 + np.exp(-(counts - min_samples_leaf)))
            smooth = global_mean * (1 - damp) + smooth * damp
            mapping = dict(zip(grp.index.tolist(), smooth.tolist()))

            oof_vals[va_idx] = va_c.map(mapping).fillna(global_mean).values
            test_fold_vals.append(test_col.map(mapping).fillna(global_mean).values)

        oof[f"te_{col}"] = oof_vals
        test_encoded[f"te_{col}"] = np.mean(np.vstack(test_fold_vals), axis=0)

    return oof, test_encoded


# ---------------------------------------------------------------------------
# Validation, thresholding and shift diagnostics
# ---------------------------------------------------------------------------


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        verbose_feature_names_out=False,
    )
    return pre, num_cols, cat_cols


def find_best_threshold(y_true: np.ndarray, probas: np.ndarray) -> Tuple[float, float]:
    best_thr = 0.5
    best_score = -1.0
    for thr in np.arange(0.01, 0.991, 0.001):
        pred = (probas >= thr).astype(int)
        # Balanced Accuracy is used because class sizes are uneven and both
        # sensitivity and specificity matter for the competition target.
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, float(best_score)


def threshold_sweep(y_true: np.ndarray, probas: np.ndarray, tag: str) -> pd.DataFrame:
    rows = []
    for thr in np.arange(0.01, 0.991, 0.001):
        pred = (probas >= thr).astype(int)
        rows.append(
            {
                "tag": tag,
                "threshold": float(thr),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
                "positive_rate": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows)


def _predict_proba_generic(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    z = model.predict(X)
    return np.asarray(z, dtype=float)


def _build_group_labels(raw_df: pd.DataFrame) -> np.ndarray:
    cols = raw_df.columns.tolist()
    c_birth = find_col(cols, ["Дата рождения"])
    c_diag = find_col(cols, ["Онкологический диагноз"])
    c_sex = find_col(cols, ["Пол"])
    c_date_diag = find_col(cols, ["Дата постановки онкологического диагноза / начала первичного лечения"])
    group_cols = [c for c in [c_birth, c_diag, c_sex, c_date_diag] if c is not None]
    if not group_cols:
        return np.arange(len(raw_df))
    g = raw_df[group_cols].copy()
    for c in group_cols:
        g[c] = g[c].map(normalize_text_value).fillna("__nan__")
    key = g.astype(str).agg("|".join, axis=1)
    return key.values


def adversarial_validation(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame, feature_cols: Optional[List[str]] = None
) -> Tuple[float, pd.Series, np.ndarray]:
    tr = build_base_features(train_raw)
    te = build_base_features(test_raw)
    tr, stats, _ = add_date_features(tr, None)
    te, _, _ = add_date_features(te, stats)
    tr, stats = add_clinical_features(tr, stats)
    te, _ = add_clinical_features(te, stats)
    tr = add_interactions(tr)
    te = add_interactions(te)
    tr = _drop_raw_date_columns(tr)
    te = _drop_raw_date_columns(te)
    tr = _drop_leakage_and_id(tr)
    te = _drop_leakage_and_id(te)

    common_cols = sorted(list(set(tr.columns) & set(te.columns)))
    tr = tr[common_cols]
    te = te[common_cols]
    if feature_cols is not None:
        use = [c for c in feature_cols if c in tr.columns]
        if use:
            tr = tr[use]
            te = te[use]

    X_adv = pd.concat([tr, te], axis=0).reset_index(drop=True)
    y_adv = np.array([0] * len(tr) + [1] * len(te))

    pre, _, _ = build_preprocessor(X_adv)
    X_mat = pre.fit_transform(X_adv)
    if hasattr(X_mat, "toarray"):
        X_mat = X_mat.toarray()

    clf = LogisticRegression(max_iter=5000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_adv), dtype=float)
    for tr_idx, va_idx in cv.split(X_mat, y_adv):
        m = clone(clf)
        m.fit(X_mat[tr_idx], y_adv[tr_idx])
        oof[va_idx] = m.predict_proba(X_mat[va_idx])[:, 1]
    auc = roc_auc_score(y_adv, oof)

    # Fit full model for shift importance
    clf.fit(X_mat, y_adv)
    feats = pre.get_feature_names_out()
    coef = np.abs(clf.coef_[0])
    fi = pd.Series(coef, index=feats).sort_values(ascending=False)

    # Aggregate to base feature
    base_importance = {}
    for feat, val in fi.items():
        base = feat
        # Heuristic: OHE feature names often start with "<column>_"
        # map to longest matching original column prefix
        for c in common_cols:
            if feat.startswith(c + "_") or feat == c:
                base = c
                break
        base_importance[base] = base_importance.get(base, 0.0) + float(val)
    base_importance = pd.Series(base_importance).sort_values(ascending=False)

    # Train weights based on probability to look like test
    full_pred = clf.predict_proba(X_mat)[:, 1]
    train_weights = full_pred[: len(tr)]
    train_weights = train_weights / (np.mean(train_weights) + 1e-9)
    return float(auc), base_importance, train_weights


def _get_cv_splits(y: np.ndarray, groups: Optional[np.ndarray] = None) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    out = {}

    all_splits = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        seed_splits = list(skf.split(np.zeros(len(y)), y))
        out[f"skf5_seed_{seed}"] = seed_splits
        all_splits.extend(seed_splits)
    out["skf5_all_seeds"] = all_splits

    skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    out["skf10_seed_42"] = list(skf10.split(np.zeros(len(y)), y))

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=2025)
    out["repeated_stratified_5x5"] = list(rskf.split(np.zeros(len(y)), y))

    if groups is not None:
        unique_groups = pd.Series(groups).nunique()
        if unique_groups >= 10:
            gkf = GroupKFold(n_splits=5)
            out["groupkfold5"] = list(gkf.split(np.zeros(len(y)), y, groups=groups))

    return out


def _apply_feature_set(
    Xtr: pd.DataFrame,
    Xva: pd.DataFrame,
    Xte: pd.DataFrame,
    feature_set_name: str,
    shifted_features: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fsets = make_feature_sets(Xtr, shifted_features=shifted_features)
    cols = fsets.get(feature_set_name, fsets["full"])
    cols = [c for c in cols if c in Xtr.columns]
    if not cols:
        cols = list(Xtr.columns)
    Xtr = Xtr[cols].copy()
    Xva = Xva[cols].copy()
    Xte = Xte[cols].copy()
    return Xtr, Xva, Xte


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_sklearn_models(
    model_name: str,
    model,
    train_raw_labeled: pd.DataFrame,
    y: np.ndarray,
    test_raw: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    feature_set_name: str = "full",
    shifted_features: Optional[List[str]] = None,
    use_target_encoding: bool = False,
    sample_weights: Optional[np.ndarray] = None,
    cv_strategy_name: str = "custom",
) -> ModelResult:
    oof = np.zeros(len(train_raw_labeled), dtype=float)
    oof_cnt = np.zeros(len(train_raw_labeled), dtype=float)
    test_accum = np.zeros(len(test_raw), dtype=float)
    fold_scores = []
    fold_thrs = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits, 1):
        raw_tr = train_raw_labeled.iloc[tr_idx].copy()
        raw_va = train_raw_labeled.iloc[va_idx].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        Xtr, Xva, Xte, _ = build_feature_frame(raw_tr, raw_va, test_raw.copy())
        Xtr, Xva, Xte = _apply_feature_set(Xtr, Xva, Xte, feature_set_name, shifted_features)

        if use_target_encoding:
            cat_cols = [c for c in Xtr.columns if not is_numeric_dtype(Xtr[c])]
            te_tr, te_te = fit_oof_target_encoder(
                X_train=Xtr.reset_index(drop=True),
                X_test=Xte.reset_index(drop=True),
                y=y_tr,
                cat_cols=cat_cols,
                n_splits=min(5, max(2, int(np.min(np.bincount(y_tr))))),
                seed=42,
                smoothing=20.0,
                min_samples_leaf=5,
            )

            # TE for validation from full-train mapping (no leakage from val target)
            global_mean = float(np.mean(y_tr))
            te_va = pd.DataFrame(index=Xva.index)
            for c in cat_cols:
                tr_c = Xtr[c].fillna("__nan__").astype(str)
                va_c = Xva[c].fillna("__nan__").astype(str)
                grp = pd.DataFrame({"c": tr_c.values, "y": y_tr}).groupby("c")["y"].agg(["mean", "count"])
                smooth = (grp["mean"] * grp["count"] + global_mean * 20.0) / (grp["count"] + 20.0)
                te_va[f"te_{c}"] = va_c.map(smooth).fillna(global_mean).values

            num_cols = [c for c in Xtr.columns if is_numeric_dtype(Xtr[c])]
            Xtr_num = Xtr[num_cols].reset_index(drop=True).copy()
            Xva_num = Xva[num_cols].reset_index(drop=True).copy()
            Xte_num = Xte[num_cols].reset_index(drop=True).copy()

            Xtr_fin = pd.concat([Xtr_num, te_tr], axis=1)
            Xva_fin = pd.concat([Xva_num, te_va.reset_index(drop=True)], axis=1)
            Xte_fin = pd.concat([Xte_num, te_te], axis=1)

            imp = SimpleImputer(strategy="median")
            sc = StandardScaler()
            Xtr_m = sc.fit_transform(imp.fit_transform(Xtr_fin))
            Xva_m = sc.transform(imp.transform(Xva_fin))
            Xte_m = sc.transform(imp.transform(Xte_fin))
        else:
            pre, _, _ = build_preprocessor(Xtr)
            Xtr_m = pre.fit_transform(Xtr)
            Xva_m = pre.transform(Xva)
            Xte_m = pre.transform(Xte)
            if hasattr(Xtr_m, "toarray"):
                Xtr_m = Xtr_m.toarray()
                Xva_m = Xva_m.toarray()
                Xte_m = Xte_m.toarray()

        mdl = clone(model)
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[tr_idx]
        mdl.fit(Xtr_m, y_tr, **fit_kwargs)

        p_va = _predict_proba_generic(mdl, Xva_m)
        p_te = _predict_proba_generic(mdl, Xte_m)
        fold_threshold, fold_score = find_best_threshold(y_va, p_va)

        oof[va_idx] += p_va
        oof_cnt[va_idx] += 1.0
        test_accum += p_te
        fold_scores.append(fold_score)
        fold_thrs.append(fold_threshold)

    oof = oof / np.maximum(oof_cnt, 1.0)
    test_proba = test_accum / len(splits)
    g_thr, g_sc = find_best_threshold(y, oof)
    test_pos = float((test_proba >= g_thr).mean())
    return ModelResult(
        name=model_name,
        model_family="sklearn_te" if use_target_encoding else "sklearn_ohe",
        feature_set=feature_set_name,
        cv_strategy=cv_strategy_name,
        n_folds=len(splits),
        oof_proba=oof,
        test_proba=test_proba,
        best_threshold=g_thr,
        oof_bal_acc=g_sc,
        test_pos_rate=test_pos,
        fold_scores=fold_scores,
        fold_thresholds=fold_thrs,
        notes="weighted" if sample_weights is not None else "",
    )


def train_catboost(
    model_name: str,
    cat_params: Dict,
    train_raw_labeled: pd.DataFrame,
    y: np.ndarray,
    test_raw: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    feature_set_name: str = "full",
    shifted_features: Optional[List[str]] = None,
    sample_weights: Optional[np.ndarray] = None,
    cv_strategy_name: str = "custom",
) -> ModelResult:
    try:
        from catboost import CatBoostClassifier
    except Exception as e:
        raise RuntimeError(f"CatBoost not available: {e}")

    oof = np.zeros(len(train_raw_labeled), dtype=float)
    oof_cnt = np.zeros(len(train_raw_labeled), dtype=float)
    test_accum = np.zeros(len(test_raw), dtype=float)
    fold_scores = []
    fold_thrs = []

    base_params = dict(
        loss_function="Logloss",
        eval_metric="Logloss",
        auto_class_weights="Balanced",
        iterations=3000,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=5,
        random_strength=0.5,
        bagging_temperature=0.2,
        border_count=128,
        random_seed=42,
        verbose=False,
    )
    base_params.update(cat_params)

    for fold_id, (tr_idx, va_idx) in enumerate(splits, 1):
        raw_tr = train_raw_labeled.iloc[tr_idx].copy()
        raw_va = train_raw_labeled.iloc[va_idx].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        Xtr, Xva, Xte, _ = build_feature_frame(raw_tr, raw_va, test_raw.copy())
        Xtr, Xva, Xte = _apply_feature_set(Xtr, Xva, Xte, feature_set_name, shifted_features)

        cat_cols = [c for c in Xtr.columns if not is_numeric_dtype(Xtr[c])]
        for c in cat_cols:
            Xtr[c] = Xtr[c].fillna("missing").astype(str)
            Xva[c] = Xva[c].fillna("missing").astype(str)
            Xte[c] = Xte[c].fillna("missing").astype(str)

        for c in Xtr.columns:
            if c not in cat_cols:
                Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce")
                Xva[c] = pd.to_numeric(Xva[c], errors="coerce")
                Xte[c] = pd.to_numeric(Xte[c], errors="coerce")

        mdl = CatBoostClassifier(**base_params)
        fit_kwargs = {
            "X": Xtr,
            "y": y_tr,
            "cat_features": cat_cols,
            "eval_set": (Xva, y_va),
            "use_best_model": True,
            "early_stopping_rounds": 150,
        }
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[tr_idx]
        mdl.fit(**fit_kwargs)

        p_va = mdl.predict_proba(Xva)[:, 1]
        p_te = mdl.predict_proba(Xte)[:, 1]
        fold_threshold, fold_score = find_best_threshold(y_va, p_va)

        oof[va_idx] += p_va
        oof_cnt[va_idx] += 1.0
        test_accum += p_te
        fold_scores.append(fold_score)
        fold_thrs.append(fold_threshold)

    oof = oof / np.maximum(oof_cnt, 1.0)
    test_proba = test_accum / len(splits)
    g_thr, g_sc = find_best_threshold(y, oof)
    test_pos = float((test_proba >= g_thr).mean())
    return ModelResult(
        name=model_name,
        model_family="catboost",
        feature_set=feature_set_name,
        cv_strategy=cv_strategy_name,
        n_folds=len(splits),
        oof_proba=oof,
        test_proba=test_proba,
        best_threshold=g_thr,
        oof_bal_acc=g_sc,
        test_pos_rate=test_pos,
        fold_scores=fold_scores,
        fold_thresholds=fold_thrs,
        notes="weighted" if sample_weights is not None else "",
    )


def train_lightgbm(
    model_name: str,
    model,
    train_raw_labeled: pd.DataFrame,
    y: np.ndarray,
    test_raw: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    feature_set_name: str = "full",
    shifted_features: Optional[List[str]] = None,
    cv_strategy_name: str = "custom",
) -> ModelResult:
    # Uses generic sklearn-style path
    return train_sklearn_models(
        model_name=model_name,
        model=model,
        train_raw_labeled=train_raw_labeled,
        y=y,
        test_raw=test_raw,
        splits=splits,
        feature_set_name=feature_set_name,
        shifted_features=shifted_features,
        use_target_encoding=False,
        sample_weights=None,
        cv_strategy_name=cv_strategy_name,
    )


def train_xgboost(
    model_name: str,
    model,
    train_raw_labeled: pd.DataFrame,
    y: np.ndarray,
    test_raw: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    feature_set_name: str = "full",
    shifted_features: Optional[List[str]] = None,
    cv_strategy_name: str = "custom",
) -> ModelResult:
    # Uses generic sklearn-style path
    return train_sklearn_models(
        model_name=model_name,
        model=model,
        train_raw_labeled=train_raw_labeled,
        y=y,
        test_raw=test_raw,
        splits=splits,
        feature_set_name=feature_set_name,
        shifted_features=shifted_features,
        use_target_encoding=False,
        sample_weights=None,
        cv_strategy_name=cv_strategy_name,
    )


def build_ensembles(results: List[ModelResult], y_true: np.ndarray) -> List[ModelResult]:
    if not results:
        return []

    # Sort by score
    ranked = sorted(results, key=lambda r: r.oof_bal_acc, reverse=True)
    base = ranked

    def make_result(name: str, oof: np.ndarray, test: np.ndarray, notes: str = "") -> ModelResult:
        best_threshold, oof_score = find_best_threshold(y_true, oof)
        return ModelResult(
            name=name,
            model_family="ensemble",
            feature_set="mixed",
            cv_strategy="from_base_oof",
            n_folds=0,
            oof_proba=oof,
            test_proba=test,
            best_threshold=best_threshold,
            oof_bal_acc=oof_score,
            test_pos_rate=float((test >= best_threshold).mean()),
            fold_scores=[],
            fold_thresholds=[],
            notes=notes,
        )

    oof_mat = np.vstack([r.oof_proba for r in base]).T
    test_mat = np.vstack([r.test_proba for r in base]).T
    scores = np.array([max(r.oof_bal_acc, 1e-8) for r in base], dtype=float)
    weights = scores / scores.sum()

    ens = []
    ens.append(make_result("ensemble_avg_top3", oof_mat[:, :3].mean(axis=1), test_mat[:, :3].mean(axis=1), "top3"))
    ens.append(make_result("ensemble_avg_top5", oof_mat[:, : min(5, oof_mat.shape[1])].mean(axis=1), test_mat[:, : min(5, test_mat.shape[1])].mean(axis=1), "top5"))
    ens.append(make_result("ensemble_weighted", oof_mat @ weights, test_mat @ weights, "weighted_by_oof"))

    rank_oof = pd.DataFrame(oof_mat).rank(axis=0, method="average", pct=True).mean(axis=1).values
    rank_test = pd.DataFrame(test_mat).rank(axis=0, method="average", pct=True).mean(axis=1).values
    ens.append(make_result("ensemble_rank_average", rank_oof, rank_test, "rank"))

    vote_oof = np.zeros(oof_mat.shape[0], dtype=float)
    vote_test = np.zeros(test_mat.shape[0], dtype=float)
    for r in base:
        vote_oof += (r.oof_proba >= r.best_threshold).astype(float)
        vote_test += (r.test_proba >= r.best_threshold).astype(float)
    vote_oof /= len(base)
    vote_test /= len(base)
    ens.append(make_result("ensemble_threshold_vote", vote_oof, vote_test, "vote"))

    # Stacking uses only OOF base predictions, so the meta-model does not see
    # in-fold predictions from models trained on the same row.
    meta = LogisticRegression(class_weight="balanced", max_iter=5000, solver="liblinear", random_state=42)
    meta.fit(oof_mat, y_true)
    stack_oof = meta.predict_proba(oof_mat)[:, 1]
    stack_test = meta.predict_proba(test_mat)[:, 1]
    ens.append(make_result("ensemble_stacking_lr", stack_oof, stack_test, "stacking"))
    return ens


# ---------------------------------------------------------------------------
# Reporting and submission files
# ---------------------------------------------------------------------------


def save_submission(path: str, ids: np.ndarray, probas: np.ndarray, threshold: float) -> pd.DataFrame:
    pred = (probas >= threshold).astype(int)
    sub = pd.DataFrame({"ID": ids.astype(int), "Прогрессия": pred.astype(int)})
    sub.to_csv(path, index=False, encoding="utf-8")
    return sub


def save_reports(
    report_lines: List[str],
    model_results: List[ModelResult],
    ensemble_results: List[ModelResult],
    y_true: np.ndarray,
    adversarial_auc: float,
    shifted_importance: pd.Series,
    parse_success_by_date: Dict[str, float],
) -> None:
    best_result = sorted(model_results + ensemble_results, key=lambda r: r.oof_bal_acc, reverse=True)[0]
    best_ensemble = sorted(ensemble_results, key=lambda r: r.oof_bal_acc, reverse=True)[0] if ensemble_results else None

    # advanced_report.txt
    with open(f"{REPORTS_DIR}/advanced_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
        f.write("\n=== Best Result ===\n")
        f.write(f"name: {best_result.name}\n")
        f.write(f"family: {best_result.model_family}\n")
        f.write(f"OOF Balanced Accuracy: {best_result.oof_bal_acc:.6f}\n")
        f.write(f"threshold_selected_on_oof: {best_result.best_threshold:.6f}\n")
        f.write(f"test_positive_rate: {best_result.test_pos_rate:.6f}\n")
        if best_ensemble is not None:
            f.write("\n=== Best Ensemble ===\n")
            f.write(f"name: {best_ensemble.name}\n")
            f.write(f"OOF Balanced Accuracy: {best_ensemble.oof_bal_acc:.6f}\n")
            f.write(f"threshold_selected_on_oof: {best_ensemble.best_threshold:.6f}\n")
            f.write(f"test_positive_rate: {best_ensemble.test_pos_rate:.6f}\n")
            f.write(f"notes: {best_ensemble.notes}\n")
        f.write("\n=== Adversarial Validation ===\n")
        f.write(f"AUC: {adversarial_auc:.6f}\n")
        f.write("\nTop shifted features:\n")
        for k, v in shifted_importance.head(30).items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\nDate parsing success rates:\n")
        for c, r in parse_success_by_date.items():
            f.write(f"{c}: {r:.4f}\n")

    # model_scores.csv
    score_rows = []
    for r in model_results + ensemble_results:
        score_rows.append(
            {
                "name": r.name,
                "model_family": r.model_family,
                "feature_set": r.feature_set,
                "cv_strategy": r.cv_strategy,
                "n_folds": r.n_folds,
                "oof_bal_acc": r.oof_bal_acc,
                "best_threshold": r.best_threshold,
                "test_pos_rate": r.test_pos_rate,
                "notes": r.notes,
            }
        )
    scores_df = pd.DataFrame(score_rows).sort_values("oof_bal_acc", ascending=False)
    scores_df.to_csv(f"{REPORTS_DIR}/model_scores.csv", index=False, encoding="utf-8")

    # OOF / test probabilities
    oof_df = pd.DataFrame({"y_true": y_true.astype(int)})
    test_df = pd.DataFrame()
    for r in model_results + ensemble_results:
        oof_df[f"oof_{r.name}"] = r.oof_proba
        test_df[f"test_{r.name}"] = r.test_proba
    oof_df.to_csv(f"{REPORTS_DIR}/oof_predictions.csv", index=False, encoding="utf-8")
    test_df.to_csv(f"{REPORTS_DIR}/test_probabilities.csv", index=False, encoding="utf-8")

    # adversarial_validation.csv
    adv_df = shifted_importance.reset_index()
    adv_df.columns = ["feature", "shift_importance"]
    adv_df.insert(0, "adversarial_auc", adversarial_auc)
    adv_df.to_csv(f"{REPORTS_DIR}/adversarial_validation.csv", index=False, encoding="utf-8")


def _train_feature_importance_for_best(
    best_result: ModelResult,
    train_raw_labeled: pd.DataFrame,
    y: np.ndarray,
    test_raw: pd.DataFrame,
    shifted_features: List[str],
) -> pd.DataFrame:
    # Train on full labeled data, extract feature importance where possible
    raw_tr = train_raw_labeled.copy()
    raw_va = train_raw_labeled.copy().iloc[: min(20, len(train_raw_labeled))].copy()
    Xtr, _, Xte, _ = build_feature_frame(raw_tr, raw_va, test_raw.copy())
    Xtr, _, _, = _apply_feature_set(Xtr, Xtr.copy(), Xte.copy(), best_result.feature_set, shifted_features)

    if best_result.model_family == "catboost":
        try:
            from catboost import CatBoostClassifier

            cat_cols = [c for c in Xtr.columns if not is_numeric_dtype(Xtr[c])]
            for c in cat_cols:
                Xtr[c] = Xtr[c].fillna("missing").astype(str)
            for c in Xtr.columns:
                if c not in cat_cols:
                    Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce")

            # Stable, moderately regularized settings
            model = CatBoostClassifier(
                iterations=2000,
                depth=4,
                learning_rate=0.05,
                l2_leaf_reg=3,
                random_strength=0.5,
                bagging_temperature=0.2,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                verbose=False,
                random_seed=42,
            )
            model.fit(Xtr, y, cat_features=cat_cols)
            fi = model.get_feature_importance(prettified=True)
            fi = fi.rename(columns={"Feature Id": "feature", "Importances": "importance"})
            fi = fi[["feature", "importance"]].sort_values("importance", ascending=False).head(50)
            return fi
        except Exception:
            pass

    # fallback: ExtraTrees for importances on same feature set
    pre, _, _ = build_preprocessor(Xtr)
    Xmat = pre.fit_transform(Xtr)
    if hasattr(Xmat, "toarray"):
        Xmat = Xmat.toarray()
    model = ExtraTreesClassifier(
        n_estimators=1200,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xmat, y)
    feats = pre.get_feature_names_out()
    imp = model.feature_importances_
    fi = pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False).head(50)
    return fi


def _positive_rate_threshold(target_pos_rate: float, probas: np.ndarray) -> float:
    probs_sorted = np.sort(probas)
    idx = int(np.clip((1.0 - target_pos_rate) * len(probas), 0, len(probas) - 1))
    return float(probs_sorted[idx])


def main():
    ensure_dirs()
    set_seed(42)

    train = normalize_column_names(read_csv_robust(TRAIN_PATH))
    test = normalize_column_names(read_csv_robust(TEST_PATH))

    y_raw, target_source = prepare_target(train)
    labeled_mask = y_raw.notna()
    dropped_unlabeled = int((~labeled_mask).sum())
    train_labeled = train.loc[labeled_mask].reset_index(drop=True).copy()
    y = y_raw.loc[labeled_mask].astype(int).to_numpy()

    # ID handling
    id_col = None
    for c in ID_CANDIDATES:
        if c in test.columns:
            id_col = c
            break
    if id_col is None:
        test_ids = np.arange(1, len(test) + 1)
    else:
        test_ids = test[id_col].astype(int).values

    leakage_info = detect_leakage_columns(list(train.columns), list(test.columns))
    save_leakage_report(leakage_info)

    # Data audit details
    report_lines = []
    report_lines.append(f"train_shape: {train.shape}")
    report_lines.append(f"test_shape: {test.shape}")
    report_lines.append(f"target_source: {target_source}")
    report_lines.append(f"labeled_rows: {len(train_labeled)}")
    report_lines.append(f"dropped_unlabeled_rows: {dropped_unlabeled}")
    report_lines.append(f"target_distribution:\n{pd.Series(y).value_counts().to_string()}")
    report_lines.append(f"explicit_leakage_drop: {leakage_info['explicit_drop']}")
    report_lines.append(f"suspicious_columns: {leakage_info['suspicious']}")
    report_lines.append(f"train_only_columns: {leakage_info['train_only']}")
    report_lines.append(f"test_only_columns: {leakage_info['test_only']}")
    report_lines.append(f"columns_train: {list(train.columns)}")
    report_lines.append(f"dtypes_train:\n{train.dtypes.to_string()}")
    report_lines.append(f"dtypes_test:\n{test.dtypes.to_string()}")
    report_lines.append(f"missing_train:\n{train.isna().sum().sort_values(ascending=False).to_string()}")
    report_lines.append(f"missing_test:\n{test.isna().sum().sort_values(ascending=False).to_string()}")
    report_lines.append(f"duplicate_rows_train: {int(train.duplicated().sum())}")
    report_lines.append(f"duplicate_rows_test: {int(test.duplicated().sum())}")
    for c in ID_CANDIDATES:
        if c in train.columns:
            report_lines.append(f"duplicate_{c}_train: {int(train[c].duplicated().sum())}")
        if c in test.columns:
            report_lines.append(f"duplicate_{c}_test: {int(test[c].duplicated().sum())}")
    if id_col is not None:
        report_lines.append(f"test_id_col: {id_col}")
        report_lines.append(f"test_id_min: {int(np.min(test_ids))}, max: {int(np.max(test_ids))}, n_unique: {int(pd.Series(test_ids).nunique())}")
        report_lines.append(f"test_id_starts_from_1: {bool(np.min(test_ids) == 1)}")
        report_lines.append(f"test_id_is_sequential: {bool(np.array_equal(np.sort(test_ids), np.arange(np.min(test_ids), np.max(test_ids)+1)))}")
    else:
        report_lines.append("test_id_col: not found, synthetic 1..N will be used")

    # object unique values
    report_lines.append("object_unique_values_top20:")
    for c in train.columns:
        if train[c].dtype == object or is_string_dtype(train[c]):
            vc = train[c].map(normalize_text_value).value_counts(dropna=False).head(20)
            report_lines.append(f"[{c}]")
            report_lines.append(vc.to_string())

    # numeric parsing success rate and date parsing success
    parse_success_by_date = {}
    audit_features = build_base_features(train_labeled.copy())
    date_cols = [c for c in audit_features.columns if "дата" in norm_key(c)]
    for c in date_cols:
        dt = _parse_date_series(audit_features[c])
        parse_success_by_date[c] = float(dt.notna().mean())
    report_lines.append("date_parsing_success_rate:")
    for c, r in parse_success_by_date.items():
        report_lines.append(f"{c}: {r:.4f}")

    numeric_parse_report = []
    for c in audit_features.columns:
        if audit_features[c].dtype == object or is_string_dtype(audit_features[c]):
            parsed = audit_features[c].map(parse_numeric_value)
            succ = float(parsed.notna().mean())
            if succ > 0.3:
                numeric_parse_report.append((c, succ))
    numeric_parse_report = sorted(numeric_parse_report, key=lambda x: -x[1])
    report_lines.append("numeric_parsing_success_rate_over_0.3:")
    for c, s in numeric_parse_report:
        report_lines.append(f"{c}: {s:.4f}")

    # train-test duplicate patient rows after dropping target/leak/id
    tr_no_target = _drop_leakage_and_id(build_base_features(train_labeled.copy()))
    te_no_target = _drop_leakage_and_id(build_base_features(test.copy()))
    common = sorted(list(set(tr_no_target.columns) & set(te_no_target.columns)))
    tr_common = tr_no_target[common].copy()
    te_common = te_no_target[common].copy()
    for c in common:
        if tr_common[c].dtype == object or is_string_dtype(tr_common[c]):
            tr_common[c] = tr_common[c].fillna("__nan__").astype(str)
            te_common[c] = te_common[c].fillna("__nan__").astype(str)
    tr_hash = pd.util.hash_pandas_object(tr_common, index=False)
    te_hash = pd.util.hash_pandas_object(te_common, index=False)
    overlap = int(pd.Index(tr_hash).isin(set(te_hash)).sum())
    report_lines.append(f"exact_duplicate_like_rows_between_train_labeled_and_test: {overlap}")

    # Adversarial validation
    adv_auc, shifted_importance, adv_weights = adversarial_validation(train_labeled, test)
    report_lines.append(f"adversarial_auc: {adv_auc:.6f}")
    shifted_top = shifted_importance.head(8).index.tolist()
    report_lines.append(f"top_shifted_features: {shifted_top}")
    shifted_for_drop = shifted_importance.head(5).index.tolist() if adv_auc >= 0.58 else []

    # CV strategies
    groups = _build_group_labels(train_labeled)
    cv_splits = _get_cv_splits(y, groups=groups)
    report_lines.append(f"cv_strategies: {list(cv_splits.keys())}")

    # Model experiments
    all_results: List[ModelResult] = []
    feature_set_rows = []

    # 1) Logistic baselines and variants (fast, stable)
    lr_base = LogisticRegression(class_weight="balanced", C=0.8, max_iter=6000, solver="liblinear", random_state=42)
    for fs in ["compact", "full", "no_shift"]:
        r = train_sklearn_models(
            model_name=f"logreg_{fs}",
            model=lr_base,
            train_raw_labeled=train_labeled,
            y=y,
            test_raw=test,
            splits=cv_splits["skf5_all_seeds"],
            feature_set_name=fs,
            shifted_features=shifted_for_drop,
            use_target_encoding=False,
            sample_weights=None,
            cv_strategy_name="skf5_all_seeds",
        )
        all_results.append(r)
        feature_set_rows.append({"feature_set": fs, "model": r.name, "oof_bal_acc": r.oof_bal_acc})

    # Target encoding logistic
    r_te = train_sklearn_models(
        model_name="logreg_target_encoding_full",
        model=LogisticRegression(class_weight="balanced", C=1.0, max_iter=6000, solver="liblinear", random_state=42),
        train_raw_labeled=train_labeled,
        y=y,
        test_raw=test,
        splits=cv_splits["skf5_all_seeds"],
        feature_set_name="full",
        shifted_features=shifted_for_drop,
        use_target_encoding=True,
        sample_weights=None,
        cv_strategy_name="skf5_all_seeds",
    )
    all_results.append(r_te)

    # 2) Additional sklearn models
    sk_models = {
        "histgb_full": HistGradientBoostingClassifier(
            learning_rate=0.03, max_depth=4, max_leaf_nodes=31, min_samples_leaf=10, random_state=42
        ),
        "rf_compact": RandomForestClassifier(
            n_estimators=1600, max_depth=7, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "extratrees_full": ExtraTreesClassifier(
            n_estimators=1800, max_depth=9, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "svm_rbf_compact": SVC(C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=42),
        "knn_compact": KNeighborsClassifier(n_neighbors=17, weights="distance"),
    }
    for nm, mdl in sk_models.items():
        fs = "compact" if "compact" in nm else "full"
        r = train_sklearn_models(
            model_name=nm,
            model=mdl,
            train_raw_labeled=train_labeled,
            y=y,
            test_raw=test,
            splits=cv_splits["skf5_seed_42"],  # diagnostic, single seed
            feature_set_name=fs,
            shifted_features=shifted_for_drop,
            use_target_encoding=False,
            sample_weights=None,
            cv_strategy_name="skf5_seed_42",
        )
        all_results.append(r)
        feature_set_rows.append({"feature_set": fs, "model": r.name, "oof_bal_acc": r.oof_bal_acc})

    # 3) CatBoost search - Stage A quick screening (1 seed)
    catboost_available = True
    catboost_errors = []
    cat_stageA_results = []
    cat_param_grid = [
        {"depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 3, "random_strength": 0.3, "bagging_temperature": 0.0},
        {"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3, "random_strength": 0.5, "bagging_temperature": 0.2},
        {"depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 5, "random_strength": 0.8, "bagging_temperature": 0.5},
        {"depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 10, "random_strength": 1.0, "bagging_temperature": 1.0},
        {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 20, "random_strength": 0.5, "bagging_temperature": 0.2},
        {"depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 1, "random_strength": 0.2, "bagging_temperature": 0.0},
    ]

    for i, prm in enumerate(cat_param_grid, 1):
        for fs in ["compact", "full", "no_shift"]:
            model_name = f"catboost_A{i}_{fs}"
            try:
                r = train_catboost(
                    model_name=model_name,
                    cat_params=prm,
                    train_raw_labeled=train_labeled,
                    y=y,
                    test_raw=test,
                    splits=cv_splits["skf5_seed_42"],
                    feature_set_name=fs,
                    shifted_features=shifted_for_drop,
                    sample_weights=None,
                    cv_strategy_name="skf5_seed_42",
                )
                cat_stageA_results.append(r)
                feature_set_rows.append({"feature_set": fs, "model": r.name, "oof_bal_acc": r.oof_bal_acc})
            except Exception as e:
                catboost_available = False
                catboost_errors.append(str(e))

    if cat_stageA_results:
        cat_stageA_results = sorted(cat_stageA_results, key=lambda r: r.oof_bal_acc, reverse=True)
        top_stageA = cat_stageA_results[:3]
        # Stage B robust evaluation on all seeds and adversarial weighting variants
        for base_res in top_stageA:
            # recover params from name index
            idx = int(base_res.name.split("_")[1][1:]) - 1
            prm = cat_param_grid[idx]
            fs = base_res.feature_set

            r_full = train_catboost(
                model_name=f"catboost_B_{idx+1}_{fs}_allseeds",
                cat_params=prm,
                train_raw_labeled=train_labeled,
                y=y,
                test_raw=test,
                splits=cv_splits["skf5_all_seeds"],
                feature_set_name=fs,
                shifted_features=shifted_for_drop,
                sample_weights=None,
                cv_strategy_name="skf5_all_seeds",
            )
            all_results.append(r_full)

            r_w = train_catboost(
                model_name=f"catboost_B_{idx+1}_{fs}_advweighted",
                cat_params=prm,
                train_raw_labeled=train_labeled,
                y=y,
                test_raw=test,
                splits=cv_splits["skf5_all_seeds"],
                feature_set_name=fs,
                shifted_features=shifted_for_drop,
                sample_weights=adv_weights,
                cv_strategy_name="skf5_all_seeds",
            )
            all_results.append(r_w)

    # Add stageA cat models too
    all_results.extend(cat_stageA_results)

    # 4) LightGBM / XGBoost optional
    optional_errors = {}
    try:
        from lightgbm import LGBMClassifier

        lgb = LGBMClassifier(
            objective="binary",
            n_estimators=1400,
            learning_rate=0.03,
            num_leaves=15,
            class_weight="balanced",
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=20,
            max_depth=4,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=42,
        )
        r_lgb = train_lightgbm(
            model_name="lightgbm_full",
            model=lgb,
            train_raw_labeled=train_labeled,
            y=y,
            test_raw=test,
            splits=cv_splits["skf5_seed_42"],
            feature_set_name="full",
            shifted_features=shifted_for_drop,
            cv_strategy_name="skf5_seed_42",
        )
        all_results.append(r_lgb)
    except Exception as e:
        optional_errors["lightgbm"] = repr(e)

    try:
        from xgboost import XGBClassifier

        ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
        xgb = XGBClassifier(
            n_estimators=1600,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=ratio,
            random_state=42,
        )
        r_xgb = train_xgboost(
            model_name="xgboost_full",
            model=xgb,
            train_raw_labeled=train_labeled,
            y=y,
            test_raw=test,
            splits=cv_splits["skf5_seed_42"],
            feature_set_name="full",
            shifted_features=shifted_for_drop,
            cv_strategy_name="skf5_seed_42",
        )
        all_results.append(r_xgb)
    except Exception as e:
        optional_errors["xgboost"] = repr(e)

    # 5) CV strategy robustness probes
    # Re-evaluate strongest available single model across extra CV strategies
    if all_results:
        best_now = sorted(all_results, key=lambda r: r.oof_bal_acc, reverse=True)[0]
        probe_model = LogisticRegression(class_weight="balanced", C=0.8, max_iter=6000, solver="liblinear", random_state=42)
        r_skf10 = train_sklearn_models(
            model_name="logreg_probe_skf10",
            model=probe_model,
            train_raw_labeled=train_labeled,
            y=y,
            test_raw=test,
            splits=cv_splits["skf10_seed_42"],
            feature_set_name="compact",
            shifted_features=shifted_for_drop,
            use_target_encoding=False,
            sample_weights=None,
            cv_strategy_name="skf10_seed_42",
        )
        r_rep = train_sklearn_models(
            model_name="logreg_probe_repeated5x5",
            model=probe_model,
            train_raw_labeled=train_labeled,
            y=y,
            test_raw=test,
            splits=cv_splits["repeated_stratified_5x5"],
            feature_set_name="compact",
            shifted_features=shifted_for_drop,
            use_target_encoding=False,
            sample_weights=None,
            cv_strategy_name="repeated_stratified_5x5",
        )
        all_results.append(r_skf10)
        all_results.append(r_rep)

        if "groupkfold5" in cv_splits:
            r_group = train_sklearn_models(
                model_name="logreg_probe_groupkfold5",
                model=probe_model,
                train_raw_labeled=train_labeled,
                y=y,
                test_raw=test,
                splits=cv_splits["groupkfold5"],
                feature_set_name="compact",
                shifted_features=shifted_for_drop,
                use_target_encoding=False,
                sample_weights=None,
                cv_strategy_name="groupkfold5",
            )
            all_results.append(r_group)

    # Feature set score report
    pd.DataFrame(feature_set_rows).sort_values("oof_bal_acc", ascending=False).to_csv(
        f"{REPORTS_DIR}/feature_set_scores.csv", index=False, encoding="utf-8"
    )

    # Ensembles
    ensemble_results = build_ensembles(all_results, y)

    # Ranking and recommendations
    ranked_all = sorted(all_results + ensemble_results, key=lambda r: r.oof_bal_acc, reverse=True)
    best = ranked_all[0]

    # Save core reports
    save_reports(
        report_lines=report_lines,
        model_results=all_results,
        ensemble_results=ensemble_results,
        y_true=y,
        adversarial_auc=adv_auc,
        shifted_importance=shifted_importance,
        parse_success_by_date=parse_success_by_date,
    )

    # Threshold sweeps for top models + best
    sweep_frames = []
    for r in ranked_all[: min(6, len(ranked_all))]:
        sweep_frames.append(threshold_sweep(y, r.oof_proba, tag=r.name))
    pd.concat(sweep_frames, ignore_index=True).to_csv(f"{REPORTS_DIR}/threshold_sweep.csv", index=False, encoding="utf-8")

    # Feature importance for best model
    fi_df = _train_feature_importance_for_best(best, train_labeled, y, test, shifted_for_drop)
    fi_df.to_csv(f"{REPORTS_DIR}/feature_importance.csv", index=False, encoding="utf-8")

    # Save baseline final submission as best OOF-stable
    final_sub = save_submission(MAIN_SUBMISSION_PATH, test_ids, best.test_proba, best.best_threshold)

    # Save many candidate submissions
    # 1) top ranked models/ensembles
    submission_meta = []
    for i, r in enumerate(ranked_all[:20], 1):
        fname = f"{SUBMISSIONS_DIR}/submission_{i:02d}_{r.name}.csv"
        sub = save_submission(fname, test_ids, r.test_proba, r.best_threshold)
        submission_meta.append(
            {
                "file": fname,
                "name": r.name,
                "oof_bal_acc": r.oof_bal_acc,
                "threshold": r.best_threshold,
                "test_pos_rate": float(sub["Прогрессия"].mean()),
                "notes": r.notes,
            }
        )

    # 2) explicit threshold probing on best candidate
    train_pos_rate = float(np.mean(y))
    threshold_candidates = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, best.best_threshold]
    for delta in [0.0, 0.05, -0.05, 0.10, -0.10, 0.15, -0.15]:
        target_rate = float(np.clip(train_pos_rate + delta, 0.05, 0.95))
        thr = _positive_rate_threshold(target_rate, best.test_proba)
        threshold_candidates.append(thr)
    threshold_candidates = sorted(list(set([round(float(t), 6) for t in threshold_candidates])))

    for thr in threshold_candidates:
        name = f"{SUBMISSIONS_DIR}/submission_threshold_{thr:.3f}.csv"
        sub = save_submission(name, test_ids, best.test_proba, thr)
        submission_meta.append(
            {
                "file": name,
                "name": f"best_prob_threshold_{thr:.3f}",
                "oof_bal_acc": float(balanced_accuracy_score(y, (best.oof_proba >= thr).astype(int))),
                "threshold": float(thr),
                "test_pos_rate": float(sub["Прогрессия"].mean()),
                "notes": "threshold_probe",
            }
        )

    # 3) Named variants requested for LB probing
    # conservative / aggressive / train pos rate
    conservative_thr = min(0.75, best.best_threshold + 0.08)
    aggressive_thr = max(0.25, best.best_threshold - 0.10)
    train_pos_thr = _positive_rate_threshold(train_pos_rate, best.test_proba)

    named_variants = [
        (f"{SUBMISSIONS_DIR}/submission_01_best_oof.csv", best.test_proba, best.best_threshold, "best_oof"),
        (f"{SUBMISSIONS_DIR}/submission_02_conservative.csv", best.test_proba, conservative_thr, "conservative_threshold"),
        (f"{SUBMISSIONS_DIR}/submission_03_aggressive.csv", best.test_proba, aggressive_thr, "aggressive_threshold"),
        (f"{SUBMISSIONS_DIR}/submission_04_threshold_train_posrate.csv", best.test_proba, train_pos_thr, "train_posrate_match"),
    ]

    # If specific model names exist, add them explicitly
    by_name = {r.name: r for r in ranked_all}
    for probe_name, file_name in [
        ("ensemble_weighted", f"{SUBMISSIONS_DIR}/submission_05_weighted_ensemble.csv"),
        ("ensemble_rank_average", f"{SUBMISSIONS_DIR}/submission_06_rank_average.csv"),
        ("ensemble_stacking_lr", f"{SUBMISSIONS_DIR}/submission_07_stacking.csv"),
    ]:
        if probe_name in by_name:
            rr = by_name[probe_name]
            named_variants.append((file_name, rr.test_proba, rr.best_threshold, probe_name))

    # best catboost-like variants if exist
    cat_candidates = [r for r in ranked_all if "catboost" in r.name]
    if cat_candidates:
        named_variants.append(
            (f"{SUBMISSIONS_DIR}/submission_08_catboost_best.csv", cat_candidates[0].test_proba, cat_candidates[0].best_threshold, cat_candidates[0].name)
        )
        compact_cat = [r for r in cat_candidates if "_compact" in r.name]
        if compact_cat:
            named_variants.append(
                (f"{SUBMISSIONS_DIR}/submission_09_catboost_compact.csv", compact_cat[0].test_proba, compact_cat[0].best_threshold, compact_cat[0].name)
            )
        noshift_cat = [r for r in cat_candidates if "_no_shift" in r.name]
        if noshift_cat:
            named_variants.append(
                (f"{SUBMISSIONS_DIR}/submission_10_catboost_noshift.csv", noshift_cat[0].test_proba, noshift_cat[0].best_threshold, noshift_cat[0].name)
            )

    for path, probs, thr, nm in named_variants:
        sub = save_submission(path, test_ids, probs, thr)
        submission_meta.append(
            {
                "file": path,
                "name": nm,
                "oof_bal_acc": float(balanced_accuracy_score(y, (best.oof_proba >= thr).astype(int))) if len(probs) == len(best.test_proba) else np.nan,
                "threshold": float(thr),
                "test_pos_rate": float(sub["Прогрессия"].mean()),
                "notes": "named_variant",
            }
        )

    sub_meta_df = pd.DataFrame(submission_meta).drop_duplicates(subset=["file"]).sort_values("oof_bal_acc", ascending=False)
    sub_meta_df.to_csv(f"{SUBMISSIONS_DIR}/submission_catalog.csv", index=False, encoding="utf-8")

    # README for submissions
    with open(f"{SUBMISSIONS_DIR}/README_SUBMISSIONS.txt", "w", encoding="utf-8") as f:
        f.write("Recommended upload order\n")
        f.write("========================\n\n")
        top_try = sub_meta_df.head(15)
        for i, row in enumerate(top_try.itertuples(index=False), 1):
            f.write(
                f"{i}. {row.file}\n"
                f"   name: {row.name}\n"
                f"   oof_bal_acc: {row.oof_bal_acc:.6f}\n"
                f"   threshold: {row.threshold:.4f}\n"
                f"   expected_test_pos_rate: {row.test_pos_rate:.4f}\n"
                f"   notes: {row.notes}\n\n"
            )
        f.write("First try: highest OOF with stable CV strategy and reasonable positive rate.\n")
        f.write("Also test threshold variants because public LB can have different class balance.\n")

    # Additional required file copies with exact names
    # Keep a few legacy-style aliases
    # submission_best_single_model analog
    best_single = sorted(all_results, key=lambda r: r.oof_bal_acc, reverse=True)[0]
    save_submission(f"{SUBMISSIONS_DIR}/submission_best_single_model.csv", test_ids, best_single.test_proba, best_single.best_threshold)

    # Save summary json
    summary = {
        "best_model": best.name,
        "best_oof_bal_acc": float(best.oof_bal_acc),
        "best_threshold": float(best.best_threshold),
        "best_test_positive_rate": float(best.test_pos_rate),
        "adversarial_auc": float(adv_auc),
        "n_models": len(all_results),
        "n_ensembles": len(ensemble_results),
        "optional_model_errors": optional_errors,
        "catboost_errors": catboost_errors,
    }
    with open(f"{REPORTS_DIR}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # final console output
    print("\n=== FINAL SUMMARY ===")
    print(f"Best OOF Balanced Accuracy: {best.oof_bal_acc:.6f}")
    print(f"Best threshold: {best.best_threshold:.4f}")
    print(f"Test positive rate: {best.test_pos_rate:.4f}")
    print("\nTop 10 recommended submissions:")
    for row in sub_meta_df.head(10).itertuples(index=False):
        print(f"- {row.file} | oof={row.oof_bal_acc:.6f} | thr={row.threshold:.4f} | pos_rate={row.test_pos_rate:.4f}")

    best_vs_prev = best.oof_bal_acc - 0.7745
    if best_vs_prev < 0.01:
        print("\nWarning: CV gain over ~0.7745 is small; public LB may diverge, use threshold and model diversity probes.")
    else:
        print("\nWarning: if public LB diverges from CV, prioritize robust thresholds and less shift-sensitive submissions.")


if __name__ == "__main__":
    main()
