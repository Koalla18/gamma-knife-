import os
import re
import itertools
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")

SEEDS_STAGE3 = [42, 123, 777, 1001, 2025, 3407, 9090, 2718, 31415]
SEEDS_LIGHT = [42, 777, 2025, 3407, 1001]
LEAKAGE_COLUMNS = ["Прогрессия", "Интракраниальная прогрессия", "Локальный рецидив", "Дистантные метастазы"]
ID_CANDIDATES = ["ID", "Id", "id"]
MISSING_TOKENS = {"", "-", "—", "--", "nan", "none", "null", "na", "n/a", "#ref!"}
YES_TOKENS = {"есть", "да", "1", "true", "yes", "y", "+"}
NO_TOKENS = {"нет", "0", "false", "no", "n", "не удален", "не удалён", "отсутствует"}
CURRENT_BEST_PUBLIC_SCORE = 0.84090


@dataclass
class Candidate:
    name: str
    method: str
    oof_bal_acc: float
    threshold: float
    test_pos_rate: float
    changed_vs_anchor: int
    diff_pct_vs_anchor: float
    note: str
    file_path: str


def ensure_dirs() -> None:
    os.makedirs("stage3_reports", exist_ok=True)
    os.makedirs("stage3_submissions", exist_ok=True)


def read_csv_robust(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "cp1251", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read {path}: {last_err}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in out.columns]
    return out


def normalize_text(v):
    if pd.isna(v):
        return np.nan
    s = str(v).replace("\xa0", " ").strip().lower().replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    if s in MISSING_TOKENS:
        return np.nan
    return s


def parse_numeric(v):
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
    s = normalize_text(s)
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
        y = train_df["Прогрессия"]
        if is_numeric_dtype(y):
            return pd.to_numeric(y, errors="coerce"), "Прогрессия"
        z = y.map(normalize_text)
        return z.map(lambda x: 1.0 if x in YES_TOKENS else (0.0 if x in NO_TOKENS else np.nan)), "Прогрессия"

    if "Интракраниальная прогрессия" in train_df.columns:
        z = train_df["Интракраниальная прогрессия"].map(normalize_text)
        y = z.map(lambda x: np.nan if pd.isna(x) else (0.0 if x == "нет" else 1.0))
        return y.astype(float), "Интракраниальная прогрессия"

    raise ValueError("Target not found.")


def drop_leakage_and_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drop_cols = [c for c in LEAKAGE_COLUMNS if c in out.columns]
    for c in ID_CANDIDATES:
        if c in out.columns:
            drop_cols.append(c)
    return out.drop(columns=drop_cols, errors="ignore")


def _parse_date_series(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")
    unresolved = d.isna() & s.notna()
    if unresolved.any():
        d.loc[unresolved] = pd.to_datetime(s[unresolved], dayfirst=True, errors="coerce")
    return d


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = normalize_column_names(train_df.copy())
    te = normalize_column_names(test_df.copy())
    for df in [tr, te]:
        for c in df.columns:
            if is_string_dtype(df[c]) or df[c].dtype == object:
                df[c] = df[c].map(normalize_text)

    cols = tr.columns.tolist()
    date_cols = [c for c in cols if "дата" in norm_key(c)]
    for c in date_cols:
        tr_dt = _parse_date_series(tr[c])
        te_dt = _parse_date_series(te[c]) if c in te.columns else pd.Series(pd.NaT, index=te.index)
        tr[f"{c}__is_valid_date"] = tr_dt.notna().astype(float)
        te[f"{c}__is_valid_date"] = te_dt.notna().astype(float)
        tr[f"{c}__year"] = tr_dt.dt.year.astype(float)
        te[f"{c}__year"] = te_dt.dt.year.astype(float)
        tr[f"{c}__month"] = tr_dt.dt.month.astype(float)
        te[f"{c}__month"] = te_dt.dt.month.astype(float)
        tr[f"{c}__dt"] = tr_dt
        te[f"{c}__dt"] = te_dt

    c_birth = find_col(cols, ["Дата рождения"])
    c_diag = find_col(cols, ["Дата постановки онкологического диагноза / начала первичного лечения"])
    c_mgm = find_col(cols, ["Дата развития МГМ"])
    c_rx = find_col(cols, ["Дата 1-ой РХ", "Дата 1 ой РХ", "Дата 1-ои РХ"])
    c_rem = find_col(cols, ["Дата удаления первичного очага"])
    c_ovgm = find_col(cols, ["Дата проведения ОВГМ"])
    c_op = find_col(cols, ["Дата операции на ГМ"])

    def dt_name(c):
        return f"{c}__dt" if c is not None and f"{c}__dt" in tr.columns else None

    def add_interval(name: str, end_dt: Optional[str], start_dt: Optional[str]):
        if end_dt is None or start_dt is None:
            tr[name] = np.nan
            te[name] = np.nan
        else:
            tr[name] = (tr[end_dt] - tr[start_dt]).dt.days.astype(float)
            te[name] = (te[end_dt] - te[start_dt]).dt.days.astype(float)

        q1 = tr[name].quantile(0.01) if tr[name].notna().sum() > 10 else -36500
        q99 = tr[name].quantile(0.99) if tr[name].notna().sum() > 10 else 36500
        tr[f"{name}_clipped"] = tr[name].clip(q1, q99)
        te[f"{name}_clipped"] = te[name].clip(q1, q99)
        tr[f"{name}_log1p_pos"] = np.log1p(tr[name].clip(lower=0))
        te[f"{name}_log1p_pos"] = np.log1p(te[name].clip(lower=0))
        tr[f"{name}_negative_flag"] = (tr[name] < 0).astype(float)
        te[f"{name}_negative_flag"] = (te[name] < 0).astype(float)
        tr[f"{name}_missing_flag"] = tr[name].isna().astype(float)
        te[f"{name}_missing_flag"] = te[name].isna().astype(float)

    tr["primary_removed_flag"] = tr[dt_name(c_rem)].notna().astype(float) if dt_name(c_rem) else 0.0
    te["primary_removed_flag"] = te[dt_name(c_rem)].notna().astype(float) if dt_name(c_rem) else 0.0
    tr["ovgm_flag"] = tr[dt_name(c_ovgm)].notna().astype(float) if dt_name(c_ovgm) else 0.0
    te["ovgm_flag"] = te[dt_name(c_ovgm)].notna().astype(float) if dt_name(c_ovgm) else 0.0
    tr["brain_operation_flag"] = tr[dt_name(c_op)].notna().astype(float) if dt_name(c_op) else 0.0
    te["brain_operation_flag"] = te[dt_name(c_op)].notna().astype(float) if dt_name(c_op) else 0.0

    add_interval("age_at_rx_days", dt_name(c_rx), dt_name(c_birth))
    tr["age_at_rx_years"] = tr["age_at_rx_days"] / 365.25
    te["age_at_rx_years"] = te["age_at_rx_days"] / 365.25

    add_interval("age_at_diagnosis_days", dt_name(c_diag), dt_name(c_birth))
    tr["age_at_diagnosis_years"] = tr["age_at_diagnosis_days"] / 365.25
    te["age_at_diagnosis_years"] = te["age_at_diagnosis_days"] / 365.25

    add_interval("age_at_mgm_days", dt_name(c_mgm), dt_name(c_birth))
    tr["age_at_mgm_years"] = tr["age_at_mgm_days"] / 365.25
    te["age_at_mgm_years"] = te["age_at_mgm_days"] / 365.25

    add_interval("diagnosis_to_mgm_days", dt_name(c_mgm), dt_name(c_diag))
    add_interval("mgm_to_rx_days", dt_name(c_rx), dt_name(c_mgm))
    add_interval("diagnosis_to_rx_days", dt_name(c_rx), dt_name(c_diag))
    add_interval("primary_removal_to_rx_days", dt_name(c_rx), dt_name(c_rem))
    add_interval("ovgm_to_rx_days", dt_name(c_rx), dt_name(c_ovgm))
    add_interval("brain_operation_to_rx_days", dt_name(c_rx), dt_name(c_op))

    c_lesions = find_col(cols, ["Число очагов в ГМ"])
    c_total = find_col(cols, ["Суммарный объём очагов", "Суммарный объем очагов"])
    c_max = find_col(cols, ["Объём максимального очага", "Объем максимального очага"])
    c_karn = find_col(cols, ["Индекс Карновского"])
    c_treat = find_col(cols, ["Лекарственное лечение"])
    c_diagc = find_col(cols, ["Онкологический диагноз"])
    c_mut = find_col(cols, ["Активирующие мутации"])
    c_ext = find_col(cols, ["Экстракраниальные метастазы"])
    c_sex = find_col(cols, ["Пол"])

    num_sources = [c_lesions, c_total, c_max, c_karn]
    for c in num_sources:
        if c is not None and c in tr.columns:
            tr[c] = tr[c].map(parse_numeric)
            te[c] = te[c].map(parse_numeric)

    tr["lesion_count"] = tr[c_lesions] if c_lesions else np.nan
    te["lesion_count"] = te[c_lesions] if c_lesions else np.nan
    tr["total_volume"] = tr[c_total] if c_total else np.nan
    te["total_volume"] = te[c_total] if c_total else np.nan
    tr["max_volume"] = tr[c_max] if c_max else np.nan
    te["max_volume"] = te[c_max] if c_max else np.nan
    tr["karnovsky"] = tr[c_karn] if c_karn else np.nan
    te["karnovsky"] = te[c_karn] if c_karn else np.nan

    for df in [tr, te]:
        df["avg_volume_per_lesion"] = np.where(df["lesion_count"] > 0, df["total_volume"] / df["lesion_count"], np.nan)
        df["max_to_total_volume_ratio"] = np.where(df["total_volume"] > 0, df["max_volume"] / df["total_volume"], np.nan)
        df["residual_volume"] = df["total_volume"] - df["max_volume"]
        df["karnovsky_deficit"] = 100.0 - df["karnovsky"]
        df["log1p_lesion_count"] = np.log1p(df["lesion_count"].clip(lower=0))
        df["log1p_total_volume"] = np.log1p(df["total_volume"].clip(lower=0))
        df["log1p_max_volume"] = np.log1p(df["max_volume"].clip(lower=0))
        df["lesion_x_total"] = df["lesion_count"] * df["total_volume"]
        df["lesion_x_max"] = df["lesion_count"] * df["max_volume"]
        df["lesion_x_kdef"] = df["lesion_count"] * df["karnovsky_deficit"]
        df["total_x_kdef"] = df["total_volume"] * df["karnovsky_deficit"]
        df["ovgm_x_lesions"] = df["ovgm_flag"] * df["lesion_count"]
        df["ovgm_x_total"] = df["ovgm_flag"] * df["total_volume"]
        df["operation_x_max"] = df["brain_operation_flag"] * df["max_volume"]
        df["has_activating_mutations"] = df[c_mut].map(lambda x: 1.0 if x in YES_TOKENS else (0.0 if x in NO_TOKENS else np.nan)) if c_mut else np.nan
        df["has_extracranial_metastases"] = df[c_ext].map(lambda x: 1.0 if x in YES_TOKENS else (0.0 if x in NO_TOKENS else np.nan)) if c_ext else np.nan
        df["sex_bin"] = df[c_sex].map(lambda x: 0.0 if x in {"м", "муж", "мужской"} else (1.0 if x in {"ж", "жен", "женский"} else np.nan)) if c_sex else np.nan
        df["diagnosis_cat"] = df[c_diagc].fillna("missing").astype(str) if c_diagc else "missing"
        df["treatment_cat"] = df[c_treat].fillna("missing").astype(str) if c_treat else "missing"
        df["mutation_cat"] = df[c_mut].fillna("missing").astype(str) if c_mut else "missing"
        df["extracranial_cat"] = df[c_ext].fillna("missing").astype(str) if c_ext else "missing"
        df["diagnosis_x_treatment"] = df["diagnosis_cat"] + "__" + df["treatment_cat"]
        df["mutations_x_treatment"] = df["mutation_cat"] + "__" + df["treatment_cat"]
        df["extracranial_x_diagnosis"] = df["extracranial_cat"] + "__" + df["diagnosis_cat"]
        df["extracranial_x_treatment"] = df["extracranial_cat"] + "__" + df["treatment_cat"]

    # drop raw datetime and raw date strings
    to_drop = [c for c in tr.columns if c.endswith("__dt")]
    tr = tr.drop(columns=to_drop, errors="ignore")
    te = te.drop(columns=to_drop, errors="ignore")
    tr = tr.drop(columns=date_cols, errors="ignore")
    te = te.drop(columns=[c for c in date_cols if c in te.columns], errors="ignore")

    tr = drop_leakage_and_id(tr)
    te = drop_leakage_and_id(te)

    common = sorted(list(set(tr.columns) | set(te.columns)))
    tr = tr.reindex(columns=common)
    te = te.reindex(columns=common)
    return tr, te


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], verbose_feature_names_out=False)
    return pre, num_cols, cat_cols


def find_best_threshold(y_true: np.ndarray, probas: np.ndarray) -> Tuple[float, float]:
    best_thr = 0.5
    best_score = -1.0
    for thr in np.arange(0.01, 0.991, 0.001):
        pred = (probas >= thr).astype(int)
        sc = balanced_accuracy_score(y_true, pred)
        if sc > best_score:
            best_score = sc
            best_thr = float(thr)
    return best_thr, float(best_score)


def predict_proba_generic(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return np.asarray(model.predict(X), dtype=float)


def train_oof_model(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    sample_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(X), dtype=float)
    cnt = np.zeros(len(X), dtype=float)
    test_accum = np.zeros(len(X_test), dtype=float)
    for tr_idx, va_idx in splits:
        Xtr, Xva = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        ytr = y[tr_idx]
        pre, _, _ = build_preprocessor(Xtr)
        Xtr_m = pre.fit_transform(Xtr)
        Xva_m = pre.transform(Xva)
        Xte_m = pre.transform(X_test)
        if hasattr(Xtr_m, "toarray"):
            Xtr_m = Xtr_m.toarray()
            Xva_m = Xva_m.toarray()
            Xte_m = Xte_m.toarray()

        mdl = model
        if hasattr(mdl, "set_params") and hasattr(mdl, "random_state"):
            try:
                mdl = mdl.set_params(random_state=42)
            except Exception:
                pass

        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[tr_idx]
        mdl.fit(Xtr_m, ytr, **fit_kwargs)

        pva = predict_proba_generic(mdl, Xva_m)
        pte = predict_proba_generic(mdl, Xte_m)
        oof[va_idx] += pva
        cnt[va_idx] += 1.0
        test_accum += pte
    oof = oof / np.maximum(cnt, 1.0)
    test_probs = test_accum / len(splits)
    return oof, test_probs


def train_catboost_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    params: Dict,
    sample_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    from catboost import CatBoostClassifier

    oof = np.zeros(len(X), dtype=float)
    cnt = np.zeros(len(X), dtype=float)
    test_accum = np.zeros(len(X_test), dtype=float)
    cat_cols = [c for c in X.columns if not is_numeric_dtype(X[c])]

    base_params = dict(
        iterations=2500,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_strength=0.5,
        bagging_temperature=0.2,
        border_count=128,
        loss_function="Logloss",
        eval_metric="Logloss",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
    )
    base_params.update(params or {})

    X_use = X.copy()
    Xte_use = X_test.copy()
    for c in cat_cols:
        X_use[c] = X_use[c].fillna("missing").astype(str)
        Xte_use[c] = Xte_use[c].fillna("missing").astype(str)
    for c in X_use.columns:
        if c not in cat_cols:
            X_use[c] = pd.to_numeric(X_use[c], errors="coerce")
            Xte_use[c] = pd.to_numeric(Xte_use[c], errors="coerce")

    for tr_idx, va_idx in splits:
        Xtr = X_use.iloc[tr_idx].copy()
        Xva = X_use.iloc[va_idx].copy()
        ytr = y[tr_idx]
        yva = y[va_idx]
        model = CatBoostClassifier(**base_params)
        fit_kwargs = {
            "X": Xtr,
            "y": ytr,
            "cat_features": cat_cols,
            "eval_set": (Xva, yva),
            "use_best_model": True,
            "early_stopping_rounds": 120,
        }
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[tr_idx]
        model.fit(**fit_kwargs)
        pva = model.predict_proba(Xva)[:, 1]
        pte = model.predict_proba(Xte_use)[:, 1]
        oof[va_idx] += pva
        cnt[va_idx] += 1.0
        test_accum += pte

    oof = oof / np.maximum(cnt, 1.0)
    test_probs = test_accum / len(splits)
    return oof, test_probs


def collect_submission_files() -> List[str]:
    paths = []
    candidates = [".", "submissions", "stage3_submissions"]
    for root in candidates:
        if not os.path.isdir(root):
            continue
        for fn in os.listdir(root):
            if fn.lower().endswith(".csv") and "submission" in fn.lower():
                paths.append(os.path.join(root, fn))
    # Dedup and stable
    return sorted(list(set(paths)))


def load_submission(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        if "ID" in df.columns and "Прогрессия" in df.columns and len(df.columns) == 2:
            return df
    except Exception:
        pass
    return None


def infer_anchor_submission(existing_submissions: List[str]) -> Tuple[str, Optional[pd.DataFrame], str]:
    # Priority 1: known public score file
    lb_files = [
        "stage3_reports/public_lb_scores.csv",
        "public_lb_scores.csv",
        "reports/public_lb_scores.csv",
    ]
    for lb in lb_files:
        if os.path.exists(lb):
            try:
                df = pd.read_csv(lb)
                if {"submission_file", "public_score"}.issubset(df.columns):
                    df = df.dropna(subset=["public_score"]).sort_values("public_score", ascending=False)
                    if len(df) > 0:
                        best_file = str(df.iloc[0]["submission_file"])
                        if os.path.exists(best_file):
                            sub = load_submission(best_file)
                            if sub is not None:
                                return best_file, sub, f"public_lb_file:{lb}"
            except Exception:
                pass

    # Priority 2: current submission.csv
    if os.path.exists("submission.csv"):
        sub = load_submission("submission.csv")
        if sub is not None:
            return "submission.csv", sub, "fallback_current_submission"

    # Priority 3: first valid existing
    for p in existing_submissions:
        sub = load_submission(p)
        if sub is not None:
            return p, sub, "fallback_first_valid"

    return "", None, "no_anchor_found"


def create_public_lb_template(existing_submissions: List[str]) -> None:
    tpl_path = "stage3_reports/public_lb_scores_template.csv"
    if os.path.exists(tpl_path):
        return
    df = pd.DataFrame(
        {
            "submission_file": existing_submissions,
            "public_score": [np.nan] * len(existing_submissions),
            "notes": ["" for _ in existing_submissions],
        }
    )
    df.to_csv(tpl_path, index=False, encoding="utf-8")


def load_probability_sources() -> Dict[str, Dict]:
    """
    Returns dict:
      source_name: {"y": np.ndarray, "oof": DataFrame(cols=model names), "test": DataFrame(cols=model names)}
    """
    sources = {}

    # Pair 1: stage2/advanced reports
    p1_oof = "reports/oof_predictions.csv"
    p1_test = "reports/test_probabilities.csv"
    if os.path.exists(p1_oof) and os.path.exists(p1_test):
        try:
            oof_df = pd.read_csv(p1_oof)
            test_df = pd.read_csv(p1_test)
            y_col = "y_true" if "y_true" in oof_df.columns else ("target" if "target" in oof_df.columns else None)
            if y_col is not None:
                y = oof_df[y_col].values.astype(int)
                oof_cols = {c.replace("oof_", ""): c for c in oof_df.columns if c.startswith("oof_")}
                test_cols = {c.replace("test_", ""): c for c in test_df.columns if c.startswith("test_")}
                common = sorted(list(set(oof_cols.keys()) & set(test_cols.keys())))
                if common:
                    sources["reports_stage"] = {
                        "y": y,
                        "oof": oof_df[[oof_cols[k] for k in common]].rename(columns={oof_cols[k]: k for k in common}),
                        "test": test_df[[test_cols[k] for k in common]].rename(columns={test_cols[k]: k for k in common}),
                    }
        except Exception:
            pass

    # Pair 2: older root outputs
    p2_oof = "oof_predictions.csv"
    p2_test = "test_probabilities_by_model.csv"
    if os.path.exists(p2_oof) and os.path.exists(p2_test):
        try:
            oof_df = pd.read_csv(p2_oof)
            test_df = pd.read_csv(p2_test)
            y_col = "y_true" if "y_true" in oof_df.columns else ("target" if "target" in oof_df.columns else None)
            if y_col is not None:
                y = oof_df[y_col].values.astype(int)
                oof_cols = {c.replace("oof_", ""): c for c in oof_df.columns if c.startswith("oof_")}
                # old file may have prefix test_
                test_cols = {c.replace("test_", ""): c for c in test_df.columns if c.startswith("test_")}
                common = sorted(list(set(oof_cols.keys()) & set(test_cols.keys())))
                if common:
                    sources["root_legacy"] = {
                        "y": y,
                        "oof": oof_df[[oof_cols[k] for k in common]].rename(columns={oof_cols[k]: k for k in common}),
                        "test": test_df[[test_cols[k] for k in common]].rename(columns={test_cols[k]: k for k in common}),
                    }
        except Exception:
            pass

    return sources


def select_best_probability_base(prob_sources: Dict[str, Dict]) -> Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Returns:
      source_name, model_name, y, oof_probs, test_probs, best_thr, best_oof_score
    """
    best = None
    for src_name, src in prob_sources.items():
        y = src["y"]
        for model_name in src["oof"].columns:
            oof = src["oof"][model_name].values
            test = src["test"][model_name].values
            thr, sc = find_best_threshold(y, oof)
            if best is None or sc > best[-1]:
                best = (src_name, model_name, y, oof, test, thr, sc)
    if best is None:
        raise RuntimeError("No probability sources with OOF/test pairs were found.")
    return best


def jaccard_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(int)
    b = b.astype(int)
    inter = np.sum((a == 1) & (b == 1))
    union = np.sum((a == 1) | (b == 1))
    if union == 0:
        return 1.0
    return float(inter / union)


def compute_submission_similarity(anchor_df: pd.DataFrame, submissions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    anc = anchor_df.sort_values("ID")["Прогрессия"].values.astype(int)
    for name, df in submissions.items():
        pred = df.sort_values("ID")["Прогрессия"].values.astype(int)
        changed = int(np.sum(pred != anc))
        pct = changed / len(anc)
        j = jaccard_binary(pred, anc)
        rows.append(
            {
                "comparison_type": "anchor_vs_submission",
                "submission_a": "anchor",
                "submission_b": name,
                "changed_labels": changed,
                "changed_pct": pct,
                "positive_rate_a": float(np.mean(anc)),
                "positive_rate_b": float(np.mean(pred)),
                "positive_rate_diff": float(np.mean(pred) - np.mean(anc)),
                "jaccard_positive": j,
            }
        )

    # pairwise among submissions (limited to first 30 for size)
    keys = sorted(list(submissions.keys()))[:30]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = submissions[keys[i]].sort_values("ID")["Прогрессия"].values.astype(int)
            b = submissions[keys[j]].sort_values("ID")["Прогрессия"].values.astype(int)
            changed = int(np.sum(a != b))
            pct = changed / len(a)
            rows.append(
                {
                    "comparison_type": "pairwise",
                    "submission_a": keys[i],
                    "submission_b": keys[j],
                    "changed_labels": changed,
                    "changed_pct": pct,
                    "positive_rate_a": float(np.mean(a)),
                    "positive_rate_b": float(np.mean(b)),
                    "positive_rate_diff": float(np.mean(b) - np.mean(a)),
                    "jaccard_positive": jaccard_binary(a, b),
                }
            )

    return pd.DataFrame(rows)


def save_submission(path: str, ids: np.ndarray, labels: np.ndarray) -> None:
    df = pd.DataFrame({"ID": ids.astype(int), "Прогрессия": labels.astype(int)})
    df.to_csv(path, index=False, encoding="utf-8")


def labels_from_threshold(probs: np.ndarray, thr: float) -> np.ndarray:
    return (probs >= thr).astype(int)


def threshold_from_target_posrate(probs: np.ndarray, target_rate: float) -> float:
    p = np.sort(probs)
    idx = int(np.clip((1.0 - target_rate) * len(probs), 0, len(probs) - 1))
    return float(p[idx])


def risk_category(diff_pct: float) -> str:
    if diff_pct <= 0.05:
        return "close_1_5pct"
    if diff_pct <= 0.12:
        return "medium_5_12pct"
    if diff_pct <= 0.25:
        return "aggressive_12_25pct"
    return "extreme_25pct_plus"


def geometric_mean_probs(mat: np.ndarray) -> np.ndarray:
    mat = np.clip(mat, 1e-6, 1 - 1e-6)
    return np.exp(np.mean(np.log(mat), axis=1))


def logit_average_probs(mat: np.ndarray) -> np.ndarray:
    mat = np.clip(mat, 1e-6, 1 - 1e-6)
    logits = np.log(mat / (1 - mat))
    mean_logit = np.mean(logits, axis=1)
    return 1.0 / (1.0 + np.exp(-mean_logit))


def generate_weight_vectors(k: int, step: float, max_count: int = 300) -> List[np.ndarray]:
    vectors = []
    if k == 2:
        for a in np.arange(0, 1 + 1e-9, step):
            vectors.append(np.array([a, 1 - a], dtype=float))
    elif k == 3:
        # integer partitions on grid
        n = int(round(1 / step))
        for a in range(n + 1):
            for b in range(n + 1 - a):
                c = n - a - b
                vectors.append(np.array([a, b, c], dtype=float) / n)
    else:
        # deterministic plus random-like grid
        vectors.append(np.ones(k, dtype=float) / k)
        for i in range(k):
            w = np.ones(k, dtype=float) * ((1 - 0.6) / max(k - 1, 1))
            w[i] = 0.6
            vectors.append(w)
        rng = np.random.default_rng(42 + k)
        for _ in range(min(max_count, 200)):
            w = rng.random(k)
            w = w / w.sum()
            # quantize
            w = np.round(w / step) * step
            if w.sum() == 0:
                continue
            w = w / w.sum()
            vectors.append(w)
    # unique
    uniq = []
    seen = set()
    for w in vectors:
        key = tuple(np.round(w, 4).tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(w)
        if len(uniq) >= max_count:
            break
    return uniq


def main():
    ensure_dirs()

    # -------------------
    # Phase 0: project audit
    # -------------------
    train = normalize_column_names(read_csv_robust("train.csv"))
    test = normalize_column_names(read_csv_robust("test.csv"))
    y_raw, target_source = prepare_target(train)
    labeled_mask = y_raw.notna()
    train_labeled = train.loc[labeled_mask].reset_index(drop=True).copy()
    y = y_raw.loc[labeled_mask].astype(int).values
    dropped_unlabeled = int((~labeled_mask).sum())

    id_col = next((c for c in ID_CANDIDATES if c in test.columns), None)
    if id_col is None:
        test_ids = np.arange(1, len(test) + 1)
    else:
        test_ids = test[id_col].astype(int).values

    existing_sub_files = collect_submission_files()
    create_public_lb_template(existing_sub_files)
    anchor_path, anchor_df, anchor_reason = infer_anchor_submission(existing_sub_files)
    if anchor_df is None:
        raise RuntimeError("No valid anchor submission found.")

    # normalize anchor IDs order
    anchor_df = anchor_df.sort_values("ID").reset_index(drop=True)
    anchor_labels = anchor_df["Прогрессия"].astype(int).values

    # load all valid submissions for similarity
    sub_dict = {}
    for p in existing_sub_files:
        df = load_submission(p)
        if df is None:
            continue
        df = df.sort_values("ID").reset_index(drop=True)
        if len(df) != len(anchor_df):
            continue
        if not np.array_equal(df["ID"].values, anchor_df["ID"].values):
            continue
        sub_dict[p] = df

    sim_df = compute_submission_similarity(anchor_df, sub_dict)
    sim_df.to_csv("stage3_reports/submission_similarity.csv", index=False, encoding="utf-8")

    # -------------------
    # Phase 1: best probability source
    # -------------------
    prob_sources = load_probability_sources()
    if len(prob_sources) == 0:
        raise RuntimeError("No probability sources found. Need reports/oof_predictions.csv + test probabilities.")

    src_name, base_model_name, y_oof, base_oof_probs, base_test_probs, base_thr, base_oof_score = select_best_probability_base(prob_sources)
    if len(y_oof) != len(y):
        # fallback to training labels length if mismatch
        y = y_oof

    # For traceability
    source_info = {
        "selected_source": src_name,
        "selected_model": base_model_name,
        "selected_oof_score": float(base_oof_score),
        "selected_threshold": float(base_thr),
    }

    # -------------------
    # Phase 2: LB-aware threshold sweep
    # -------------------
    threshold_rows = []
    candidates: List[Candidate] = []
    seen_label_hashes = set()

    def maybe_add_candidate(name: str, method: str, probs: np.ndarray, thr: float, note: str):
        nonlocal candidates, seen_label_hashes
        labels = labels_from_threshold(probs, thr)
        key = tuple(labels.tolist())
        if key in seen_label_hashes:
            return
        seen_label_hashes.add(key)
        changed = int(np.sum(labels != anchor_labels))
        diff_pct = changed / len(labels)
        oof_sc = float(balanced_accuracy_score(y, (base_oof_probs >= thr).astype(int)))
        pos_rate = float(labels.mean())
        out_path = f"stage3_submissions/{name}.csv"
        save_submission(out_path, test_ids, labels)
        threshold_rows.append(
            {
                "candidate_name": name,
                "method": method,
                "threshold": float(thr),
                "oof_bal_acc_proxy": oof_sc,
                "test_positive_rate": pos_rate,
                "changed_vs_anchor": changed,
                "diff_pct_vs_anchor": diff_pct,
                "risk_category": risk_category(diff_pct),
                "note": note,
                "file_path": out_path,
            }
        )
        candidates.append(
            Candidate(
                name=name,
                method=method,
                oof_bal_acc=oof_sc,
                threshold=float(thr),
                test_pos_rate=pos_rate,
                changed_vs_anchor=changed,
                diff_pct_vs_anchor=diff_pct,
                note=note,
                file_path=out_path,
            )
        )

    # Full grid 0.01..0.99 step 0.002
    grid = np.arange(0.01, 0.991, 0.002)
    # local around best threshold
    local_offsets = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    local = [base_thr]
    for d in local_offsets:
        local.extend([base_thr - d, base_thr + d])
    local = [float(np.clip(t, 0.01, 0.99)) for t in local]
    grid = np.unique(np.concatenate([grid, np.array(local)]))

    for thr in grid:
        maybe_add_candidate(
            name=f"submission_thr_{thr:.3f}_pos_{(base_test_probs >= thr).mean():.3f}",
            method="threshold_sweep",
            probs=base_test_probs,
            thr=float(thr),
            note="global+local_threshold_grid",
        )

    # target positive-rate candidates
    train_pos_rate = float(np.mean(y))
    anchor_pos_rate = float(np.mean(anchor_labels))
    oof_pos_rate = float(np.mean(base_oof_probs >= base_thr))
    target_rates = [train_pos_rate, oof_pos_rate, anchor_pos_rate]
    for d in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        target_rates.extend([anchor_pos_rate - d, anchor_pos_rate + d])
    # minority recall aggressive / conservative
    target_rates.extend([min(0.95, anchor_pos_rate + 0.12), max(0.05, anchor_pos_rate - 0.12)])
    target_rates = sorted(list(set([float(np.clip(r, 0.05, 0.95)) for r in target_rates])))
    for r in target_rates:
        thr = threshold_from_target_posrate(base_test_probs, r)
        maybe_add_candidate(
            name=f"submission_posrate_target_{r:.3f}_thr_{thr:.3f}",
            method="target_posrate_sweep",
            probs=base_test_probs,
            thr=float(thr),
            note="target_positive_rate",
        )

    pd.DataFrame(threshold_rows).to_csv("stage3_reports/threshold_grid.csv", index=False, encoding="utf-8")

    # -------------------
    # Phase 3: blend search
    # -------------------
    src = prob_sources[src_name]
    model_scores = []
    for m in src["oof"].columns:
        thr, sc = find_best_threshold(y, src["oof"][m].values)
        model_scores.append((m, sc, thr))
    model_scores = sorted(model_scores, key=lambda x: x[1], reverse=True)
    top_models = [m for m, _, _ in model_scores[: min(8, len(model_scores))]]

    blend_rows = []
    blend_candidates = []
    # Precompute matrices
    oof_mat_all = src["oof"][top_models].values
    test_mat_all = src["test"][top_models].values

    # Methods with full top models
    for method_name, oof_probs, test_probs in [
        ("rank_average_all", pd.DataFrame(oof_mat_all).rank(axis=0, pct=True).mean(axis=1).values, pd.DataFrame(test_mat_all).rank(axis=0, pct=True).mean(axis=1).values),
        ("geometric_mean_all", geometric_mean_probs(np.clip(oof_mat_all, 1e-6, 1 - 1e-6)), geometric_mean_probs(np.clip(test_mat_all, 1e-6, 1 - 1e-6))),
        ("logit_average_all", logit_average_probs(oof_mat_all), logit_average_probs(test_mat_all)),
        ("median_all", np.median(oof_mat_all, axis=1), np.median(test_mat_all, axis=1)),
        ("trimmed_mean_all", np.mean(np.sort(oof_mat_all, axis=1)[:, 1:-1], axis=1) if oof_mat_all.shape[1] > 2 else np.mean(oof_mat_all, axis=1),
         np.mean(np.sort(test_mat_all, axis=1)[:, 1:-1], axis=1) if test_mat_all.shape[1] > 2 else np.mean(test_mat_all, axis=1)),
    ]:
        thr, sc = find_best_threshold(y, oof_probs)
        labels = (test_probs >= thr).astype(int)
        changed = int(np.sum(labels != anchor_labels))
        diff_pct = changed / len(labels)
        path = f"stage3_submissions/submission_blend_{method_name}.csv"
        save_submission(path, test_ids, labels)
        blend_rows.append(
            {
                "blend_name": method_name,
                "models": "|".join(top_models),
                "weights": "n/a",
                "method": method_name,
                "oof_bal_acc": sc,
                "best_threshold": thr,
                "test_pos_rate": float(labels.mean()),
                "changed_vs_anchor": changed,
                "diff_pct_vs_anchor": diff_pct,
                "file_path": path,
            }
        )
        blend_candidates.append((method_name, oof_probs, test_probs, thr, sc))

    # Weighted blends for combinations of 2..5 models
    comb_limit = {2: 30, 3: 30, 4: 20, 5: 10}
    for k in [2, 3, 4, 5]:
        combos = list(itertools.combinations(top_models, k))
        combos = combos[: comb_limit[k]]
        step = 0.05 if k <= 3 else 0.10
        w_vecs = generate_weight_vectors(k, step=step, max_count=120 if k <= 3 else 60)
        for combo in combos:
            idx = [top_models.index(m) for m in combo]
            oof_m = oof_mat_all[:, idx]
            test_m = test_mat_all[:, idx]
            # Simple average baseline
            oof_avg = np.mean(oof_m, axis=1)
            test_avg = np.mean(test_m, axis=1)
            thr, sc = find_best_threshold(y, oof_avg)
            labels = (test_avg >= thr).astype(int)
            path = f"stage3_submissions/submission_blend_avg_{'_'.join(combo)}.csv"
            save_submission(path, test_ids, labels)
            blend_rows.append(
                {
                    "blend_name": f"avg_{'_'.join(combo)}",
                    "models": "|".join(combo),
                    "weights": "equal",
                    "method": "avg",
                    "oof_bal_acc": sc,
                    "best_threshold": thr,
                    "test_pos_rate": float(labels.mean()),
                    "changed_vs_anchor": int(np.sum(labels != anchor_labels)),
                    "diff_pct_vs_anchor": float(np.mean(labels != anchor_labels)),
                    "file_path": path,
                }
            )
            blend_candidates.append((f"avg_{'_'.join(combo)}", oof_avg, test_avg, thr, sc))

            # Weighted grid
            for w in w_vecs:
                oof_bl = oof_m @ w
                test_bl = test_m @ w
                thr, sc = find_best_threshold(y, oof_bl)
                labels = (test_bl >= thr).astype(int)
                blend_rows.append(
                    {
                        "blend_name": f"wavg_{'_'.join(combo)}",
                        "models": "|".join(combo),
                        "weights": "|".join([f"{x:.2f}" for x in w]),
                        "method": "weighted_avg",
                        "oof_bal_acc": sc,
                        "best_threshold": thr,
                        "test_pos_rate": float(labels.mean()),
                        "changed_vs_anchor": int(np.sum(labels != anchor_labels)),
                        "diff_pct_vs_anchor": float(np.mean(labels != anchor_labels)),
                        "file_path": "",
                    }
                )
                # Save only promising
                if sc >= base_oof_score - 0.01:
                    name = f"submission_blend_wavg_k{k}_{abs(hash((combo, tuple(np.round(w, 3))))) % 10**8}"
                    path = f"stage3_submissions/{name}.csv"
                    save_submission(path, test_ids, labels)
                    blend_candidates.append((name, oof_bl, test_bl, thr, sc))

    blend_df = pd.DataFrame(blend_rows).sort_values("oof_bal_acc", ascending=False).reset_index(drop=True)
    blend_df.to_csv("stage3_reports/blend_grid.csv", index=False, encoding="utf-8")

    # Add top blend candidates to generic candidates list
    for name, oofp, testp, thr, sc in sorted(blend_candidates, key=lambda z: z[4], reverse=True)[:80]:
        labels = (testp >= thr).astype(int)
        path = f"stage3_submissions/submission_{name}.csv" if not name.endswith(".csv") else f"stage3_submissions/{name}"
        if not os.path.exists(path):
            save_submission(path, test_ids, labels)
        candidates.append(
            Candidate(
                name=name,
                method="blend_search",
                oof_bal_acc=float(sc),
                threshold=float(thr),
                test_pos_rate=float(labels.mean()),
                changed_vs_anchor=int(np.sum(labels != anchor_labels)),
                diff_pct_vs_anchor=float(np.mean(labels != anchor_labels)),
                note="blend",
                file_path=path,
            )
        )

    # -------------------
    # Phase 4: feature ablation with best model family (CatBoost + ExtraTrees)
    # -------------------
    X_train_full, X_test_full = build_features(train_labeled, test.copy())
    feature_names = list(X_train_full.columns)

    # adversarial shift importance
    X_adv = pd.concat([X_train_full.copy(), X_test_full.copy()], axis=0).reset_index(drop=True)
    y_adv = np.array([0] * len(X_train_full) + [1] * len(X_test_full))
    pre_adv, _, _ = build_preprocessor(X_adv)
    X_adv_m = pre_adv.fit_transform(X_adv)
    if hasattr(X_adv_m, "toarray"):
        X_adv_m = X_adv_m.toarray()
    adv_clf = LogisticRegression(max_iter=4000, random_state=42)
    adv_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    adv_oof = np.zeros(len(X_adv), dtype=float)
    for tr_idx, va_idx in adv_cv.split(X_adv_m, y_adv):
        m = LogisticRegression(max_iter=4000, random_state=42)
        m.fit(X_adv_m[tr_idx], y_adv[tr_idx])
        adv_oof[va_idx] = m.predict_proba(X_adv_m[va_idx])[:, 1]
    adv_auc = roc_auc_score(y_adv, adv_oof)

    adv_clf.fit(X_adv_m, y_adv)
    adv_feats = pre_adv.get_feature_names_out()
    coef = np.abs(adv_clf.coef_[0])
    shift_imp = pd.Series(coef, index=adv_feats).sort_values(ascending=False)
    shift_base = {}
    for feat, val in shift_imp.items():
        base = feat
        for c in feature_names:
            if feat.startswith(c + "_") or feat == c:
                base = c
                break
        shift_base[base] = shift_base.get(base, 0.0) + float(val)
    shift_base = pd.Series(shift_base).sort_values(ascending=False)
    high_shift_features = shift_base.head(8).index.tolist()

    # Baseline feature importance for top-N sets
    try:
        from catboost import CatBoostClassifier

        cat_cols = [c for c in X_train_full.columns if not is_numeric_dtype(X_train_full[c])]
        X_cb = X_train_full.copy()
        for c in cat_cols:
            X_cb[c] = X_cb[c].fillna("missing").astype(str)
        for c in X_cb.columns:
            if c not in cat_cols:
                X_cb[c] = pd.to_numeric(X_cb[c], errors="coerce")
        cb_fi_model = CatBoostClassifier(
            iterations=1200,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=3,
            auto_class_weights="Balanced",
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
        )
        cb_fi_model.fit(X_cb, y, cat_features=cat_cols)
        fi_tbl = cb_fi_model.get_feature_importance(prettified=True)
        fi_tbl = fi_tbl.rename(columns={"Feature Id": "feature", "Importances": "importance"})
        top_by_imp = fi_tbl["feature"].astype(str).tolist()
    except Exception:
        pre_fi, _, _ = build_preprocessor(X_train_full)
        Xm = pre_fi.fit_transform(X_train_full)
        if hasattr(Xm, "toarray"):
            Xm = Xm.toarray()
        et = ExtraTreesClassifier(
            n_estimators=1200, max_depth=9, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1
        )
        et.fit(Xm, y)
        f_names = pre_fi.get_feature_names_out()
        imp = pd.Series(et.feature_importances_, index=f_names).sort_values(ascending=False)
        top_by_imp = imp.index.tolist()

    def featset(name: str) -> List[str]:
        cols = list(X_train_full.columns)
        if name == "all":
            return cols
        if name == "compact_clinical":
            keep = [
                "age_at_rx_years",
                "diagnosis_to_mgm_days",
                "mgm_to_rx_days",
                "diagnosis_to_rx_days",
                "primary_removal_to_rx_days",
                "ovgm_to_rx_days",
                "brain_operation_to_rx_days",
                "primary_removed_flag",
                "ovgm_flag",
                "brain_operation_flag",
                "karnovsky",
                "karnovsky_deficit",
                "lesion_count",
                "total_volume",
                "max_volume",
                "has_activating_mutations",
                "has_extracranial_metastases",
                "diagnosis_cat",
                "treatment_cat",
                "mutation_cat",
                "extracranial_cat",
            ]
            return [c for c in keep if c in cols]
        if name == "no_date_year_month":
            return [c for c in cols if not c.endswith("__year") and not c.endswith("__month")]
        if name == "intervals_only_no_year_month":
            keep = [c for c in cols if "days" in c or "age_at_" in c or "flag" in c]
            return keep
        if name == "no_high_shift_features":
            return [c for c in cols if c not in high_shift_features]
        if name == "top10":
            return [c for c in top_by_imp[:10] if c in cols]
        if name == "top15":
            return [c for c in top_by_imp[:15] if c in cols]
        if name == "top20":
            return [c for c in top_by_imp[:20] if c in cols]
        if name == "top30":
            return [c for c in top_by_imp[:30] if c in cols]
        if name == "no_interactions":
            return [c for c in cols if "_x_" not in c and "diagnosis_x_" not in c and "mutations_x_" not in c and "extracranial_x_" not in c]
        if name == "strong_interactions_only":
            keep = ["lesion_x_total", "lesion_x_max", "lesion_x_kdef", "total_x_kdef", "ovgm_x_lesions", "ovgm_x_total", "operation_x_max"]
            return [c for c in keep if c in cols]
        if name == "catboost_native_only":
            keep = ["diagnosis_cat", "treatment_cat", "mutation_cat", "extracranial_cat", "sex_bin", "karnovsky", "lesion_count", "total_volume", "max_volume", "age_at_rx_years", "ovgm_flag", "brain_operation_flag"]
            return [c for c in keep if c in cols]
        if name == "one_hot_only":
            return cols
        if name == "only_target_encoded_cat_plus_numeric":
            # approximated via numeric + raw cat to be TE-ready
            return cols
        return cols

    feature_set_names = [
        "all",
        "compact_clinical",
        "no_date_year_month",
        "intervals_only_no_year_month",
        "no_high_shift_features",
        "top10",
        "top15",
        "top20",
        "top30",
        "no_interactions",
        "strong_interactions_only",
        "catboost_native_only",
        "one_hot_only",
        "only_target_encoded_cat_plus_numeric",
    ]

    splits_base = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_full, y))
    ablation_rows = []

    catboost_available = True
    try:
        import catboost  # noqa: F401
    except Exception:
        catboost_available = False

    for fs_name in feature_set_names:
        cols = featset(fs_name)
        if not cols:
            continue
        Xtr = X_train_full[cols].copy()
        Xte = X_test_full[cols].copy()

        # CatBoost
        if catboost_available:
            try:
                oof_cb, test_cb = train_catboost_oof(
                    X=Xtr,
                    y=y,
                    X_test=Xte,
                    splits=splits_base,
                    params={"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
                )
                thr_cb, sc_cb = find_best_threshold(y, oof_cb)
                lab_cb = (test_cb >= thr_cb).astype(int)
                path_cb = f"stage3_submissions/submission_ablation_catboost_{fs_name}.csv"
                save_submission(path_cb, test_ids, lab_cb)
                ablation_rows.append(
                    {
                        "feature_set": fs_name,
                        "model": "catboost",
                        "oof_bal_acc": sc_cb,
                        "best_threshold": thr_cb,
                        "test_pos_rate": float(lab_cb.mean()),
                        "file_path": path_cb,
                    }
                )
                candidates.append(
                    Candidate(
                        name=f"ablation_catboost_{fs_name}",
                        method="feature_ablation",
                        oof_bal_acc=float(sc_cb),
                        threshold=float(thr_cb),
                        test_pos_rate=float(lab_cb.mean()),
                        changed_vs_anchor=int(np.sum(lab_cb != anchor_labels)),
                        diff_pct_vs_anchor=float(np.mean(lab_cb != anchor_labels)),
                        note="catboost_ablation",
                        file_path=path_cb,
                    )
                )
            except Exception:
                pass

        # ExtraTrees (fast reference)
        try:
            et_model = ExtraTreesClassifier(
                n_estimators=1200, max_depth=9, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1
            )
            oof_et, test_et = train_oof_model(et_model, Xtr, y, Xte, splits_base)
            thr_et, sc_et = find_best_threshold(y, oof_et)
            lab_et = (test_et >= thr_et).astype(int)
            path_et = f"stage3_submissions/submission_ablation_extratrees_{fs_name}.csv"
            save_submission(path_et, test_ids, lab_et)
            ablation_rows.append(
                {
                    "feature_set": fs_name,
                    "model": "extratrees",
                    "oof_bal_acc": sc_et,
                    "best_threshold": thr_et,
                    "test_pos_rate": float(lab_et.mean()),
                    "file_path": path_et,
                }
            )
        except Exception:
            pass

    feature_ablation_df = pd.DataFrame(ablation_rows).sort_values("oof_bal_acc", ascending=False)
    feature_ablation_df.to_csv("stage3_reports/feature_ablation_report.csv", index=False, encoding="utf-8")

    # -------------------
    # Phase 5: pseudo-labeling
    # -------------------
    # Use best available blend probability as base
    if len(blend_candidates) > 0:
        blend_candidates_sorted = sorted(blend_candidates, key=lambda x: x[4], reverse=True)
        pseudo_base_oof = blend_candidates_sorted[0][1]
        pseudo_base_test = blend_candidates_sorted[0][2]
    else:
        pseudo_base_oof = base_oof_probs
        pseudo_base_test = base_test_probs

    pseudo_rows = []
    pseudo_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_full, y))
    confidence_pairs = [(0.03, 0.97), (0.05, 0.95), (0.07, 0.93), (0.10, 0.90)]
    pseudo_weights = [0.25, 0.5, 0.75]

    if catboost_available:
        from catboost import CatBoostClassifier

        for lo, hi in confidence_pairs:
            mask_conf = (pseudo_base_test <= lo) | (pseudo_base_test >= hi)
            if mask_conf.sum() == 0:
                continue
            pseudo_X = X_test_full.loc[mask_conf].copy()
            pseudo_y = (pseudo_base_test[mask_conf] >= 0.5).astype(int)

            for pw in pseudo_weights:
                # OOF-like on real train folds, with fixed pseudo pool
                oof = np.zeros(len(X_train_full), dtype=float)
                cnt = np.zeros(len(X_train_full), dtype=float)
                test_accum = np.zeros(len(X_test_full), dtype=float)

                for tr_idx, va_idx in pseudo_splits:
                    Xtr = X_train_full.iloc[tr_idx].copy()
                    ytr = y[tr_idx]
                    Xva = X_train_full.iloc[va_idx].copy()
                    yva = y[va_idx]

                    Xtrain_aug = pd.concat([Xtr, pseudo_X], axis=0, ignore_index=True)
                    ytrain_aug = np.concatenate([ytr, pseudo_y])
                    sw = np.concatenate([np.ones(len(Xtr)), np.ones(len(pseudo_X)) * pw])

                    cat_cols = [c for c in Xtrain_aug.columns if not is_numeric_dtype(Xtrain_aug[c])]
                    for c in cat_cols:
                        Xtrain_aug[c] = Xtrain_aug[c].fillna("missing").astype(str)
                        Xva[c] = Xva[c].fillna("missing").astype(str)
                    Xte = X_test_full.copy()
                    for c in cat_cols:
                        Xte[c] = Xte[c].fillna("missing").astype(str)
                    for c in Xtrain_aug.columns:
                        if c not in cat_cols:
                            Xtrain_aug[c] = pd.to_numeric(Xtrain_aug[c], errors="coerce")
                            Xva[c] = pd.to_numeric(Xva[c], errors="coerce")
                            Xte[c] = pd.to_numeric(Xte[c], errors="coerce")

                    mdl = CatBoostClassifier(
                        iterations=2200,
                        depth=4,
                        learning_rate=0.05,
                        l2_leaf_reg=3,
                        loss_function="Logloss",
                        auto_class_weights="Balanced",
                        verbose=False,
                        random_seed=42,
                    )
                    mdl.fit(
                        Xtrain_aug,
                        ytrain_aug,
                        sample_weight=sw,
                        cat_features=cat_cols,
                        eval_set=(Xva, yva),
                        use_best_model=True,
                        early_stopping_rounds=120,
                    )
                    oof[va_idx] += mdl.predict_proba(Xva)[:, 1]
                    cnt[va_idx] += 1
                    test_accum += mdl.predict_proba(Xte)[:, 1]

                oof = oof / np.maximum(cnt, 1.0)
                test_probs = test_accum / len(pseudo_splits)
                thr, sc = find_best_threshold(y, oof)
                labels = (test_probs >= thr).astype(int)
                fname = f"stage3_submissions/submission_pseudo_lo{lo:.2f}_hi{hi:.2f}_w{pw:.2f}.csv"
                save_submission(fname, test_ids, labels)
                pseudo_rows.append(
                    {
                        "lo": lo,
                        "hi": hi,
                        "pseudo_weight": pw,
                        "n_pseudo": int(mask_conf.sum()),
                        "oof_bal_acc": sc,
                        "best_threshold": thr,
                        "test_pos_rate": float(labels.mean()),
                        "changed_vs_anchor": int(np.sum(labels != anchor_labels)),
                        "file_path": fname,
                    }
                )
                candidates.append(
                    Candidate(
                        name=os.path.basename(fname).replace(".csv", ""),
                        method="pseudo_label",
                        oof_bal_acc=float(sc),
                        threshold=float(thr),
                        test_pos_rate=float(labels.mean()),
                        changed_vs_anchor=int(np.sum(labels != anchor_labels)),
                        diff_pct_vs_anchor=float(np.mean(labels != anchor_labels)),
                        note=f"pseudo_{lo}_{hi}_w{pw}",
                        file_path=fname,
                    )
                )

    pseudo_df = pd.DataFrame(pseudo_rows).sort_values("oof_bal_acc", ascending=False)
    pseudo_df.to_csv("stage3_reports/pseudo_label_report.csv", index=False, encoding="utf-8")

    # -------------------
    # Phase 6: train/test shift weighting
    # -------------------
    # train weights from adversarial probs
    train_adv_score = adv_clf.predict_proba(X_adv_m)[:, 1][: len(X_train_full)]
    train_adv_weight = train_adv_score / (np.mean(train_adv_score) + 1e-9)
    mild_weight = 0.5 + 0.5 * train_adv_weight
    strong_weight = train_adv_weight

    shift_rows = []
    if catboost_available:
        shift_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_full, y))
        for mode, sw in [("unweighted", None), ("mild", mild_weight), ("strong", strong_weight)]:
            oof_cb, test_cb = train_catboost_oof(
                X=X_train_full,
                y=y,
                X_test=X_test_full,
                splits=shift_splits,
                params={"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
                sample_weights=sw,
            )
            thr, sc = find_best_threshold(y, oof_cb)
            labels = (test_cb >= thr).astype(int)
            path = f"stage3_submissions/submission_shift_weight_{mode}.csv"
            save_submission(path, test_ids, labels)
            shift_rows.append(
                {
                    "mode": mode,
                    "oof_bal_acc": sc,
                    "threshold": thr,
                    "test_pos_rate": float(labels.mean()),
                    "file_path": path,
                }
            )

        # most test-like rows subsets
        rank_idx = np.argsort(-train_adv_weight)  # descending similar to test
        for keep in [0.70, 0.80, 0.90]:
            n_keep = int(len(X_train_full) * keep)
            keep_idx = rank_idx[:n_keep]
            Xk = X_train_full.iloc[keep_idx].reset_index(drop=True)
            yk = y[keep_idx]
            split_k = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(Xk, yk))
            oof_k, test_k = train_catboost_oof(
                X=Xk,
                y=yk,
                X_test=X_test_full,
                splits=split_k,
                params={"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
                sample_weights=None,
            )
            thr_k, sc_k = find_best_threshold(yk, oof_k)
            labels_k = (test_k >= thr_k).astype(int)
            path_k = f"stage3_submissions/submission_shift_toplike_{int(keep*100)}.csv"
            save_submission(path_k, test_ids, labels_k)
            shift_rows.append(
                {
                    "mode": f"toplike_{int(keep*100)}",
                    "oof_bal_acc": sc_k,
                    "threshold": thr_k,
                    "test_pos_rate": float(labels_k.mean()),
                    "file_path": path_k,
                }
            )

    # -------------------
    # Phase 7: noisy-label handling
    # -------------------
    noisy_rows = []
    # Use base_oof + model disagreement proxy if available
    if src["oof"].shape[1] > 1:
        oof_stack = src["oof"].values
        oof_mean = np.mean(oof_stack, axis=1)
        oof_std = np.std(oof_stack, axis=1)
    else:
        oof_mean = base_oof_probs
        oof_std = np.zeros_like(base_oof_probs)

    noise_score = y * (1 - oof_mean) + (1 - y) * oof_mean + 0.5 * oof_std
    order = np.argsort(-noise_score)

    if catboost_available:
        noise_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_full, y))
        # downweight hard examples
        for frac, downw in [(0.01, 0.5), (0.02, 0.5), (0.03, 0.5), (0.03, 0.25)]:
            sw = np.ones(len(y))
            n = max(1, int(len(y) * frac))
            sw[order[:n]] = downw
            oof_n, test_n = train_catboost_oof(
                X=X_train_full,
                y=y,
                X_test=X_test_full,
                splits=noise_splits,
                params={"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
                sample_weights=sw,
            )
            thr_n, sc_n = find_best_threshold(y, oof_n)
            labels_n = (test_n >= thr_n).astype(int)
            path_n = f"stage3_submissions/submission_noisy_downweight_frac{frac:.2f}_w{downw:.2f}.csv"
            save_submission(path_n, test_ids, labels_n)
            noisy_rows.append(
                {
                    "strategy": "downweight",
                    "frac": frac,
                    "weight": downw,
                    "oof_bal_acc": sc_n,
                    "threshold": thr_n,
                    "test_pos_rate": float(labels_n.mean()),
                    "file_path": path_n,
                }
            )

        # remove suspicious
        for frac in [0.01, 0.02, 0.03]:
            n = max(1, int(len(y) * frac))
            keep_idx = np.setdiff1d(np.arange(len(y)), order[:n])
            Xk = X_train_full.iloc[keep_idx].reset_index(drop=True)
            yk = y[keep_idx]
            split_k = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(Xk, yk))
            oof_k, test_k = train_catboost_oof(
                X=Xk,
                y=yk,
                X_test=X_test_full,
                splits=split_k,
                params={"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
                sample_weights=None,
            )
            thr_k, sc_k = find_best_threshold(yk, oof_k)
            labels_k = (test_k >= thr_k).astype(int)
            path_k = f"stage3_submissions/submission_noisy_removed_frac{frac:.2f}.csv"
            save_submission(path_k, test_ids, labels_k)
            noisy_rows.append(
                {
                    "strategy": "remove",
                    "frac": frac,
                    "weight": np.nan,
                    "oof_bal_acc": sc_k,
                    "threshold": thr_k,
                    "test_pos_rate": float(labels_k.mean()),
                    "file_path": path_k,
                }
            )

    # -------------------
    # Phase 8: class-weight search
    # -------------------
    cw_rows = []
    neg = float((y == 0).sum())
    pos = float((y == 1).sum())
    natural_ratio = neg / max(pos, 1.0)
    multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    cw_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_full, y))

    if catboost_available:
        for m in multipliers:
            pw = natural_ratio * m
            # use manual class_weights
            oof_cw, test_cw = train_catboost_oof(
                X=X_train_full,
                y=y,
                X_test=X_test_full,
                splits=cw_splits,
                params={
                    "depth": 4,
                    "learning_rate": 0.05,
                    "l2_leaf_reg": 3,
                    "auto_class_weights": None,
                    "class_weights": [1.0, pw],
                },
            )
            thr_cw, sc_cw = find_best_threshold(y, oof_cw)
            labels_cw = (test_cw >= thr_cw).astype(int)
            path_cw = f"stage3_submissions/submission_classweight_m{m:.2f}.csv"
            save_submission(path_cw, test_ids, labels_cw)
            cw_rows.append(
                {
                    "model": "catboost",
                    "multiplier": m,
                    "effective_pos_weight": pw,
                    "oof_bal_acc": sc_cw,
                    "threshold": thr_cw,
                    "test_pos_rate": float(labels_cw.mean()),
                    "file_path": path_cw,
                }
            )
    else:
        # fallback logistic class_weight
        for m in multipliers:
            cw = {0: 1.0, 1: m}
            lr = LogisticRegression(max_iter=5000, class_weight=cw, solver="liblinear", random_state=42)
            oof_lr, test_lr = train_oof_model(lr, X_train_full, y, X_test_full, cw_splits)
            thr_lr, sc_lr = find_best_threshold(y, oof_lr)
            labels_lr = (test_lr >= thr_lr).astype(int)
            path_lr = f"stage3_submissions/submission_classweight_logreg_m{m:.2f}.csv"
            save_submission(path_lr, test_ids, labels_lr)
            cw_rows.append(
                {
                    "model": "logreg",
                    "multiplier": m,
                    "effective_pos_weight": m,
                    "oof_bal_acc": sc_lr,
                    "threshold": thr_lr,
                    "test_pos_rate": float(labels_lr.mean()),
                    "file_path": path_lr,
                }
            )

    pd.DataFrame(cw_rows).sort_values("oof_bal_acc", ascending=False).to_csv(
        "stage3_reports/class_weight_search.csv", index=False, encoding="utf-8"
    )

    # -------------------
    # Phase 9: stability + anchor-flip generation
    # -------------------
    # seed ensemble via logistic (fast) to get uncertainty estimates
    seed_oofs = []
    seed_tests = []
    for seed in SEEDS_STAGE3:
        split_seed = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train_full, y))
        lr = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=seed)
        oof_s, test_s = train_oof_model(lr, X_train_full, y, X_test_full, split_seed)
        seed_oofs.append(oof_s)
        seed_tests.append(test_s)
    seed_oofs = np.vstack(seed_oofs)
    seed_tests = np.vstack(seed_tests)
    test_mean = np.mean(seed_tests, axis=0)
    test_std = np.std(seed_tests, axis=0)

    # stable consensus labels
    stable_thr = threshold_from_target_posrate(test_mean, anchor_pos_rate)
    stable_labels = (test_mean >= stable_thr).astype(int)
    stable_mask = (test_std <= np.quantile(test_std, 0.5))
    conservative_labels = anchor_labels.copy()
    conservative_labels[stable_mask] = stable_labels[stable_mask]
    save_submission("stage3_submissions/submission_stable_consensus.csv", test_ids, conservative_labels)

    # Anchor-based flip budgets
    # Use best blend confidence if available else base probs
    conf_probs = pseudo_base_test if "pseudo_base_test" in locals() else base_test_probs
    pred_labels = (conf_probs >= base_thr).astype(int)
    disagree_mask = pred_labels != anchor_labels
    confidence = np.abs(conf_probs - anchor_labels)  # disagreement strength for anchor labels
    disagree_idx = np.where(disagree_mask)[0]
    order_conf = disagree_idx[np.argsort(-confidence[disagree_idx])]

    flip_budgets = [1, 2, 3, 5, 10, 15, 20, max(1, int(0.05 * len(anchor_labels))), max(1, int(0.10 * len(anchor_labels)))]
    flip_rows = []
    for b in sorted(list(set(flip_budgets))):
        b = int(min(b, len(order_conf)))
        if b <= 0:
            continue
        flip_idx = order_conf[:b]
        labels = anchor_labels.copy()
        labels[flip_idx] = pred_labels[flip_idx]
        path = f"stage3_submissions/submission_anchor_flip_{b}.csv"
        save_submission(path, test_ids, labels)
        n01 = int(np.sum((anchor_labels == 0) & (labels == 1)))
        n10 = int(np.sum((anchor_labels == 1) & (labels == 0)))
        avg_conf = float(np.mean(confidence[flip_idx])) if len(flip_idx) > 0 else 0.0
        flip_rows.append(
            {
                "flip_budget": b,
                "n_0_to_1": n01,
                "n_1_to_0": n10,
                "avg_flip_confidence": avg_conf,
                "result_pos_rate": float(labels.mean()),
                "file_path": path,
            }
        )
        candidates.append(
            Candidate(
                name=f"anchor_flip_{b}",
                method="anchor_flip",
                oof_bal_acc=float(base_oof_score),  # proxy
                threshold=float(base_thr),
                test_pos_rate=float(labels.mean()),
                changed_vs_anchor=int(np.sum(labels != anchor_labels)),
                diff_pct_vs_anchor=float(np.mean(labels != anchor_labels)),
                note=f"flip_budget={b}",
                file_path=path,
            )
        )

    flip_df = pd.DataFrame(flip_rows)
    flip_df.to_csv("stage3_reports/anchor_flip_report.csv", index=False, encoding="utf-8")

    # -------------------
    # Build final candidate catalog
    # -------------------
    # Include threshold and blend candidates already in candidates
    # plus top ablation/pseudo/classweight/noise/shift via created files parsing
    def add_candidates_from_report(df: pd.DataFrame, method: str, name_col: str, oof_col: str, thr_col: str, pos_col: str, path_col: str, note: str):
        if df is None or len(df) == 0:
            return
        for row in df.head(min(40, len(df))).itertuples(index=False):
            path = getattr(row, path_col)
            if not isinstance(path, str) or not path:
                continue
            if not os.path.exists(path):
                continue
            sub = pd.read_csv(path)
            lab = sub["Прогрессия"].astype(int).values
            candidates.append(
                Candidate(
                    name=str(getattr(row, name_col)) if hasattr(row, name_col) else os.path.basename(path).replace(".csv", ""),
                    method=method,
                    oof_bal_acc=float(getattr(row, oof_col)),
                    threshold=float(getattr(row, thr_col)),
                    test_pos_rate=float(getattr(row, pos_col)),
                    changed_vs_anchor=int(np.sum(lab != anchor_labels)),
                    diff_pct_vs_anchor=float(np.mean(lab != anchor_labels)),
                    note=note,
                    file_path=path,
                )
            )

    # load reports
    blend_top = pd.read_csv("stage3_reports/blend_grid.csv") if os.path.exists("stage3_reports/blend_grid.csv") else pd.DataFrame()
    pseudo_top = pd.read_csv("stage3_reports/pseudo_label_report.csv") if os.path.exists("stage3_reports/pseudo_label_report.csv") else pd.DataFrame()
    ablation_top = pd.read_csv("stage3_reports/feature_ablation_report.csv") if os.path.exists("stage3_reports/feature_ablation_report.csv") else pd.DataFrame()
    cw_top = pd.read_csv("stage3_reports/class_weight_search.csv") if os.path.exists("stage3_reports/class_weight_search.csv") else pd.DataFrame()
    thr_top = pd.read_csv("stage3_reports/threshold_grid.csv") if os.path.exists("stage3_reports/threshold_grid.csv") else pd.DataFrame()
    noisy_top = pd.DataFrame(noisy_rows)
    shift_top = pd.DataFrame(shift_rows)

    if len(blend_top):
        add_candidates_from_report(blend_top.sort_values("oof_bal_acc", ascending=False), "blend_grid", "blend_name", "oof_bal_acc", "best_threshold", "test_pos_rate", "file_path", "blend_grid")
    if len(pseudo_top):
        # create display name
        pseudo_top = pseudo_top.copy()
        pseudo_top["name"] = pseudo_top.apply(lambda r: f"pseudo_lo{r['lo']}_hi{r['hi']}_w{r['pseudo_weight']}", axis=1)
        add_candidates_from_report(pseudo_top.sort_values("oof_bal_acc", ascending=False), "pseudo", "name", "oof_bal_acc", "best_threshold", "test_pos_rate", "file_path", "pseudo")
    if len(ablation_top):
        ablation_top = ablation_top.copy()
        ablation_top["name"] = ablation_top["model"].astype(str) + "_" + ablation_top["feature_set"].astype(str)
        add_candidates_from_report(ablation_top.sort_values("oof_bal_acc", ascending=False), "ablation", "name", "oof_bal_acc", "best_threshold", "test_pos_rate", "file_path", "ablation")
    if len(cw_top):
        cw_top = cw_top.copy()
        cw_top["name"] = cw_top["model"].astype(str) + "_cw_" + cw_top["multiplier"].astype(str)
        add_candidates_from_report(cw_top.sort_values("oof_bal_acc", ascending=False), "class_weight", "name", "oof_bal_acc", "threshold", "test_pos_rate", "file_path", "class_weight")
    if len(noisy_top):
        noisy_top = noisy_top.copy()
        noisy_top["name"] = noisy_top["strategy"].astype(str) + "_" + noisy_top["frac"].astype(str)
        add_candidates_from_report(noisy_top.sort_values("oof_bal_acc", ascending=False), "noisy", "name", "oof_bal_acc", "threshold", "test_pos_rate", "file_path", "noisy")
    if len(shift_top):
        shift_top = shift_top.copy()
        shift_top["name"] = shift_top["mode"].astype(str)
        add_candidates_from_report(shift_top.sort_values("oof_bal_acc", ascending=False), "shift", "name", "oof_bal_acc", "threshold", "test_pos_rate", "file_path", "shift")

    # dedup candidates by file path
    uniq = {}
    for c in candidates:
        if c.file_path not in uniq or c.oof_bal_acc > uniq[c.file_path].oof_bal_acc:
            uniq[c.file_path] = c
    candidates = list(uniq.values())

    # rank candidates: not only OOF, include diversity bands
    # scoring heuristic
    ranked_rows = []
    for c in candidates:
        diversity_bonus = 0.0
        if 0.01 <= c.diff_pct_vs_anchor <= 0.05:
            diversity_bonus = 0.003
        elif 0.05 < c.diff_pct_vs_anchor <= 0.12:
            diversity_bonus = 0.002
        elif 0.12 < c.diff_pct_vs_anchor <= 0.25:
            diversity_bonus = 0.001
        # penalty for extreme pos rate
        pos_penalty = 0.0
        if c.test_pos_rate < 0.15 or c.test_pos_rate > 0.80:
            pos_penalty = 0.01
        final_score = c.oof_bal_acc + diversity_bonus - pos_penalty
        ranked_rows.append(
            {
                "name": c.name,
                "method": c.method,
                "oof_bal_acc": c.oof_bal_acc,
                "threshold": c.threshold,
                "test_pos_rate": c.test_pos_rate,
                "changed_vs_anchor": c.changed_vs_anchor,
                "diff_pct_vs_anchor": c.diff_pct_vs_anchor,
                "note": c.note,
                "file_path": c.file_path,
                "ranking_score": final_score,
            }
        )
    rank_df = pd.DataFrame(ranked_rows).sort_values("ranking_score", ascending=False).reset_index(drop=True)
    rank_df.to_csv("stage3_reports/candidate_ranking.csv", index=False, encoding="utf-8")

    # choose final stable candidate:
    # top ranking with diff <= 0.12 preferred; fallback top overall
    stable_pool = rank_df[(rank_df["diff_pct_vs_anchor"] <= 0.12)]
    if len(stable_pool) > 0:
        best_row = stable_pool.iloc[0]
    else:
        best_row = rank_df.iloc[0]

    # copy selected candidate to submission.csv
    selected_file = best_row["file_path"]
    sel = pd.read_csv(selected_file)
    sel = sel[["ID", "Прогрессия"]]
    sel.to_csv("submission.csv", index=False, encoding="utf-8")

    # final recommendations
    rec_lines = []
    rec_lines.append(f"Current best known public score: {CURRENT_BEST_PUBLIC_SCORE:.5f}")
    rec_lines.append(f"Detected anchor submission file: {anchor_path}")
    rec_lines.append(f"Anchor detection reason: {anchor_reason}")
    rec_lines.append(f"Anchor positive rate: {anchor_pos_rate:.4f}")
    rec_lines.append("")
    rec_lines.append("Top 20 recommended submissions to upload next:")
    top20 = rank_df.head(20)
    for i, row in enumerate(top20.itertuples(index=False), 1):
        reason = []
        if row.diff_pct_vs_anchor <= 0.05:
            reason.append("close high-confidence variant")
        elif row.diff_pct_vs_anchor <= 0.12:
            reason.append("medium-diversity candidate")
        else:
            reason.append("aggressive-diversity candidate")
        if "pseudo" in row.method:
            reason.append("pseudo-label experiment")
        if "anchor_flip" in row.method:
            reason.append("anchor-controlled flips")
        if "blend" in row.method:
            reason.append("ensemble diversity")
        rec_lines.append(
            f"{i}. {row.file_path}\n"
            f"   method={row.method}, threshold={row.threshold:.4f}, pos_rate={row.test_pos_rate:.4f}, "
            f"oof_bal_acc={row.oof_bal_acc:.6f}, changed_vs_anchor={int(row.changed_vs_anchor)} ({row.diff_pct_vs_anchor:.3%})\n"
            f"   why: {', '.join(reason)}"
        )
    rec_lines.append("")
    rec_lines.append("Upload order strategy:")
    rec_lines.append("1) Closest high-confidence threshold variants (1-5% label diff).")
    rec_lines.append("2) Diverse blend variants (5-12% label diff).")
    rec_lines.append("3) Pseudo-label variants.")
    rec_lines.append("4) Anchor-flip variants.")
    rec_lines.append("5) Aggressive variants (>12% label diff).")

    with open("stage3_reports/final_recommendations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rec_lines) + "\n")

    # stage3 report summary
    report_lines = []
    report_lines.append("Stage3 LB Optimizer Report")
    report_lines.append("=" * 40)
    report_lines.append(f"train_shape: {train.shape}")
    report_lines.append(f"test_shape: {test.shape}")
    report_lines.append(f"target_source: {target_source}")
    report_lines.append(f"labeled_train_rows: {len(train_labeled)}")
    report_lines.append(f"dropped_unlabeled_rows: {dropped_unlabeled}")
    report_lines.append(f"selected_prob_source: {src_name}")
    report_lines.append(f"selected_prob_model: {base_model_name}")
    report_lines.append(f"selected_prob_oof_bal_acc: {base_oof_score:.6f}")
    report_lines.append(f"selected_prob_threshold: {base_thr:.4f}")
    report_lines.append(f"anchor_submission: {anchor_path}")
    report_lines.append(f"anchor_reason: {anchor_reason}")
    report_lines.append(f"adversarial_auc: {adv_auc:.6f}")
    report_lines.append(f"high_shift_features: {high_shift_features}")
    report_lines.append(f"generated_submissions_count: {len(list(set([c.file_path for c in candidates])))}")
    report_lines.append(f"selected_final_submission_source: {selected_file}")
    report_lines.append(f"selected_final_oof_bal_acc: {best_row['oof_bal_acc']:.6f}")
    report_lines.append(f"selected_final_threshold: {best_row['threshold']:.4f}")
    report_lines.append(f"selected_final_pos_rate: {best_row['test_pos_rate']:.4f}")
    report_lines.append(f"selected_final_diff_vs_anchor: {best_row['diff_pct_vs_anchor']:.3%}")
    report_lines.append("")
    report_lines.append("Optional library status:")
    try:
        import catboost  # noqa: F401
        report_lines.append("catboost: available")
    except Exception as e:
        report_lines.append(f"catboost: unavailable ({e})")
    try:
        import lightgbm  # noqa: F401
        report_lines.append("lightgbm: available")
    except Exception as e:
        report_lines.append(f"lightgbm: unavailable ({e})")
    try:
        import xgboost  # noqa: F401
        report_lines.append("xgboost: available")
    except Exception as e:
        report_lines.append(f"xgboost: unavailable ({e})")

    with open("stage3_reports/stage3_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    # Also save a compact machine-readable summary
    summary_json = {
        "current_best_public_score": CURRENT_BEST_PUBLIC_SCORE,
        "anchor_submission": anchor_path,
        "anchor_reason": anchor_reason,
        "selected_probability_source": src_name,
        "selected_probability_model": base_model_name,
        "selected_probability_oof_bal_acc": float(base_oof_score),
        "selected_probability_threshold": float(base_thr),
        "selected_final_submission_source": str(selected_file),
        "selected_final_oof_bal_acc": float(best_row["oof_bal_acc"]),
        "selected_final_threshold": float(best_row["threshold"]),
        "selected_final_test_pos_rate": float(best_row["test_pos_rate"]),
        "selected_final_diff_vs_anchor": float(best_row["diff_pct_vs_anchor"]),
    }
    with open("stage3_reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print("Stage3 pipeline finished.")
    print(f"Best OOF proxy: {best_row['oof_bal_acc']:.6f}")
    print(f"Best threshold: {best_row['threshold']:.4f}")
    print(f"Test positive rate: {best_row['test_pos_rate']:.4f}")
    print("Top 10 recommended submissions:")
    for row in rank_df.head(10).itertuples(index=False):
        print(
            f"- {row.file_path} | method={row.method} | oof={row.oof_bal_acc:.6f} | "
            f"thr={row.threshold:.4f} | diff={row.diff_pct_vs_anchor:.3%}"
        )
    print("Warning: CV/OFFLINE ranking can diverge from public LB; upload diverse candidates in recommended order.")


if __name__ == "__main__":
    main()
