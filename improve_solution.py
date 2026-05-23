import os
import re
import json
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")

TARGET_CANDIDATES = ["Прогрессия", "Интракраниальная прогрессия"]
LEAKAGE_COLUMNS = ["Прогрессия", "Интракраниальная прогрессия", "Локальный рецидив", "Дистантные метастазы"]
ID_CANDIDATES = ["ID", "Id", "id"]
SEEDS = [42, 777, 2025]

YES_TOKENS = {"есть", "да", "1", "true", "yes", "y", "+"}
NO_TOKENS = {"нет", "0", "false", "no", "n", "не удален", "не удалён", "отсутствует"}
MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "—", "--", "#ref!"}
MALE_TOKENS = {"м", "муж", "мужской", "male"}
FEMALE_TOKENS = {"ж", "жен", "женский", "female"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(v):
    if pd.isna(v):
        return np.nan
    s = str(v).replace("\xa0", " ").strip().lower().replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    if s in MISSING_TOKENS:
        return np.nan
    return s


def normalize_col_name(c: str) -> str:
    c = str(c).strip()
    c = re.sub(r"\s+", " ", c)
    return c


def norm_key(s: str) -> str:
    s = normalize_text(s)
    if pd.isna(s):
        return ""
    return re.sub(r"[^a-zа-я0-9]+", "", s)


def read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp1251", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[load] {path} encoding={enc} shape={df.shape}")
            return df
        except Exception:
            pass
    raise RuntimeError(f"Cannot read {path}")


def find_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lk = {norm_key(c): c for c in columns}
    for cand in candidates:
        k = norm_key(cand)
        if k in lk:
            return lk[k]
    for cand in candidates:
        k = norm_key(cand)
        for ek, col in lk.items():
            if k and (k in ek or ek in k):
                return col
    return None


def to_numeric(s: pd.Series) -> pd.Series:
    if is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = s.astype("string")
    x = x.str.replace("\xa0", " ", regex=False).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    x = x.str.replace(r"[^0-9\.\-]+", "", regex=True).replace({"": np.nan, ".": np.nan, "-": np.nan})
    return pd.to_numeric(x, errors="coerce")


def map_binary(series: pd.Series) -> pd.Series:
    z = series.map(normalize_text)
    return z.map(lambda v: 1.0 if v in YES_TOKENS else (0.0 if v in NO_TOKENS else np.nan)).astype(float)


def map_sex(series: pd.Series) -> pd.Series:
    z = series.map(normalize_text)
    return z.map(lambda v: 0.0 if v in MALE_TOKENS else (1.0 if v in FEMALE_TOKENS else np.nan)).astype(float)


def prepare_target(train: pd.DataFrame) -> Tuple[pd.Series, str]:
    if "Прогрессия" in train.columns:
        t = train["Прогрессия"]
        if is_numeric_dtype(t):
            return pd.to_numeric(t, errors="coerce"), "Прогрессия"
        z = t.map(normalize_text)
        pos = YES_TOKENS.union({"лр", "дм", "лр+дм", "лр + дм"})
        y = z.map(lambda v: 1.0 if v in pos else (0.0 if v in NO_TOKENS else np.nan))
        return y.astype(float), "Прогрессия"
    if "Интракраниальная прогрессия" in train.columns:
        z = train["Интракраниальная прогрессия"].map(normalize_text)
        mp = {"нет": 0.0, "лр": 1.0, "дм": 1.0, "лр+дм": 1.0, "лр + дм": 1.0}
        return z.map(mp).astype(float), "Интракраниальная прогрессия"
    raise ValueError("Target not found")


def data_audit(train: pd.DataFrame, test: pd.DataFrame, y: pd.Series) -> None:
    lines = []
    lines.append(f"train_shape: {train.shape}")
    lines.append(f"test_shape: {test.shape}")
    lines.append("columns:")
    lines.extend([f"- {c}" for c in train.columns])
    lines.append(f"target_distribution:\n{y.value_counts(dropna=False).to_string()}")
    lines.append("missing_train:")
    lines.extend((train.isna().sum().sort_values(ascending=False)).to_string().split("\n"))
    lines.append("missing_test:")
    lines.extend((test.isna().sum().sort_values(ascending=False)).to_string().split("\n"))
    obj_cols = [c for c in train.columns if train[c].dtype == object or is_string_dtype(train[c])]
    lines.append("object_unique_values_top:")
    for c in obj_cols:
        vc = train[c].map(normalize_text).value_counts(dropna=False).head(20)
        lines.append(f"[{c}]")
        lines.append(vc.to_string())
    train_only = sorted(set(train.columns) - set(test.columns))
    test_only = sorted(set(test.columns) - set(train.columns))
    lines.append(f"train_only_columns: {train_only}")
    lines.append(f"test_only_columns: {test_only}")
    dup_rows = train.duplicated().sum()
    lines.append(f"duplicate_rows_train: {dup_rows}")
    for c in ID_CANDIDATES:
        if c in train.columns:
            lines.append(f"duplicate_{c}_train: {train[c].duplicated().sum()}")
        if c in test.columns:
            lines.append(f"duplicate_{c}_test: {test[c].duplicated().sum()}")
    with open("preprocessing_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


@dataclass
class FeatureStats:
    clip_bounds: Dict[str, Tuple[float, float]]
    lesion_count_q90: Optional[float]
    sum_volume_q90: Optional[float]
    max_volume_q90: Optional[float]


def clean_and_base(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [normalize_col_name(c) for c in x.columns]
    for c in x.columns:
        if is_string_dtype(x[c]) or x[c].dtype == object:
            x[c] = x[c].map(normalize_text)
    return x


def add_engineered(df: pd.DataFrame, stats: Optional[FeatureStats] = None) -> Tuple[pd.DataFrame, FeatureStats]:
    x = df.copy()
    cols = list(x.columns)

    date_cols = [c for c in cols if "дата" in norm_key(c)]
    parsed = {}
    for c in date_cols:
        p = f"{c}__dt"
        base = pd.to_datetime(x[c], format="%d.%m.%Y", errors="coerce")
        mask = base.isna() & x[c].notna()
        if mask.any():
            base.loc[mask] = pd.to_datetime(x.loc[mask, c], dayfirst=True, errors="coerce")
        x[p] = base
        parsed[c] = p

    c_birth = find_col(cols, ["Дата рождения"])
    c_diag = find_col(cols, ["Дата постановки онкологического диагноза / начала первичного лечения"])
    c_mgm = find_col(cols, ["Дата развития МГМ"])
    c_rx = find_col(cols, ["Дата 1-ой РХ"])
    c_rem = find_col(cols, ["Дата удаления первичного очага"])
    c_ovgm = find_col(cols, ["Дата проведения ОВГМ"])
    c_op = find_col(cols, ["Дата операции на ГМ"])

    def dt(c):
        return parsed.get(c) if c else None

    x["primary_removed_flag"] = (x[dt(c_rem)].notna().astype(float) if dt(c_rem) else 0.0)
    x["ovgm_flag"] = (x[dt(c_ovgm)].notna().astype(float) if dt(c_ovgm) else 0.0)
    x["brain_operation_flag"] = (x[dt(c_op)].notna().astype(float) if dt(c_op) else 0.0)

    def days(a, b):
        if a is None or b is None:
            return pd.Series(np.nan, index=x.index, dtype=float)
        return (x[a] - x[b]).dt.days.astype(float)

    x["age_at_rx_days"] = days(dt(c_rx), dt(c_birth))
    x["age_at_rx_years"] = x["age_at_rx_days"] / 365.25
    x["diagnosis_to_mgm_days"] = days(dt(c_mgm), dt(c_diag))
    x["mgm_to_rx_days"] = days(dt(c_rx), dt(c_mgm))
    x["diagnosis_to_rx_days"] = days(dt(c_rx), dt(c_diag))
    x["primary_removal_to_rx_days"] = days(dt(c_rx), dt(c_rem))
    x["ovgm_to_rx_days"] = days(dt(c_rx), dt(c_ovgm))
    x["brain_operation_to_rx_days"] = days(dt(c_rx), dt(c_op))

    n_proc = find_col(cols, ["Число РХ процедур на ГН"])
    karn = find_col(cols, ["Индекс Карновского"])
    lesions = find_col(cols, ["Число очагов в ГМ"])
    sum_vol = find_col(cols, ["Суммарный объём очагов", "Суммарный объем очагов"])
    max_vol = find_col(cols, ["Объём максимального очага", "Объем максимального очага"])

    num_cols_source = [n_proc, karn, lesions, sum_vol, max_vol]
    for c in num_cols_source:
        if c:
            x[c] = to_numeric(x[c])

    if lesions:
        x["lesion_count"] = x[lesions]
    else:
        x["lesion_count"] = np.nan
    if sum_vol:
        x["sum_volume"] = x[sum_vol]
    else:
        x["sum_volume"] = np.nan
    if max_vol:
        x["max_volume"] = x[max_vol]
    else:
        x["max_volume"] = np.nan
    if karn:
        x["karnovsky"] = x[karn]
    else:
        x["karnovsky"] = np.nan

    sex_col = find_col(cols, ["Пол"])
    mut_col = find_col(cols, ["Активирующие мутации"])
    ext_col = find_col(cols, ["Экстракраниальные метастазы"])
    treat_col = find_col(cols, ["Лекарственное лечение"])
    diag_col = find_col(cols, ["Онкологический диагноз"])

    x["sex_bin"] = map_sex(x[sex_col]) if sex_col else np.nan
    x["mutations_bin"] = map_binary(x[mut_col]) if mut_col else np.nan
    x["extracranial_bin"] = map_binary(x[ext_col]) if ext_col else np.nan

    x["volume_ratio_max_to_sum"] = np.where(x["sum_volume"] > 0, x["max_volume"] / x["sum_volume"], np.nan)
    x["avg_volume_per_lesion"] = np.where(x["lesion_count"] > 0, x["sum_volume"] / x["lesion_count"], np.nan)
    x["lesion_volume_product"] = x["lesion_count"] * x["sum_volume"]
    x["max_lesion_product"] = x["lesion_count"] * x["max_volume"]
    x["karnovsky_age_ratio"] = np.where(x["age_at_rx_years"] > 0, x["karnovsky"] / x["age_at_rx_years"], np.nan)
    x["karnovsky_deficit"] = 100 - x["karnovsky"]
    x["ovgm_x_lesions"] = x["ovgm_flag"] * x["lesion_count"]
    x["operation_x_max_volume"] = x["brain_operation_flag"] * x["max_volume"]
    x["mutations_x_treatment"] = np.where(
        x["mutations_bin"].fillna(0) > 0,
        (x[treat_col].fillna("none") if treat_col else "none").astype(str),
        "no_mut",
    )
    x["diagnosis_x_treatment"] = (
        (x[diag_col].fillna("none").astype(str) + "__" + x[treat_col].fillna("none").astype(str))
        if (diag_col and treat_col)
        else "none"
    )

    interval_cols = [
        "age_at_rx_days",
        "diagnosis_to_mgm_days",
        "mgm_to_rx_days",
        "diagnosis_to_rx_days",
        "primary_removal_to_rx_days",
        "ovgm_to_rx_days",
        "brain_operation_to_rx_days",
    ]

    if stats is None:
        clip_bounds = {}
        for c in interval_cols:
            vals = x[c].dropna()
            if len(vals) > 8:
                clip_bounds[c] = (float(vals.quantile(0.01)), float(vals.quantile(0.99)))
            else:
                clip_bounds[c] = (-36500.0, 36500.0)
        q90_les = float(x["lesion_count"].quantile(0.9)) if x["lesion_count"].notna().any() else np.nan
        q90_sum = float(x["sum_volume"].quantile(0.9)) if x["sum_volume"].notna().any() else np.nan
        q90_max = float(x["max_volume"].quantile(0.9)) if x["max_volume"].notna().any() else np.nan
        stats = FeatureStats(clip_bounds=clip_bounds, lesion_count_q90=q90_les, sum_volume_q90=q90_sum, max_volume_q90=q90_max)

    for c in interval_cols:
        lo, hi = stats.clip_bounds[c]
        x[c] = x[c].clip(lo, hi)
        x[f"{c}_abs"] = x[c].abs()
        x[f"log1p_{c}_pos"] = np.log1p(x[c].clip(lower=0))
        x[f"{c}_missing"] = x[c].isna().astype(float)

    for c in ["lesion_count", "sum_volume", "max_volume", "karnovsky"]:
        x[f"log1p_{c}"] = np.log1p(x[c].clip(lower=0))
        x[f"{c}_missing"] = x[c].isna().astype(float)

    x["high_lesion_count_flag"] = (x["lesion_count"] >= (stats.lesion_count_q90 if not np.isnan(stats.lesion_count_q90) else 1e9)).astype(float)
    x["high_volume_flag"] = (x["sum_volume"] >= (stats.sum_volume_q90 if not np.isnan(stats.sum_volume_q90) else 1e9)).astype(float)
    x["high_max_volume_flag"] = (x["max_volume"] >= (stats.max_volume_q90 if not np.isnan(stats.max_volume_q90) else 1e9)).astype(float)

    keep_dates = [c for c in x.columns if c.endswith("__dt")]
    x = x.drop(columns=keep_dates, errors="ignore")
    return x, stats


def drop_leakage_and_id(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    drop_cols = [c for c in LEAKAGE_COLUMNS if c in x.columns]
    for c in ID_CANDIDATES:
        if c in x.columns:
            drop_cols.append(c)
    return x.drop(columns=drop_cols, errors="ignore")


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], verbose_feature_names_out=False)
    return pre, num_cols, cat_cols


def find_best_threshold(y_true: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    best_thr, best_sc = 0.5, -1.0
    for thr in np.arange(0.01, 0.991, 0.001):
        pred = (p >= thr).astype(int)
        sc = balanced_accuracy_score(y_true, pred)
        if sc > best_sc:
            best_sc, best_thr = sc, float(thr)
    return best_thr, float(best_sc)


def prob_from_model(model, Xv):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xv)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(Xv)
        return 1.0 / (1.0 + np.exp(-z))
    z = model.predict(Xv)
    return np.asarray(z, dtype=float)


def get_optional_models() -> Dict[str, object]:
    models = {}
    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=1500,
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=5,
            loss_function="Logloss",
            eval_metric="Logloss",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
        )
    except Exception:
        print("[warn] catboost not available, skipping.")
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=15,
            class_weight="balanced",
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
        )
    except Exception:
        print("[warn] lightgbm not available, skipping.")
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=1.0,
            random_state=42,
        )
    except Exception:
        print("[warn] xgboost not available, skipping.")
    return models


def cv_sklearn_model(
    name: str,
    model,
    X_all: pd.DataFrame,
    y: np.ndarray,
    X_test_all: pd.DataFrame,
    seeds: List[int],
) -> Dict:
    n = len(X_all)
    n_test = len(X_test_all)
    oof_sum = np.zeros(n)
    oof_cnt = np.zeros(n)
    test_sum = np.zeros(n_test)
    fold_rows = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X_all, y), 1):
            Xtr_raw, Xva_raw = X_all.iloc[tr].copy(), X_all.iloc[va].copy()
            ytr, yva = y[tr], y[va]
            Xte_raw = X_test_all.copy()

            Xtr_eng, stats = add_engineered(Xtr_raw, None)
            Xva_eng, _ = add_engineered(Xva_raw, stats)
            Xte_eng, _ = add_engineered(Xte_raw, stats)

            Xtr = drop_leakage_and_id(Xtr_eng)
            Xva = drop_leakage_and_id(Xva_eng)
            Xte = drop_leakage_and_id(Xte_eng)

            pre, _, _ = build_preprocessor(Xtr)
            Xtr_m = pre.fit_transform(Xtr)
            Xva_m = pre.transform(Xva)
            Xte_m = pre.transform(Xte)

            mdl = clone(model)
            if hasattr(mdl, "random_state"):
                mdl.set_params(random_state=seed)
            mdl.fit(Xtr_m, ytr)

            pva = prob_from_model(mdl, Xva_m)
            pte = prob_from_model(mdl, Xte_m)
            thr, sc = find_best_threshold(yva, pva)

            oof_sum[va] += pva
            oof_cnt[va] += 1
            test_sum += pte
            fold_rows.append({"model": name, "seed": seed, "fold": fold, "fold_bal_acc": sc, "fold_thr": thr})

    oof = np.divide(oof_sum, np.maximum(oof_cnt, 1))
    test_p = test_sum / (len(seeds) * 5)
    gthr, gsc = find_best_threshold(y, oof)
    return {"name": name, "oof": oof, "test": test_p, "cv_score": gsc, "thr": gthr, "fold_report": pd.DataFrame(fold_rows)}


def cv_catboost_model(name: str, model, X_all: pd.DataFrame, y: np.ndarray, X_test_all: pd.DataFrame, seeds: List[int]) -> Dict:
    n = len(X_all)
    n_test = len(X_test_all)
    oof_sum = np.zeros(n)
    oof_cnt = np.zeros(n)
    test_sum = np.zeros(n_test)
    fold_rows = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X_all, y), 1):
            Xtr_raw, Xva_raw = X_all.iloc[tr].copy(), X_all.iloc[va].copy()
            ytr, yva = y[tr], y[va]
            Xte_raw = X_test_all.copy()
            Xtr_eng, stats = add_engineered(Xtr_raw, None)
            Xva_eng, _ = add_engineered(Xva_raw, stats)
            Xte_eng, _ = add_engineered(Xte_raw, stats)
            Xtr = drop_leakage_and_id(Xtr_eng)
            Xva = drop_leakage_and_id(Xva_eng)
            Xte = drop_leakage_and_id(Xte_eng)

            cat_cols = [c for c in Xtr.columns if not is_numeric_dtype(Xtr[c])]
            for c in cat_cols:
                Xtr[c] = Xtr[c].fillna("missing").astype(str)
                Xva[c] = Xva[c].fillna("missing").astype(str)
                Xte[c] = Xte[c].fillna("missing").astype(str)

            mdl = clone(model)
            mdl.set_params(random_seed=seed)
            mdl.fit(Xtr, ytr, cat_features=cat_cols, eval_set=(Xva, yva), use_best_model=True, verbose=False)
            pva = mdl.predict_proba(Xva)[:, 1]
            pte = mdl.predict_proba(Xte)[:, 1]
            thr, sc = find_best_threshold(yva, pva)

            oof_sum[va] += pva
            oof_cnt[va] += 1
            test_sum += pte
            fold_rows.append({"model": name, "seed": seed, "fold": fold, "fold_bal_acc": sc, "fold_thr": thr})

    oof = np.divide(oof_sum, np.maximum(oof_cnt, 1))
    test_p = test_sum / (len(seeds) * 5)
    gthr, gsc = find_best_threshold(y, oof)
    return {"name": name, "oof": oof, "test": test_p, "cv_score": gsc, "thr": gthr, "fold_report": pd.DataFrame(fold_rows)}


def build_ensembles(model_results: List[Dict], y: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    leaderboard = []
    out = {}
    names = [m["name"] for m in model_results]
    oof_mat = np.vstack([m["oof"] for m in model_results]).T
    test_mat = np.vstack([m["test"] for m in model_results]).T
    weights = np.array([max(m["cv_score"], 1e-6) for m in model_results], dtype=float)
    weights = weights / weights.sum()

    variants = {
        "avg_ensemble": (oof_mat.mean(axis=1), test_mat.mean(axis=1)),
        "weighted_ensemble": (oof_mat @ weights, test_mat @ weights),
        "rank_ensemble": (
            pd.DataFrame(oof_mat).rank(axis=0, method="average", pct=True).mean(axis=1).values,
            pd.DataFrame(test_mat).rank(axis=0, method="average", pct=True).mean(axis=1).values,
        ),
    }

    vote_oof = np.zeros(len(y))
    vote_test = np.zeros(test_mat.shape[0])
    for m in model_results:
        vote_oof += (m["oof"] >= m["thr"]).astype(int)
        vote_test += (m["test"] >= m["thr"]).astype(int)
    vote_oof = vote_oof / len(model_results)
    vote_test = vote_test / len(model_results)
    variants["majority_vote"] = (vote_oof, vote_test)

    meta = LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=42)
    meta.fit(oof_mat, y)
    stack_oof = meta.predict_proba(oof_mat)[:, 1]
    stack_test = meta.predict_proba(test_mat)[:, 1]
    variants["stacking"] = (stack_oof, stack_test)

    for nm, (oofp, testp) in variants.items():
        thr, sc = find_best_threshold(y, oofp)
        out[nm] = {"oof": oofp, "test": testp, "thr": thr, "score": sc}
        leaderboard.append({"model": nm, "cv_bal_acc": sc, "best_threshold": thr})

    for m in model_results:
        leaderboard.append({"model": m["name"], "cv_bal_acc": m["cv_score"], "best_threshold": m["thr"]})
    lb = pd.DataFrame(leaderboard).sort_values("cv_bal_acc", ascending=False).reset_index(drop=True)
    return lb, out


def main():
    set_seed(42)
    train = read_csv_robust("train.csv")
    test = read_csv_robust("test.csv")
    train.columns = [normalize_col_name(c) for c in train.columns]
    test.columns = [normalize_col_name(c) for c in test.columns]
    y_raw, y_source = prepare_target(train)
    data_audit(train, test, y_raw)

    labeled = y_raw.notna()
    dropped = int((~labeled).sum())
    train_l = train.loc[labeled].copy()
    y = y_raw.loc[labeled].astype(int).to_numpy()
    print(f"[target] source={y_source}, labeled={len(train_l)}, dropped_unlabeled={dropped}")

    results = []
    base_models = {
        "logreg": LogisticRegression(class_weight="balanced", C=0.7, max_iter=5000, solver="liblinear"),
        "histgb": HistGradientBoostingClassifier(
            learning_rate=0.03, max_depth=4, max_leaf_nodes=31, min_samples_leaf=12, random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=1200, max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "extratrees": ExtraTreesClassifier(
            n_estimators=1500, max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    for name, mdl in base_models.items():
        print(f"[model] training {name}...")
        res = cv_sklearn_model(name, mdl, train_l, y, test, SEEDS)
        print(f"  -> cv_bal_acc={res['cv_score']:.6f} thr={res['thr']:.3f}")
        results.append(res)

    opt_models = get_optional_models()
    for name, mdl in opt_models.items():
        print(f"[model] training {name}...")
        if name == "catboost":
            res = cv_catboost_model(name, mdl, train_l, y, test, SEEDS)
        else:
            res = cv_sklearn_model(name, mdl, train_l, y, test, SEEDS)
        print(f"  -> cv_bal_acc={res['cv_score']:.6f} thr={res['thr']:.3f}")
        results.append(res)

    if not results:
        raise RuntimeError("No models trained.")

    fold_reports = [r["fold_report"] for r in results]
    threshold_report = pd.concat(fold_reports, ignore_index=True)
    threshold_report.to_csv("threshold_report.csv", index=False, encoding="utf-8")

    lb, ens = build_ensembles(results, y)
    lb.to_csv("model_scores.csv", index=False, encoding="utf-8")
    print("\n[leaderboard]\n", lb.to_string(index=False))

    oof_df = pd.DataFrame({"y_true": y})
    test_probs_df = pd.DataFrame()
    for r in results:
        oof_df[f"oof_{r['name']}"] = r["oof"]
        test_probs_df[f"test_{r['name']}"] = r["test"]
    for k, v in ens.items():
        oof_df[f"oof_{k}"] = v["oof"]
        test_probs_df[f"test_{k}"] = v["test"]
    oof_df.to_csv("oof_predictions.csv", index=False, encoding="utf-8")
    test_probs_df.to_csv("test_probabilities_by_model.csv", index=False, encoding="utf-8")

    best_row = lb.iloc[0]
    best_name = str(best_row["model"])
    print(f"[best] {best_name} cv_bal_acc={best_row['cv_bal_acc']:.6f}")

    variant_map = {r["name"]: {"test": r["test"], "thr": r["thr"]} for r in results}
    for k, v in ens.items():
        variant_map[k] = {"test": v["test"], "thr": v["thr"]}

    id_col = None
    for c in ID_CANDIDATES:
        if c in test.columns:
            id_col = c
            break
    if id_col is None:
        print("[warn] ID column missing in test, using 1..N")
        ids = np.arange(1, len(test) + 1)
    else:
        ids = test[id_col].astype(int).values

    def save_sub(name: str, probs: np.ndarray, thr: float):
        pred = (probs >= thr).astype(int)
        sub = pd.DataFrame({"ID": ids, "Прогрессия": pred})
        sub.to_csv(name, index=False, encoding="utf-8")

    for nm in ["avg_ensemble", "weighted_ensemble", "rank_ensemble", "stacking"]:
        if nm in variant_map:
            save_sub(f"submission_{nm}.csv", variant_map[nm]["test"], variant_map[nm]["thr"])

    best_single = max(results, key=lambda x: x["cv_score"])
    save_sub("submission_best_single_model.csv", best_single["test"], best_single["thr"])
    save_sub("submission.csv", variant_map[best_name]["test"], variant_map[best_name]["thr"])

    base_eng, st = add_engineered(train_l, None)
    selected = drop_leakage_and_id(base_eng).columns.tolist()
    with open("selected_features.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected) + "\n")

    fi_rows = []
    best_non_ens = best_single["name"]
    model_for_fi = None
    if best_non_ens in ["rf", "extratrees"]:
        model_for_fi = clone(base_models[best_non_ens])
        Xtr, _ = add_engineered(train_l.copy(), st)
        Xtr = drop_leakage_and_id(Xtr)
        pre, _, _ = build_preprocessor(Xtr)
        Xm = pre.fit_transform(Xtr)
        model_for_fi.fit(Xm, y)
        feats = pre.get_feature_names_out()
        imp = model_for_fi.feature_importances_
        fi_rows = [{"feature": f, "importance": float(i)} for f, i in zip(feats, imp)]
    if fi_rows:
        pd.DataFrame(fi_rows).sort_values("importance", ascending=False).head(50).to_csv(
            "feature_importance.csv", index=False, encoding="utf-8"
        )

    report = {
        "train_shape_original": train.shape,
        "train_labeled": int(len(train_l)),
        "test_shape": test.shape,
        "target_source": y_source,
        "dropped_unlabeled_rows": dropped,
        "best_variant": best_name,
        "best_cv_bal_acc": float(best_row["cv_bal_acc"]),
        "best_threshold": float(best_row["best_threshold"]),
        "score_warning": (
            "Potential overfitting/leakage risk: CV score is very high compared to prior public score."
            if float(best_row["cv_bal_acc"]) > 0.90
            else "No extreme CV inflation detected."
        ),
    }
    with open("training_report.txt", "w", encoding="utf-8") as f:
        for k, v in report.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTop leaderboard:\n")
        f.write(lb.head(15).to_string(index=False))
        f.write("\n")
    with open("training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[done] saved submission.csv and diagnostics.")


if __name__ == "__main__":
    main()
