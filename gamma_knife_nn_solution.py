import copy
import os
import random
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


LEAKAGE_COLUMNS = [
    "Прогрессия",
    "Интракраниальная прогрессия",
    "Локальный рецидив",
    "Дистантные метастазы",
]

ID_CANDIDATES = ["ID", "Id", "id"]
MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "#ref!", "-", "--"}
YES_TOKENS = {"есть", "да", "1", "true", "yes", "y", "+"}
NO_TOKENS = {"нет", "0", "false", "no", "n", "не удален", "не удалён", "отсутствует"}
MALE_TOKENS = {"м", "муж", "мужской", "male", "man"}
FEMALE_TOKENS = {"ж", "жен", "женский", "female", "woman"}


@dataclass
class TrainConfig:
    n_splits: int = 5
    seed: int = 42
    max_epochs: int = 200
    early_stopping_patience: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_weighted_sampler: bool = True
    hidden_dims: Tuple[int, int, int] = (256, 128, 64)
    dropout: Tuple[float, float, float] = (0.35, 0.25, 0.15)
    threshold_grid: Tuple[float, float, float] = (0.05, 0.95, 0.01)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_text(value) -> Optional[str]:
    if pd.isna(value):
        return np.nan
    text = str(value).replace("\xa0", " ").strip().lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    if text in MISSING_TOKENS:
        return np.nan
    return text


def normalize_col_key(name: str) -> str:
    normalized = normalize_text(name)
    if pd.isna(normalized):
        return ""
    return re.sub(r"[^a-zа-я0-9]+", "", normalized)


def read_csv_robust(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1251", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[read_csv_robust] Loaded {path} with encoding={enc}, shape={df.shape}")
            return df
        except Exception as err:
            last_error = err
            print(f"[read_csv_robust] Failed {path} with encoding={enc}: {err}")
    raise RuntimeError(f"Could not read {path} with supported encodings. Last error: {last_error}")


def parse_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    out = df.copy()
    date_cols = []
    parsed_map = {}
    for col in out.columns:
        col_norm = normalize_text(col)
        if isinstance(col_norm, str) and "дата" in col_norm:
            parsed_col = f"{col}__parsed_dt"
            parsed = pd.to_datetime(out[col], format="%d.%m.%Y", errors="coerce")
            unresolved_mask = parsed.isna() & out[col].notna()
            if unresolved_mask.any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed_fallback = pd.to_datetime(
                        out.loc[unresolved_mask, col], dayfirst=True, errors="coerce"
                    )
                parsed.loc[unresolved_mask] = parsed_fallback
            out[parsed_col] = parsed
            date_cols.append(col)
            parsed_map[col] = parsed_col
    return out, date_cols, parsed_map


def build_col_lookup(columns: List[str]) -> Dict[str, str]:
    lookup = {}
    for col in columns:
        key = normalize_col_key(col)
        if key and key not in lookup:
            lookup[key] = col
    return lookup


def find_col(lookup: Dict[str, str], candidates: List[str]) -> Optional[str]:
    candidate_keys = [normalize_col_key(c) for c in candidates]
    for key in candidate_keys:
        if key in lookup:
            return lookup[key]
    for key in candidate_keys:
        for existing_key, col in lookup.items():
            if key and (key in existing_key or existing_key in key):
                return col
    return None


def to_numeric_series(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string")
    s = s.str.replace("\xa0", " ", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-]+", "", regex=True)
    s = s.replace({"": np.nan, ".": np.nan, "-": np.nan, "--": np.nan})
    return pd.to_numeric(s, errors="coerce")


def map_binary_yes_no(series: pd.Series) -> pd.Series:
    normalized = series.map(normalize_text)

    def _map(v):
        if pd.isna(v):
            return np.nan
        if v in YES_TOKENS:
            return 1.0
        if v in NO_TOKENS:
            return 0.0
        return np.nan

    return normalized.map(_map).astype("float64")


def map_sex(series: pd.Series) -> pd.Series:
    normalized = series.map(normalize_text)

    def _map(v):
        if pd.isna(v):
            return np.nan
        if v in MALE_TOKENS:
            return 0.0
        if v in FEMALE_TOKENS:
            return 1.0
        return np.nan

    return normalized.map(_map).astype("float64")


def safe_days_diff(df: pd.DataFrame, end_col: Optional[str], start_col: Optional[str]) -> pd.Series:
    if end_col is None or start_col is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return (df[end_col] - df[start_col]).dt.days.astype("float64")


def prepare_target(train_df: pd.DataFrame) -> Tuple[pd.Series, str]:
    cols = [str(c).strip() for c in train_df.columns]
    if "Прогрессия" in cols:
        raw = train_df["Прогрессия"]
        if is_numeric_dtype(raw):
            y = pd.to_numeric(raw, errors="coerce")
        else:
            normalized = raw.map(normalize_text)
            positive = YES_TOKENS.union({"лр", "дм", "лр+дм", "прогрессия"})
            negative = NO_TOKENS.union({"безпрогрессии"})
            y = normalized.map(lambda v: 1.0 if v in positive else (0.0 if v in negative else np.nan))
        return y.astype("float64"), "Прогрессия"

    if "Интракраниальная прогрессия" in cols:
        mapping = {"нет": 0.0, "лр": 1.0, "дм": 1.0, "лр+дм": 1.0, "лр + дм": 1.0}
        normalized = train_df["Интракраниальная прогрессия"].map(normalize_text)
        y = normalized.map(mapping).astype("float64")
        return y, "Интракраниальная прогрессия"

    raise ValueError(
        "Target not found. Expected 'Прогрессия' or 'Интракраниальная прогрессия' in train.csv."
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    for col in out.columns:
        if is_string_dtype(out[col]) or out[col].dtype == object:
            out[col] = out[col].map(normalize_text)

    out, date_cols, parsed_map = parse_dates(out)
    lookup = build_col_lookup(list(out.columns))

    def _find(cands: List[str]) -> Optional[str]:
        return find_col(lookup, cands)

    numeric_candidates = {
        "rh_procedures": ["Число РХ процедур на ГН"],
        "karnofsky": ["Индекс Карновского"],
        "lesion_count": ["Число очагов в ГМ"],
        "sum_volume": ["Суммарный объём очагов", "Суммарный объем очагов"],
        "max_volume": ["Объём максимального очага", "Объем максимального очага"],
    }
    resolved_numeric_cols = {}
    for alias, cands in numeric_candidates.items():
        col = _find(cands)
        resolved_numeric_cols[alias] = col
        if col is not None:
            out[col] = to_numeric_series(out[col])
            out[alias] = out[col]
        else:
            out[alias] = np.nan

    sex_col = _find(["Пол"])
    if sex_col is not None:
        out["sex_bin"] = map_sex(out[sex_col])
        out = out.drop(columns=[sex_col], errors="ignore")
    else:
        out["sex_bin"] = np.nan

    mutation_col = _find(["Активирующие мутации"])
    if mutation_col is not None:
        out["activating_mutation_bin"] = map_binary_yes_no(out[mutation_col])
        out = out.drop(columns=[mutation_col], errors="ignore")
    else:
        out["activating_mutation_bin"] = np.nan

    extracranial_col = _find(["Экстракраниальные метастазы"])
    if extracranial_col is not None:
        out["extracranial_metastases_bin"] = map_binary_yes_no(out[extracranial_col])
        out = out.drop(columns=[extracranial_col], errors="ignore")
    else:
        out["extracranial_metastases_bin"] = np.nan

    birth_date_col = _find(["Дата рождения"])
    diagnosis_date_col = _find(
        ["Дата постановки онкологического диагноза / начала первичного лечения"]
    )
    primary_removal_date_col = _find(["Дата удаления первичного очага"])
    mgm_date_col = _find(["Дата развития МГМ"])
    ovgm_date_col = _find(["Дата проведения ОВГМ"])
    brain_op_date_col = _find(["Дата операции на ГМ"])
    rx_date_col = _find(["Дата 1-ой РХ", "Дата 1-ои РХ", "Дата 1 ой РХ"])

    birth_dt = parsed_map.get(birth_date_col) if birth_date_col else None
    diagnosis_dt = parsed_map.get(diagnosis_date_col) if diagnosis_date_col else None
    primary_removal_dt = parsed_map.get(primary_removal_date_col) if primary_removal_date_col else None
    mgm_dt = parsed_map.get(mgm_date_col) if mgm_date_col else None
    ovgm_dt = parsed_map.get(ovgm_date_col) if ovgm_date_col else None
    brain_op_dt = parsed_map.get(brain_op_date_col) if brain_op_date_col else None
    rx_dt = parsed_map.get(rx_date_col) if rx_date_col else None

    out["primary_removed_flag"] = (
        out[primary_removal_dt].notna().astype("float64") if primary_removal_dt else 0.0
    )
    out["ovgm_flag"] = out[ovgm_dt].notna().astype("float64") if ovgm_dt else 0.0
    out["brain_operation_flag"] = out[brain_op_dt].notna().astype("float64") if brain_op_dt else 0.0

    out["age_at_rx_days"] = safe_days_diff(out, rx_dt, birth_dt)
    out["age_at_rx_years"] = out["age_at_rx_days"] / 365.25
    out["diagnosis_to_mgm_days"] = safe_days_diff(out, mgm_dt, diagnosis_dt)
    out["mgm_to_rx_days"] = safe_days_diff(out, rx_dt, mgm_dt)
    out["diagnosis_to_rx_days"] = safe_days_diff(out, rx_dt, diagnosis_dt)
    out["primary_removal_to_rx_days"] = safe_days_diff(out, rx_dt, primary_removal_dt)
    out["ovgm_to_rx_days"] = safe_days_diff(out, rx_dt, ovgm_dt)
    out["brain_operation_to_rx_days"] = safe_days_diff(out, rx_dt, brain_op_dt)

    for original_date_col, parsed_col in parsed_map.items():
        out[f"{original_date_col}__year"] = out[parsed_col].dt.year.astype("float64")
        out[f"{original_date_col}__month"] = out[parsed_col].dt.month.astype("float64")

    lesion_count = out["lesion_count"]
    sum_volume = out["sum_volume"]
    max_volume = out["max_volume"]
    karnofsky = out["karnofsky"]

    out["volume_ratio_max_to_sum"] = np.where(sum_volume > 0, max_volume / sum_volume, np.nan)
    out["avg_volume_per_lesion"] = np.where(lesion_count > 0, sum_volume / lesion_count, np.nan)
    out["lesion_sum_volume_interaction"] = lesion_count * sum_volume
    out["lesion_max_volume_interaction"] = lesion_count * max_volume
    out["karnofsky_per_age"] = np.where(out["age_at_rx_years"] > 0, karnofsky / out["age_at_rx_years"], np.nan)
    out["ovgm_x_lesion_count"] = out["ovgm_flag"] * lesion_count
    out["brain_operation_x_max_volume"] = out["brain_operation_flag"] * max_volume

    skewed_base_features = [
        "lesion_count",
        "sum_volume",
        "max_volume",
        "age_at_rx_days",
        "diagnosis_to_mgm_days",
        "mgm_to_rx_days",
        "diagnosis_to_rx_days",
        "primary_removal_to_rx_days",
        "ovgm_to_rx_days",
        "brain_operation_to_rx_days",
    ]
    for col in skewed_base_features:
        if col in out.columns:
            out[f"log1p_{col}"] = np.log1p(out[col].clip(lower=0))

    important_numeric = [
        "karnofsky",
        "lesion_count",
        "sum_volume",
        "max_volume",
        "age_at_rx_years",
        "diagnosis_to_rx_days",
    ]
    for col in important_numeric:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype("float64")

    out = out.drop(columns=date_cols + list(parsed_map.values()), errors="ignore")
    return out


def drop_leakage_and_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    drop_cols = [c for c in LEAKAGE_COLUMNS if c in out.columns]
    for cand in ID_CANDIDATES:
        if cand in out.columns:
            drop_cols.append(cand)
    return out.drop(columns=drop_cols, errors="ignore")


def build_preprocessor(X_train_df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = [c for c in X_train_df.columns if is_numeric_dtype(X_train_df[c])]
    cat_cols = [c for c in X_train_df.columns if c not in num_cols]

    transformers = []
    if num_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        raise RuntimeError("No features left after preprocessing. Check feature engineering and column dropping.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, num_cols, cat_cols


def cast_categorical_to_object(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype("object")
    return out


class TabularNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, int, int] = (256, 128, 64),
        dropout: Tuple[float, float, float] = (0.35, 0.25, 0.15),
    ):
        super().__init__()
        h1, h2, h3 = hidden_dims
        d1, d2, d3 = dropout
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, h1),
            nn.SiLU(),
            nn.BatchNorm1d(h1),
            nn.Dropout(d1),
            nn.Linear(h1, h2),
            nn.SiLU(),
            nn.BatchNorm1d(h2),
            nn.Dropout(d2),
            nn.Linear(h2, h3),
            nn.SiLU(),
            nn.Dropout(d3),
            nn.Linear(h3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    low: float = 0.05,
    high: float = 0.95,
    step: float = 0.01,
) -> Tuple[float, float]:
    best_thr = 0.5
    best_score = -1.0
    thresholds = np.arange(low, high + 1e-9, step)
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        try:
            score = balanced_accuracy_score(y_true, y_pred)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, float(best_score)


def predict_proba_torch(
    model: nn.Module, X_np: np.ndarray, device: torch.device, batch_size: int = 256
) -> np.ndarray:
    model.eval()
    X_tensor = torch.from_numpy(X_np.astype(np.float32))
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb).squeeze(1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds.append(probs)
    return np.concatenate(preds)


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    seed: int,
) -> Dict:
    set_seed(seed)
    input_dim = X_train.shape[1]
    model = TabularNN(input_dim=input_dim, hidden_dims=config.hidden_dims, dropout=config.dropout).to(device)

    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32))

    if config.use_weighted_sampler:
        counts = np.bincount(y_train.astype(int), minlength=2)
        class_weights = {0: 1.0 / max(counts[0], 1), 1: 1.0 / max(counts[1], 1)}
        sample_weights = np.array([class_weights[int(v)] for v in y_train], dtype=np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=config.batch_size,
            sampler=sampler,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=config.batch_size,
            shuffle=True,
        )

    pos_count = max(float(y_train.sum()), 1.0)
    neg_count = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight_value = neg_count / pos_count
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_proba = None
    best_score = -1.0
    best_threshold = 0.5
    best_epoch = 0
    patience_counter = 0

    thr_low, thr_high, thr_step = config.threshold_grid

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        val_proba = predict_proba_torch(model, X_val, device=device, batch_size=512)
        val_thr, val_score = find_best_threshold(
            y_true=y_val,
            y_proba=val_proba,
            low=thr_low,
            high=thr_high,
            step=thr_step,
        )
        scheduler.step(val_score)

        if val_score > best_score + 1e-7:
            best_score = val_score
            best_threshold = val_thr
            best_val_proba = val_proba.copy()
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 20 == 0:
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"    epoch={epoch:03d} loss={mean_loss:.4f} "
                f"val_bal_acc={val_score:.5f} best={best_score:.5f} lr={current_lr:.6f}"
            )

        if patience_counter >= config.early_stopping_patience:
            print(f"    early stopping at epoch={epoch}, best_epoch={best_epoch}, best_val_bal_acc={best_score:.5f}")
            break

    model.load_state_dict(best_state)

    if best_val_proba is None:
        best_val_proba = predict_proba_torch(model, X_val, device=device, batch_size=512)
        best_threshold, best_score = find_best_threshold(
            y_true=y_val,
            y_proba=best_val_proba,
            low=thr_low,
            high=thr_high,
            step=thr_step,
        )

    return {
        "model": model,
        "best_val_proba": best_val_proba,
        "best_val_score": float(best_score),
        "best_threshold": float(best_threshold),
        "best_epoch": int(best_epoch),
        "input_dim": input_dim,
    }


def train_full_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    seed: int,
    epochs: int,
) -> nn.Module:
    set_seed(seed)
    model = TabularNN(input_dim=X_train.shape[1], hidden_dims=config.hidden_dims, dropout=config.dropout).to(device)

    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32))

    if config.use_weighted_sampler:
        counts = np.bincount(y_train.astype(int), minlength=2)
        class_weights = {0: 1.0 / max(counts[0], 1), 1: 1.0 / max(counts[1], 1)}
        sample_weights = np.array([class_weights[int(v)] for v in y_train], dtype=np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=config.batch_size,
            sampler=sampler,
            shuffle=False,
        )
    else:
        loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config.batch_size, shuffle=True)

    pos_count = max(float(y_train.sum()), 1.0)
    neg_count = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight_value = neg_count / pos_count
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    print(f"[final_model] Training on full data for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"    [final_model] epoch={epoch:03d} loss={np.mean(losses):.4f}")
    return model


def compute_permutation_importance(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    threshold: float,
    device: torch.device,
    repeats: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    baseline_proba = predict_proba_torch(model, X_val, device=device, batch_size=512)
    baseline_pred = (baseline_proba >= threshold).astype(int)
    baseline_score = balanced_accuracy_score(y_val, baseline_pred)

    importances = []
    X_work = X_val.copy()
    n_features = X_work.shape[1]
    print(f"[perm_importance] baseline balanced accuracy={baseline_score:.5f}, features={n_features}")

    for j in range(n_features):
        drops = []
        original = X_work[:, j].copy()
        for _ in range(repeats):
            perm = original.copy()
            rng.shuffle(perm)
            X_work[:, j] = perm
            proba = predict_proba_torch(model, X_work, device=device, batch_size=512)
            pred = (proba >= threshold).astype(int)
            score = balanced_accuracy_score(y_val, pred)
            drops.append(baseline_score - score)
        X_work[:, j] = original
        importances.append(float(np.mean(drops)))

    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_bal_acc_drop": importances,
        }
    ).sort_values("importance_bal_acc_drop", ascending=False)
    return imp_df


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    config = TrainConfig()
    set_seed(config.seed)
    device = detect_device()
    print(f"[main] Using device: {device}")

    train_path = "train.csv"
    test_path = "test.csv"
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Expected train.csv and test.csv in current working directory.")

    train_raw = read_csv_robust(train_path)
    test_raw = read_csv_robust(test_path)

    y_raw, target_col_used = prepare_target(train_raw)
    labeled_mask = y_raw.notna()
    dropped_unlabeled = int((~labeled_mask).sum())
    if dropped_unlabeled > 0:
        print(f"[main] Warning: dropping {dropped_unlabeled} train rows with unknown target label.")

    train_labeled = train_raw.loc[labeled_mask].copy()
    y = y_raw.loc[labeled_mask].astype(int).to_numpy()

    train_row_ids = train_labeled.index.to_numpy()
    for cand in ID_CANDIDATES:
        if cand in train_labeled.columns:
            train_row_ids = train_labeled[cand].to_numpy()
            break

    train_features = build_features(train_labeled)
    test_features = build_features(test_raw)

    X = drop_leakage_and_id(train_features)
    X_test = drop_leakage_and_id(test_features)

    print(f"[main] Target source: {target_col_used}")
    unique, counts = np.unique(y, return_counts=True)
    class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"[main] Class distribution: {class_dist}")
    print(f"[main] Train feature shape before CV preprocessing: {X.shape}")
    print(f"[main] Test feature shape before CV preprocessing: {X_test.shape}")

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    min_class_count = min(n_pos, n_neg)
    n_splits = min(config.n_splits, min_class_count) if min_class_count >= 2 else 2
    if n_splits < 2:
        raise RuntimeError("Not enough labeled samples per class for cross-validation.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    oof_proba = np.zeros(len(X), dtype=np.float64)
    fold_test_proba = []
    fold_scores = []
    fold_thresholds = []
    fold_epochs = []
    fold_artifacts = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n[CV] Fold {fold_idx}/{n_splits}")
        X_tr_df = X.iloc[tr_idx].copy()
        y_tr = y[tr_idx]
        X_va_df = X.iloc[va_idx].copy()
        y_va = y[va_idx]

        preprocessor, num_cols, cat_cols = build_preprocessor(X_tr_df)
        X_tr_df_cast = cast_categorical_to_object(X_tr_df, cat_cols)
        X_va_df_cast = cast_categorical_to_object(X_va_df, cat_cols)
        X_test_cast = cast_categorical_to_object(X_test, cat_cols)

        X_tr_np = preprocessor.fit_transform(X_tr_df_cast)
        X_va_np = preprocessor.transform(X_va_df_cast)
        X_te_np = preprocessor.transform(X_test_cast)

        if hasattr(X_tr_np, "toarray"):
            X_tr_np = X_tr_np.toarray()
        if hasattr(X_va_np, "toarray"):
            X_va_np = X_va_np.toarray()
        if hasattr(X_te_np, "toarray"):
            X_te_np = X_te_np.toarray()

        X_tr_np = X_tr_np.astype(np.float32)
        X_va_np = X_va_np.astype(np.float32)
        X_te_np = X_te_np.astype(np.float32)

        print(
            f"  transformed dims: train={X_tr_np.shape}, val={X_va_np.shape}, test={X_te_np.shape}, "
            f"num_cols={len(num_cols)}, cat_cols={len(cat_cols)}"
        )

        fold_result = train_fold(
            X_train=X_tr_np,
            y_train=y_tr,
            X_val=X_va_np,
            y_val=y_va,
            config=config,
            device=device,
            seed=config.seed + fold_idx,
        )

        val_proba = fold_result["best_val_proba"]
        val_thr = fold_result["best_threshold"]
        val_score = fold_result["best_val_score"]
        val_pred = (val_proba >= val_thr).astype(int)
        score_check = balanced_accuracy_score(y_va, val_pred)

        print(
            f"  fold_bal_acc={val_score:.5f} (check={score_check:.5f}), "
            f"best_threshold={val_thr:.3f}, best_epoch={fold_result['best_epoch']}"
        )

        test_proba_fold = predict_proba_torch(fold_result["model"], X_te_np, device=device, batch_size=512)

        oof_proba[va_idx] = val_proba
        fold_test_proba.append(test_proba_fold)
        fold_scores.append(val_score)
        fold_thresholds.append(val_thr)
        fold_epochs.append(fold_result["best_epoch"])

        feature_names = list(preprocessor.get_feature_names_out())
        fold_artifacts.append(
            {
                "fold": fold_idx,
                "score": val_score,
                "threshold": val_thr,
                "model": fold_result["model"],
                "X_val": X_va_np,
                "y_val": y_va,
                "feature_names": feature_names,
            }
        )

    global_thr, oof_score = find_best_threshold(
        y_true=y,
        y_proba=oof_proba,
        low=config.threshold_grid[0],
        high=config.threshold_grid[1],
        step=config.threshold_grid[2],
    )
    print("\n[CV] Fold balanced accuracy scores:", [round(s, 6) for s in fold_scores])
    print("[CV] Fold best thresholds:", [round(t, 4) for t in fold_thresholds])
    print(f"[CV] OOF balanced accuracy={oof_score:.6f}, global best threshold={global_thr:.4f}")

    cv_test_proba_mean = np.mean(np.vstack(fold_test_proba), axis=0)

    print("\n[final_model] Fitting preprocessor on full training data...")
    preprocessor_full, _, cat_cols_full = build_preprocessor(X)
    X_cast_full = cast_categorical_to_object(X, cat_cols_full)
    X_test_cast_full = cast_categorical_to_object(X_test, cat_cols_full)

    X_full_np = preprocessor_full.fit_transform(X_cast_full)
    X_test_full_np = preprocessor_full.transform(X_test_cast_full)

    if hasattr(X_full_np, "toarray"):
        X_full_np = X_full_np.toarray()
    if hasattr(X_test_full_np, "toarray"):
        X_test_full_np = X_test_full_np.toarray()

    X_full_np = X_full_np.astype(np.float32)
    X_test_full_np = X_test_full_np.astype(np.float32)
    feature_count_full = X_full_np.shape[1]
    print(f"[final_model] Full transformed feature count: {feature_count_full}")

    usable_epochs = [e for e in fold_epochs if e > 0]
    final_epochs = int(np.clip(np.median(usable_epochs) if usable_epochs else 80, 30, 150))
    final_model = train_full_model(
        X_train=X_full_np,
        y_train=y,
        config=config,
        device=device,
        seed=config.seed + 10_000,
        epochs=final_epochs,
    )
    full_test_proba = predict_proba_torch(final_model, X_test_full_np, device=device, batch_size=512)

    final_test_proba = 0.5 * cv_test_proba_mean + 0.5 * full_test_proba
    test_pred = (final_test_proba >= global_thr).astype(int)

    id_col = None
    for cand in ID_CANDIDATES:
        if cand in test_raw.columns:
            id_col = cand
            break
    if id_col is not None:
        test_ids = test_raw[id_col]
    else:
        print("[main] Warning: ID column not found in test.csv. Using 1..N as IDs.")
        test_ids = pd.Series(np.arange(1, len(test_raw) + 1))

    submission = pd.DataFrame({"ID": test_ids.astype(int), "Прогрессия": test_pred.astype(int)})
    submission.to_csv("submission.csv", index=False, encoding="utf-8")
    print("[output] Saved submission.csv")

    oof_df = pd.DataFrame(
        {
            "row_id": train_row_ids,
            "target": y.astype(int),
            "oof_proba": oof_proba,
            "oof_pred": (oof_proba >= global_thr).astype(int),
        }
    )
    oof_df.to_csv("oof_predictions.csv", index=False, encoding="utf-8")
    print("[output] Saved oof_predictions.csv")

    best_fold_idx = int(np.argmax(fold_scores))
    best_fold_artifact = fold_artifacts[best_fold_idx]
    try:
        print(f"[perm_importance] Computing on fold {best_fold_artifact['fold']}...")
        imp_df = compute_permutation_importance(
            model=best_fold_artifact["model"],
            X_val=best_fold_artifact["X_val"],
            y_val=best_fold_artifact["y_val"],
            feature_names=best_fold_artifact["feature_names"],
            threshold=best_fold_artifact["threshold"],
            device=device,
            repeats=3,
            seed=config.seed,
        )
        imp_df.to_csv("feature_importance_permutation.csv", index=False, encoding="utf-8")
        print("[output] Saved feature_importance_permutation.csv")
    except Exception as err:
        print(f"[perm_importance] Skipped due to error: {err}")

    report_lines = [
        "Gamma Knife Binary Classification Training Report",
        "=" * 52,
        f"Train rows (original): {len(train_raw)}",
        f"Train rows used (labeled): {len(train_labeled)}",
        f"Train rows dropped (unlabeled target): {dropped_unlabeled}",
        f"Test rows: {len(test_raw)}",
        f"Target source column: {target_col_used}",
        f"Target distribution: {class_dist}",
        f"Feature count after full preprocessing: {feature_count_full}",
        f"Fold scores (balanced accuracy): {[round(s, 6) for s in fold_scores]}",
        f"Fold thresholds: {[round(t, 4) for t in fold_thresholds]}",
        f"OOF balanced accuracy: {round(oof_score, 6)}",
        f"Selected global threshold: {round(global_thr, 6)}",
        f"Final full-model epochs: {final_epochs}",
        "Output files: submission.csv, oof_predictions.csv, training_report.txt",
    ]
    if os.path.exists("feature_importance_permutation.csv"):
        report_lines.append("Optional output: feature_importance_permutation.csv")
    with open("training_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print("[output] Saved training_report.txt")
    print("[done] Pipeline finished successfully.")


if __name__ == "__main__":
    main()
