"""
NOAA Storm Events — Real-Time Multi-Class Storm Classification & Concept Drift Detection
=========================================================================================
Data          : Real NOAA Storm Events CSVs (same directory as this script)
                  StormEvents_details-ftp_v1.0_d1950_c20260323.csv
                  StormEvents_details-ftp_v1.0_d1950_c20260323 2.csv
                  StormEvents_details-ftp_v1.0_d1950_c20260323 3.csv
Target        : EVENT_TYPE (raw NOAA string label — no mapping, no compression)
Features      : BEGIN_LAT, BEGIN_LON, MAGNITUDE, DURATION_HOURS,
                INJURIES_DIRECT, DEATHS_DIRECT, DAMAGE_PROPERTY_K, DAMAGE_CROPS_K
Model         : Random Forest Classifier (multi-class, class_weight=balanced)
                + Random Forest Regressor (predicts DAMAGE_PROPERTY_K, DURATION_HOURS,
                  INJURIES_DIRECT — continuous targets with drift-aware retraining)
Novelty       : Isolation Forest (unsupervised, no labels)
Drift Engine  : KS + MMD + MeanChange + PerformanceDrop
                P(drift) = 0.25·KS + 0.25·MMD + 0.15·MeanChange + 0.35·PerformanceDrop
Explainability: Per-feature KS attribution + PSI + RF feature importance
Backtesting   : Walk-forward folds on every confirmed drift event
Retraining    : Cost tracking + accuracy gain analysis
Drift Memory  : Cosine-similarity recurrence detection
"""

import os
import glob
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import time
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NOAA Storm Events · Drift Detection",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { padding: 0rem 1rem; }
.stMetric {
    background-color: #1a1d2e;
    padding: 15px; border-radius: 10px;
    border: 2px solid #4CAF50;
}
.stMetric label { color:#FFFFFF !important; font-size:15px !important; font-weight:bold !important; }
.stMetric [data-testid="stMetricValue"] { color:#4CAF50 !important; font-size:26px !important; }
.drift-alert {
    padding:1.2rem; border-radius:.5rem; background:#ff4b4b;
    color:white; font-weight:bold; font-size:18px;
    text-align:center; margin:8px 0; animation:pulse 1s infinite;
}
.no-drift {
    padding:1.2rem; border-radius:.5rem; background:#00aa44;
    color:white; font-weight:bold; font-size:18px; text-align:center; margin:8px 0;
}
.novelty-alert {
    padding:1rem; border-radius:.5rem; background:#ff8c00;
    color:white; font-weight:bold; font-size:16px; text-align:center; margin:6px 0;
}
.info-box {
    padding:1rem; border-radius:.5rem; background:#1e2a3a;
    border-left: 4px solid #4CAF50; margin:8px 0; font-size:14px;
}
.reg-box {
    padding:1rem; border-radius:.5rem; background:#1e2a3a;
    border-left: 4px solid #a78bfa; margin:8px 0; font-size:14px;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.7} }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES  = sorted(glob.glob(
    os.path.join(SCRIPT_DIR, "datasets", "StormEvents_details*.csv")
))

FEATURE_COLS = [
    "BEGIN_LAT", "BEGIN_LON", "MAGNITUDE", "DURATION_HOURS",
    "INJURIES_DIRECT", "DEATHS_DIRECT", "DAMAGE_PROPERTY_K", "DAMAGE_CROPS_K",
]
FEATURE_DISPLAY = {
    "BEGIN_LAT":         "Latitude (°)",
    "BEGIN_LON":         "Longitude (°)",
    "MAGNITUDE":         "Magnitude",
    "DURATION_HOURS":    "Duration (hrs)",
    "INJURIES_DIRECT":   "Injuries",
    "DEATHS_DIRECT":     "Deaths",
    "DAMAGE_PROPERTY_K": "Property Damage ($K)",
    "DAMAGE_CROPS_K":    "Crop Damage ($K)",
}
TARGET_COL = "EVENT_TYPE"   # classification target

# ── Regression targets and their input features ───────────────────────────────
# Each regressor uses all FEATURE_COLS EXCEPT itself as input
REGRESSION_TARGETS = {
    "DAMAGE_PROPERTY_K": {
        "label":       "Property Damage ($K)",
        "unit":        "$K",
        "color":       "#f59e0b",
        "description": "Predict property damage from storm characteristics",
    },
    "DURATION_HOURS": {
        "label":       "Storm Duration (hrs)",
        "unit":        "hrs",
        "color":       "#a78bfa",
        "description": "Predict storm duration from location & intensity",
    },
    "INJURIES_DIRECT": {
        "label":       "Direct Injuries",
        "unit":        "count",
        "color":       "#f87171",
        "description": "Predict injury count from storm severity features",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_damage(val):
    if pd.isna(val) or str(val).strip() in ("", "0"):
        return 0.0
    s = str(val).strip().upper().replace(",", "")
    try:
        if s.endswith("M"):
            return float(s[:-1]) * 1000.0
        if s.endswith("K"):
            return float(s[:-1])
        if s.endswith("B"):
            return float(s[:-1]) * 1_000_000.0
        return float(s) / 1000.0
    except ValueError:
        return 0.0


@st.cache_data(show_spinner="📂 Loading & merging NOAA Storm Events CSVs…")
def load_noaa_csvs() -> pd.DataFrame:
    frames = []
    for fname in CSV_FILES:
        fpath = os.path.join(SCRIPT_DIR, fname)
        if not os.path.exists(fpath):
            st.warning(f"⚠️ File not found: {fname} — skipping.")
            continue
        try:
            df_raw = pd.read_csv(fpath, low_memory=False)
            frames.append(df_raw)
        except Exception as e:
            st.warning(f"⚠️ Could not read {fname}: {e}")

    if not frames:
        st.error("❌ No CSV files could be loaded.")
        st.stop()

    df = pd.concat(frames, ignore_index=True)

    if "YEAR" in df.columns:
        df = df[pd.to_numeric(df["YEAR"], errors="coerce") >= 1996].copy()

    df.columns = [c.strip().upper() for c in df.columns]

    df["DAMAGE_PROPERTY_K"] = df["DAMAGE_PROPERTY"].apply(parse_damage)
    df["DAMAGE_CROPS_K"]    = df["DAMAGE_CROPS"].apply(parse_damage)

    for col in ["BEGIN_LAT", "BEGIN_LON", "MAGNITUDE",
                "INJURIES_DIRECT", "DEATHS_DIRECT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def safe_dt(col):
        try:
            return pd.to_datetime(df[col], format="%d-%b-%y %H:%M:%S", errors="coerce")
        except Exception:
            return pd.to_datetime(df[col], errors="coerce")

    df["_BEGIN_DT"] = safe_dt("BEGIN_DATE_TIME")
    df["_END_DT"]   = safe_dt("END_DATE_TIME")
    df["DURATION_HOURS"] = (
        (df["_END_DT"] - df["_BEGIN_DT"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    required = ["BEGIN_LAT", "BEGIN_LON", TARGET_COL]
    df.dropna(subset=required, inplace=True)

    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    if "_BEGIN_DT" in df.columns:
        df.sort_values("_BEGIN_DT", inplace=True)

    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.title()
    df["PERIOD"]   = df["_BEGIN_DT"].dt.strftime("%b %Y").fillna("Unknown")
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LABEL ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_label_encoder(event_types):
    le = LabelEncoder()
    le.fit(sorted(set(event_types)))
    return le


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STREAM GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class StormStreamGenerator:
    def __init__(self, df: pd.DataFrame, le: LabelEncoder):
        self.df      = df.reset_index(drop=True)
        self.le      = le
        self.pointer = 0
        self.total   = len(df)

    @property
    def exhausted(self):
        return self.pointer >= self.total

    def generate_batch(self, n: int = 100):
        if self.exhausted:
            return None, None, None, None, None
        end   = min(self.pointer + n, self.total)
        chunk = self.df.iloc[self.pointer:end]
        self.pointer = end

        X      = chunk[FEATURE_COLS].values.astype(float)
        y_str  = chunk[TARGET_COL].values
        y_enc  = self.le.transform(y_str)
        period = chunk["PERIOD"].iloc[0]

        # Regression targets dict: {target_col: array}
        y_reg  = {t: chunk[t].values.astype(float) for t in REGRESSION_TARGETS}
        return X, y_enc, y_str, period, y_reg


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DRIFT DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class DriftDetector:
    def __init__(self, window_size=150):
        self.window_size = window_size

    def ks_test(self, ref, cur):
        scores = []
        for i in range(ref.shape[1]):
            stat, _ = stats.ks_2samp(ref[:, i], cur[:, i])
            scores.append(stat)
        return float(np.mean(scores)), scores

    def mmd_rbf(self, ref, cur, gamma=1.0):
        max_s = 200
        if len(ref) > max_s: ref = ref[np.random.choice(len(ref), max_s, replace=False)]
        if len(cur) > max_s: cur = cur[np.random.choice(len(cur), max_s, replace=False)]
        def rbf(X, Y):
            XX = np.sum(X**2, axis=1)[:, None]
            YY = np.sum(Y**2, axis=1)[None, :]
            return np.exp(-gamma * (XX + YY - 2 * X @ Y.T))
        nr, nc = len(ref), len(cur)
        mmd = (np.sum(rbf(ref, ref)) / (nr*nr) +
               np.sum(rbf(cur, cur)) / (nc*nc) -
               2 * np.sum(rbf(ref, cur)) / (nr*nc))
        return max(0.0, float(mmd))

    def mean_change(self, ref, cur):
        diff = np.abs(np.mean(ref, axis=0) - np.mean(cur, axis=0))
        return float(np.mean(diff)), diff

    def aggregate(self, ks, mmd, mc, perf_drop=0.0):
        w = [0.25, 0.25, 0.15, 0.35]
        s = [min(ks*2, 1), min(mmd*5, 1), min(mc*0.5, 1), min(perf_drop*3, 1)]
        prob = sum(wi*si for wi, si in zip(w, s))
        return float(prob), float(prob * 100)

    def predict_drift(self, ks_hist, mmd_hist, acc_hist, err_hist, window=5):
        if len(ks_hist) < window or len(acc_hist) < window:
            return False, 0.0, []
        rks  = ks_hist[-window:]
        rmmd = mmd_hist[-window:]
        racc = acc_hist[-window:]
        re   = err_hist[-window:]
        warnings, ws = [], 0.0
        if np.polyfit(range(window), rks,  1)[0] > 0.02:
            warnings.append("📈 KS score trending up");      ws += 0.30
        if np.polyfit(range(window), rmmd, 1)[0] > 0.01:
            warnings.append("📈 MMD score trending up");     ws += 0.30
        if np.polyfit(range(window), racc, 1)[0] < -0.01:
            warnings.append("⚠️ Accuracy trending down");   ws += 0.25
        if len(re) > 1 and np.polyfit(range(len(re)), re, 1)[0] > 0.01:
            warnings.append("📊 Error rate rising");         ws += 0.25
        if np.std(racc) > 0.05:
            warnings.append("⚡ High accuracy volatility");  ws += 0.20
        return ws > 0.4, float(ws), warnings


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EXPLAINABILITY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class DriftExplainer:
    @staticmethod
    def feature_attribution(ref, cur, names):
        rows = []
        for i, n in enumerate(names):
            stat, pv = stats.ks_2samp(ref[:, i], cur[:, i])
            rows.append({"feature": n, "ks_statistic": stat,
                         "p_value": pv, "drift_score": stat})
        return pd.DataFrame(rows).sort_values("drift_score", ascending=False)

    @staticmethod
    def calculate_psi(ref, cur, bins=10):
        psi_vals = []
        for i in range(ref.shape[1]):
            _, edges = np.histogram(ref[:, i], bins=bins)
            rd, _ = np.histogram(ref[:, i], bins=edges)
            cd, _ = np.histogram(cur[:, i], bins=edges)
            rd = rd / (len(ref) + 1e-10) + 1e-10
            cd = cd / (len(cur) + 1e-10) + 1e-10
            psi_vals.append(float(np.sum((cd - rd) * np.log(cd / rd))))
        return np.array(psi_vals)

    @staticmethod
    def psi_interpretation(val):
        if val < 0.10: return "🟢 Stable"
        if val < 0.20: return "🟡 Moderate shift"
        return               "🔴 Significant shift"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. NOVELTY DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class NoveltyDetector:
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.model  = None
        self.scaler = StandardScaler()

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42, n_estimators=100)
        self.model.fit(Xs)

    def predict(self, X):
        if self.model is None:
            return np.ones(len(X)), np.zeros(len(X))
        Xs     = self.scaler.transform(X)
        labels = self.model.predict(Xs)
        scores = -self.model.score_samples(Xs)
        return labels, scores

    def novelty_rate(self, X):
        labels, scores = self.predict(X)
        return float(np.mean(labels == -1)), scores


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RANDOM FOREST REGRESSOR WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class StormRegressor:
    """
    Wraps a separate RandomForestRegressor for each regression target.
    Input features = FEATURE_COLS minus the target column itself.
    Tracks MAE, RMSE, R² per batch with drift-aware retraining.
    """

    def __init__(self, target_col: str, n_estimators: int = 100):
        self.target_col   = target_col
        self.n_estimators = n_estimators
        self.model        = None
        self.scaler       = StandardScaler()
        self.input_cols   = [c for c in FEATURE_COLS if c != target_col]
        self.input_labels = [FEATURE_DISPLAY[c] for c in self.input_cols]

    def _prepare(self, X_full: np.ndarray, y_reg_dict: dict):
        """
        X_full  : (N, len(FEATURE_COLS)) array — all features
        y_reg_dict: {target_col: array}
        Returns (X_in, y_out) arrays for this regressor's target.
        """
        col_idx = [FEATURE_COLS.index(c) for c in self.input_cols]
        X_in    = X_full[:, col_idx]
        y_out   = y_reg_dict[self.target_col]
        return X_in, y_out

    def fit(self, X_full: np.ndarray, y_reg_dict: dict):
        X_in, y_out = self._prepare(X_full, y_reg_dict)
        Xs = self.scaler.fit_transform(X_in)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=42, n_jobs=-1,
            max_depth=12, min_samples_leaf=5)
        self.model.fit(Xs, y_out)

    def predict(self, X_full: np.ndarray) -> np.ndarray:
        col_idx = [FEATURE_COLS.index(c) for c in self.input_cols]
        X_in    = X_full[:, col_idx]
        Xs      = self.scaler.transform(X_in)
        return self.model.predict(Xs)

    def evaluate(self, X_full: np.ndarray, y_reg_dict: dict) -> dict:
        if self.model is None:
            return {}
        _, y_true = self._prepare(X_full, y_reg_dict)
        y_pred    = self.predict(X_full)
        mae       = float(mean_absolute_error(y_true, y_pred))
        rmse      = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2        = float(r2_score(y_true, y_pred))
        return {"mae": mae, "rmse": rmse, "r2": r2,
                "y_true": y_true, "y_pred": y_pred}

    @property
    def feature_importances(self):
        if self.model is None:
            return None
        return dict(zip(self.input_labels, self.model.feature_importances_))


def train_regressors(X: np.ndarray, y_reg: dict, n_estimators: int) -> dict:
    """Train one StormRegressor per target, return dict keyed by target col."""
    regressors = {}
    for col in REGRESSION_TARGETS:
        r = StormRegressor(col, n_estimators=n_estimators)
        r.fit(X, y_reg)
        regressors[col] = r
    return regressors


# ═══════════════════════════════════════════════════════════════════════════════
# 9. HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

def train_rf(X, y, n_estimators=100):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m  = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42, n_jobs=-1)
    m.fit(Xs, y)
    return m, sc

def predict_storm(model, scaler, X):
    return model.predict(scaler.transform(X))

def clf_metrics(y_true, y_pred):
    acc  = float(accuracy_score(y_true, y_pred))
    f1m  = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    f1w  = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return acc, f1m, f1w


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SESSION STATE DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "data_buffer":           deque(maxlen=5000),
    "batch_count":           0,
    "total_samples":         0,
    "stream_exhausted":      False,
    "stream_generator":      None,
    "label_encoder":         None,
    # classifier
    "model":                 None,
    "model_trained":         False,
    "scaler":                None,
    "reference_window":      None,
    "baseline_accuracy":     None,
    "accuracy_history":      [],
    "f1_macro_history":      [],
    "f1_weighted_history":   [],
    # novelty
    "novelty_detector":      None,
    "novelty_history":       [],
    # drift
    "drift_history":         [],
    "drift_memory":          [],
    "error_rate_history":    [],
    "warning_history":       [],
    # per-class
    "pr_f1_history":         [],
    "class_dist_history":    [],
    # backtest / retrain
    "backtest_results":      [],
    "retrain_log":           [],
    "total_retrain_cost_ms": 0.0,
    # with vs without
    "no_retrain_acc":        [],
    "with_retrain_acc":      [],
    "frozen_model":          None,
    "frozen_scaler":         None,
    # ── NEW: regression ──────────────────────────────────────────────────────
    "regressors":            None,   # dict {target_col: StormRegressor}
    "regressors_trained":    False,
    "reg_history":           [],     # list of dicts per batch
    "reg_ref_window":        None,   # X array used as reg reference
    "reg_retrain_log":       [],     # retrain events for regressors
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# 11. MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🌪️ NOAA Storm Events · Multi-Class Classification, Regression & Concept Drift Detection")
    st.markdown("""
<div class="info-box">
<b>Data:</b> Real NOAA Storm Events CSVs (1950+) — all 3 files merged automatically.<br>
<b>Classifier:</b> Random Forest (multi-class, class_weight=balanced) predicting <code>EVENT_TYPE</code>.<br>
<b>Regressor:</b> Random Forest Regressors predicting <b>Property Damage ($K)</b>, <b>Duration (hrs)</b>, <b>Injuries</b> — each with drift-aware retraining.<br>
<b>Novelty:</b> Isolation Forest unsupervised anomaly layer.<br>
<b>Drift:</b> KS + MMD + MeanChange + PerformanceDrop → adaptive retraining when P(drift) exceeds threshold.
</div>
""", unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_noaa_csvs()
    feat_names = [FEATURE_DISPLAY[f] for f in FEATURE_COLS]

    if st.session_state.label_encoder is None:
        le = build_label_encoder(tuple(df[TARGET_COL].unique()))
        st.session_state.label_encoder = le
    else:
        le = st.session_state.label_encoder

    all_classes  = list(le.classes_)
    n_classes    = len(all_classes)
    class_counts = df[TARGET_COL].value_counts()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown(f"**📦 Total records loaded:** {len(df):,}")
    st.sidebar.markdown(f"**🏷️ Unique EVENT_TYPEs:** {n_classes}")

    st.sidebar.markdown("---")
    window_size       = st.sidebar.slider("Window Size",      100, 500, 200)
    batch_size        = st.sidebar.slider("Batch Size",        50, 200, 100)
    drift_threshold   = st.sidebar.slider("Drift Threshold",  0.10, 0.60, 0.25)
    iso_contamination = st.sidebar.slider("IF Contamination", 0.01, 0.20, 0.05)
    n_estimators      = st.sidebar.slider("RF Trees",          50, 300, 100, step=50)
    auto_mode         = st.sidebar.checkbox("🔄 Auto-Stream", value=False)
    speed             = st.sidebar.slider("Auto-Speed (s)",   0.5, 5.0, 2.0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🌲 Regressor Targets**")
    for col, meta in REGRESSION_TARGETS.items():
        st.sidebar.markdown(f"- {meta['label']}")

    if st.sidebar.button("🔄 Reset System"):
        reset_all()

    # Init stream generator
    if st.session_state.stream_generator is None:
        st.session_state.stream_generator = StormStreamGenerator(df, le)

    generator = st.session_state.stream_generator
    detector  = DriftDetector(window_size=window_size)
    explainer = DriftExplainer()

    # ── Info row ──────────────────────────────────────────────────────────────
    ci1, ci2, ci3, ci4, ci5 = st.columns(5)
    with ci1: st.info(f"📦 **Batches:** {st.session_state.batch_count}")
    with ci2: st.info(f"📊 **Samples:** {st.session_state.total_samples:,}")
    with ci3: st.info(f"🎯 **Drift Events:** {len(st.session_state.drift_memory)}")
    with ci4: st.info(f"🔁 **Retrains:** {len(st.session_state.retrain_log)}")
    with ci5:
        needed = int(window_size * 1.5)
        if st.session_state.stream_exhausted:
            st.error("🏁 Stream Exhausted")
        elif st.session_state.total_samples < needed:
            st.warning(f"⏳ Need {needed - st.session_state.total_samples} more")
        else:
            st.success("✅ Monitoring Active")

    col1, col2, col3, col4 = st.columns(4)

    # ── Next Batch ────────────────────────────────────────────────────────────
    btn_disabled = st.session_state.stream_exhausted
    if (st.sidebar.button("▶️ Next Batch", disabled=btn_disabled) or auto_mode) \
       and not st.session_state.stream_exhausted:

        result = generator.generate_batch(batch_size)
        if result[0] is None:
            st.session_state.stream_exhausted = True
            st.warning("🏁 All NOAA data has been streamed.")
        else:
            X_batch, y_enc_batch, y_str_batch, current_period, y_reg_batch = result
            st.session_state.batch_count   += 1
            st.session_state.total_samples += len(X_batch)

            for i in range(len(X_batch)):
                entry = {
                    "X":       X_batch[i],
                    "y":       y_enc_batch[i],
                    "y_str":   y_str_batch[i],
                    "concept": current_period,
                }
                for t in REGRESSION_TARGETS:
                    entry[f"y_reg_{t}"] = y_reg_batch[t][i]
                st.session_state.data_buffer.append(entry)

            unique_str, counts = np.unique(y_str_batch, return_counts=True)
            st.session_state.class_dist_history.append({
                "batch":   st.session_state.batch_count,
                "concept": current_period,
                **{str(u): int(c) for u, c in zip(unique_str, counts)},
            })

            # ── Initial training ──────────────────────────────────────────────
            if (not st.session_state.model_trained) and \
               len(st.session_state.data_buffer) >= window_size:

                dl   = list(st.session_state.data_buffer)[:window_size]
                X_tr = np.array([d["X"] for d in dl])
                y_tr = np.array([d["y"] for d in dl])

                # --- Classifier ---
                m, sc = train_rf(X_tr, y_tr, n_estimators=n_estimators)
                st.session_state.model            = m
                st.session_state.scaler           = sc
                st.session_state.model_trained    = True
                st.session_state.reference_window = X_tr

                mf, scf = train_rf(X_tr, y_tr, n_estimators=n_estimators)
                st.session_state.frozen_model  = mf
                st.session_state.frozen_scaler = scf

                y_base   = predict_storm(m, sc, X_tr)
                base_acc, _, _ = clf_metrics(y_tr, y_base)
                st.session_state.baseline_accuracy = base_acc

                # --- Novelty ---
                nd = NoveltyDetector(contamination=iso_contamination)
                nd.fit(X_tr)
                st.session_state.novelty_detector = nd

                # --- Regressors ---
                y_reg_win = {t: np.array([d[f"y_reg_{t}"] for d in dl])
                             for t in REGRESSION_TARGETS}
                regressors = train_regressors(X_tr, y_reg_win, n_estimators)
                st.session_state.regressors         = regressors
                st.session_state.regressors_trained = True
                st.session_state.reg_ref_window     = X_tr

                st.success(
                    f"✅ Classifier + Regressors trained on first window ({current_period}) "
                    f"— {n_classes} storm classes — Baseline Accuracy: {base_acc:.1%}"
                )

            # ── Monitoring ────────────────────────────────────────────────────
            if st.session_state.model_trained and \
               len(st.session_state.data_buffer) >= window_size * 1.5:

                dl         = list(st.session_state.data_buffer)
                ref_window = st.session_state.reference_window
                cur_window = np.array([d["X"] for d in dl[-window_size:]])
                X_cur      = np.array([d["X"] for d in dl[-batch_size:]])
                y_cur      = np.array([d["y"] for d in dl[-batch_size:]])

                model = st.session_state.model
                sc    = st.session_state.scaler

                y_pred        = predict_storm(model, sc, X_cur)
                acc, f1m, f1w = clf_metrics(y_cur, y_pred)

                # Frozen model
                frozen_pred = predict_storm(
                    st.session_state.frozen_model,
                    st.session_state.frozen_scaler, X_cur)
                frozen_acc, _, _ = clf_metrics(y_cur, frozen_pred)
                st.session_state.no_retrain_acc.append(frozen_acc)
                st.session_state.with_retrain_acc.append(acc)

                # Novelty
                nd = st.session_state.novelty_detector
                novelty_rate, anomaly_scores = nd.novelty_rate(X_cur)
                st.session_state.novelty_history.append({
                    "batch":        st.session_state.batch_count,
                    "novelty_rate": novelty_rate,
                    "avg_score":    float(np.mean(anomaly_scores)),
                    "concept":      current_period,
                })

                st.session_state.accuracy_history.append(acc)
                st.session_state.f1_macro_history.append(f1m)
                st.session_state.f1_weighted_history.append(f1w)

                baseline_acc = st.session_state.baseline_accuracy or acc
                perf_drop    = max(0, (baseline_acc - acc) / (baseline_acc + 1e-9))
                error_rate   = 1.0 - acc
                st.session_state.error_rate_history.append(error_rate)

                # ── Regression evaluation (current batch) ─────────────────────
                y_reg_cur = {t: np.array([d[f"y_reg_{t}"] for d in dl[-batch_size:]])
                             for t in REGRESSION_TARGETS}
                reg_row = {"batch": st.session_state.batch_count, "concept": current_period}
                if st.session_state.regressors_trained:
                    for t, reg in st.session_state.regressors.items():
                        metrics = reg.evaluate(X_cur, y_reg_cur)
                        reg_row[f"{t}_mae"]  = metrics.get("mae",  0.0)
                        reg_row[f"{t}_rmse"] = metrics.get("rmse", 0.0)
                        reg_row[f"{t}_r2"]   = metrics.get("r2",   0.0)
                st.session_state.reg_history.append(reg_row)

                # Per-class PR/F1
                present_classes = sorted(np.unique(np.concatenate([y_cur, y_pred])))
                if len(present_classes) >= 2:
                    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
                        y_cur, y_pred,
                        labels=present_classes,
                        average=None, zero_division=0)
                    row = {
                        "batch": st.session_state.batch_count,
                        "concept": current_period,
                        "accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w,
                    }
                    for idx, cls_id in enumerate(present_classes):
                        cls_name = le.inverse_transform([cls_id])[0]
                        row[f"prec_{cls_name}"] = float(prec_arr[idx])
                        row[f"rec_{cls_name}"]  = float(rec_arr[idx])
                        row[f"f1_{cls_name}"]   = float(f1_arr[idx])
                    st.session_state.pr_f1_history.append(row)

                # Drift scores
                ks_score, _  = detector.ks_test(ref_window, cur_window)
                mmd_score    = detector.mmd_rbf(ref_window, cur_window)
                mean_ch, _   = detector.mean_change(ref_window, cur_window)
                drift_prob, severity = detector.aggregate(
                    ks_score, mmd_score, mean_ch, perf_drop)

                ks_hist  = [e["ks_score"]  for e in st.session_state.drift_history[-10:]]
                mmd_hist = [e["mmd_score"] for e in st.session_state.drift_history[-10:]]
                drift_predicted, ws, warns = detector.predict_drift(
                    ks_hist + [ks_score], mmd_hist + [mmd_score],
                    st.session_state.accuracy_history,
                    st.session_state.error_rate_history)

                st.session_state.warning_history.append({
                    "batch": st.session_state.batch_count,
                    "predicted": drift_predicted,
                    "warning_score": ws, "warnings": warns,
                })

                # Sidebar
                st.sidebar.markdown("### 🔍 Live Detection")
                st.sidebar.text(f"Batch:    {st.session_state.batch_count}")
                st.sidebar.text(f"Period:   {current_period}")
                st.sidebar.text(f"Accuracy: {acc:.1%}")
                st.sidebar.text(f"F1 Macro: {f1m:.3f}")
                st.sidebar.text(f"Novelty:  {novelty_rate:.1%}")
                st.sidebar.text(f"KS:       {ks_score:.4f}")
                st.sidebar.text(f"MMD:      {mmd_score:.4f}")
                st.sidebar.text(f"Drift P:  {drift_prob:.4f}")
                if drift_predicted:
                    st.sidebar.warning(f"⚠️ DRIFT PREDICTED ({ws:.0%})")
                if drift_prob > drift_threshold:
                    st.sidebar.error("🔴 DRIFT DETECTED")
                else:
                    st.sidebar.success("🟢 Stable")

                drift_event = {
                    "timestamp":       datetime.now(),
                    "batch":           st.session_state.batch_count,
                    "concept":         current_period,
                    "drift_prob":      drift_prob,
                    "severity":        severity,
                    "ks_score":        ks_score,
                    "mmd_score":       mmd_score,
                    "mean_change":     mean_ch,
                    "accuracy":        acc,
                    "f1_macro":        f1m,
                    "f1_weighted":     f1w,
                    "perf_drop":       perf_drop,
                    "novelty_rate":    novelty_rate,
                    "drift_detected":  drift_prob > drift_threshold,
                    "drift_predicted": drift_predicted,
                    "warning_score":   ws,
                }
                st.session_state.drift_history.append(drift_event)

                # ── Adaptive retraining ───────────────────────────────────────
                if drift_prob > drift_threshold:
                    y_cur_win  = np.array([d["y"] for d in dl[-window_size:]])
                    y_reg_win  = {t: np.array([d[f"y_reg_{t}"] for d in dl[-window_size:]])
                                  for t in REGRESSION_TARGETS}

                    test_slice = dl[-(window_size // 4):]
                    X_wf       = np.array([d["X"] for d in test_slice])
                    y_wf       = np.array([d["y"] for d in test_slice])
                    y_reg_wf   = {t: np.array([d[f"y_reg_{t}"] for d in test_slice])
                                  for t in REGRESSION_TARGETS}

                    pre_pred = predict_storm(model, sc, X_wf)
                    pre_acc, pre_f1m, _ = clf_metrics(y_wf, pre_pred)

                    # Pre-retrain regression metrics
                    pre_reg_metrics = {}
                    if st.session_state.regressors_trained:
                        for t, reg in st.session_state.regressors.items():
                            pre_reg_metrics[t] = reg.evaluate(X_wf, y_reg_wf)

                    t0 = time.time()
                    new_m, new_sc = train_rf(cur_window, y_cur_win, n_estimators=n_estimators)
                    # Retrain regressors
                    new_regressors = train_regressors(cur_window, y_reg_win, n_estimators)
                    cost_ms = (time.time() - t0) * 1000

                    post_pred = predict_storm(new_m, new_sc, X_wf)
                    post_acc, post_f1m, _ = clf_metrics(y_wf, post_pred)

                    # Post-retrain regression metrics
                    post_reg_metrics = {}
                    for t, reg in new_regressors.items():
                        post_reg_metrics[t] = reg.evaluate(X_wf, y_reg_wf)

                    new_nd = NoveltyDetector(contamination=iso_contamination)
                    new_nd.fit(cur_window)
                    st.session_state.novelty_detector = new_nd

                    # Regression retrain log
                    reg_retrain_entry = {
                        "retrain_no": len(st.session_state.reg_retrain_log) + 1,
                        "batch":      st.session_state.batch_count,
                        "concept":    current_period,
                        "cost_ms":    cost_ms,
                    }
                    for t in REGRESSION_TARGETS:
                        pre  = pre_reg_metrics.get(t, {})
                        post = post_reg_metrics.get(t, {})
                        reg_retrain_entry[f"{t}_pre_mae"]  = pre.get("mae",  0.0)
                        reg_retrain_entry[f"{t}_post_mae"] = post.get("mae", 0.0)
                        reg_retrain_entry[f"{t}_pre_r2"]   = pre.get("r2",   0.0)
                        reg_retrain_entry[f"{t}_post_r2"]  = post.get("r2",  0.0)
                        reg_retrain_entry[f"{t}_mae_gain"] = (
                            pre.get("mae", 0) - post.get("mae", 0))
                    st.session_state.reg_retrain_log.append(reg_retrain_entry)

                    st.session_state.backtest_results.append({
                        "fold":     len(st.session_state.backtest_results) + 1,
                        "batch":    st.session_state.batch_count,
                        "concept":  current_period,
                        "pre_acc":  pre_acc,
                        "post_acc": post_acc,
                        "acc_gain": post_acc - pre_acc,
                        "pre_f1m":  pre_f1m,
                        "post_f1m": post_f1m,
                        "cost_ms":  cost_ms,
                    })

                    st.session_state.total_retrain_cost_ms += cost_ms
                    st.session_state.retrain_log.append({
                        "retrain_no": len(st.session_state.retrain_log) + 1,
                        "batch":      st.session_state.batch_count,
                        "concept":    current_period,
                        "cost_ms":    cost_ms,
                        "acc_before": pre_acc,
                        "acc_after":  post_acc,
                        "acc_gain":   post_acc - pre_acc,
                        "f1m_before": pre_f1m,
                        "f1m_after":  post_f1m,
                        "drift_prob": drift_prob,
                        "severity":   severity,
                    })

                    st.session_state.model            = new_m
                    st.session_state.scaler           = new_sc
                    st.session_state.regressors       = new_regressors
                    st.session_state.reference_window = cur_window
                    new_base = predict_storm(new_m, new_sc, cur_window)
                    new_acc, _, _ = clf_metrics(y_cur_win, new_base)
                    st.session_state.baseline_accuracy = new_acc
                    st.session_state.drift_memory.append([ks_score, mmd_score, severity])

    # ── Top metrics ───────────────────────────────────────────────────────────
    if st.session_state.drift_history:
        ld = st.session_state.drift_history[-1]
        with col1:
            st.metric("Drift Probability", f"{ld['drift_prob']:.1%}",
                      delta=f"{ld['drift_prob']-drift_threshold:+.1%}" if ld["drift_detected"] else None)
        with col2:
            st.metric("Accuracy", f"{ld['accuracy']:.1%}",
                      delta=f"F1={ld['f1_macro']:.3f}")
        with col3:
            st.metric("KS Statistic", f"{ld['ks_score']:.3f}",
                      delta="Shifted" if ld["ks_score"] > 0.3 else "Stable")
        with col4:
            st.metric("F1 Weighted", f"{ld['f1_weighted']:.3f}",
                      delta="HIGH" if ld["severity"] > 50 else "Low severity")

        if ld["drift_detected"]:
            st.markdown(
                f'<div class="drift-alert">⚠️ DRIFT DETECTED — Batch #{ld["batch"]} '
                f'— {ld["concept"]} — Accuracy: {ld["accuracy"]:.1%} — Model Retrained</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="no-drift">✓ Stable — {ld["concept"]} '
                f'— Accuracy: {ld["accuracy"]:.1%} — F1 Macro: {ld["f1_macro"]:.3f}</div>',
                unsafe_allow_html=True)

        if st.session_state.novelty_history:
            nr = st.session_state.novelty_history[-1]["novelty_rate"]
            if nr > 0.15:
                st.markdown(
                    f'<div class="novelty-alert">🔶 ISOLATION FOREST: '
                    f'{nr:.0%} of current batch flagged as novel storm patterns</div>',
                    unsafe_allow_html=True)

        # Regressor quick summary banner
        if st.session_state.reg_history:
            last_reg = st.session_state.reg_history[-1]
            parts = []
            for t, meta in REGRESSION_TARGETS.items():
                r2  = last_reg.get(f"{t}_r2",  0.0)
                mae = last_reg.get(f"{t}_mae", 0.0)
                parts.append(f"{meta['label']}: R²={r2:.3f} MAE={mae:.1f}{meta['unit']}")
            st.markdown(
                f'<div class="reg-box">🌲 <b>RF Regressors</b> — ' + " &nbsp;|&nbsp; ".join(parts) + "</div>",
                unsafe_allow_html=True)
    else:
        st.info("👈 Click '▶️ Next Batch' in the sidebar or enable Auto-Stream to begin.")

    # ═══════════════════════════════════════════════════════════════════════════
    # DATASET OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════════

    with st.expander("📋 Dataset Overview — NOAA Storm Events", expanded=False):
        ov1, ov2 = st.columns(2)
        with ov1:
            st.markdown(f"**Total records:** {len(df):,}")
            st.markdown(f"**Unique EVENT_TYPEs:** {n_classes}")
            st.markdown(f"**Date range:** {df['PERIOD'].iloc[0]} → {df['PERIOD'].iloc[-1]}")
            st.markdown(f"**Features:** {', '.join(FEATURE_COLS)}")
        with ov2:
            top10 = class_counts.head(10).reset_index()
            top10.columns = ["EVENT_TYPE", "Count"]
            top10["Share"] = (top10["Count"] / len(df) * 100).round(1).astype(str) + "%"
            st.dataframe(top10, use_container_width=True, height=260)

        _dist_df = class_counts.reset_index()
        _dist_df.columns = ["EVENT_TYPE", "Count"]
        fig_dist = px.bar(_dist_df, x="EVENT_TYPE", y="Count",
            title="Full Dataset — Storm Event Type Distribution",
            template="plotly_dark", color="Count",
            color_continuous_scale="Viridis")
        fig_dist.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig_dist, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN TABS
    # ═══════════════════════════════════════════════════════════════════════════

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab_reg = st.tabs([
        "📈 Drift Timeline",
        "🔮 Predictive Warnings",
        "🔬 PSI & Feature Analysis",
        "🌲 Isolation Forest",
        "📊 Classification Performance",
        "🎯 Per-Class Metrics",
        "🧠 Drift Memory",
        "📉 RF Regressors",        # ← NEW TAB
    ])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        if len(st.session_state.drift_history) > 1:
            df_d = pd.DataFrame(st.session_state.drift_history)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_d["batch"], y=df_d["drift_prob"],
                mode="lines+markers", name="Drift Probability",
                line=dict(color="#ff4b4b", width=3), marker=dict(size=8)))
            fig.add_trace(go.Scatter(x=df_d["batch"], y=df_d["severity"]/100,
                mode="lines", name="Severity (norm)",
                line=dict(color="#ffa500", width=2, dash="dash")))
            fig.add_hline(y=drift_threshold, line_dash="dot",
                annotation_text=f"Threshold ({drift_threshold:.0%})",
                line_color="green", line_width=2)
            fig.update_layout(title="Drift Probability Over Time",
                xaxis_title="Batch", yaxis_title="Score",
                hovermode="x unified", height=420, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            ca, cb = st.columns(2)
            with ca:
                fig2 = go.Figure()
                for col_n, color in [("ks_score","cyan"),("mmd_score","orange"),
                                     ("mean_change","yellow"),("perf_drop","red")]:
                    fig2.add_trace(go.Scatter(x=df_d["batch"], y=df_d[col_n],
                        name=col_n, mode="lines+markers", line=dict(color=color)))
                fig2.update_layout(title="Individual Detector Scores",
                    height=350, template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)
            with cb:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df_d["batch"], y=df_d["accuracy"],
                    mode="lines+markers", fill="tozeroy", name="Accuracy",
                    line=dict(color="#00ccff", width=2)))
                dr = df_d[df_d["drift_detected"]]
                fig3.add_trace(go.Scatter(x=dr["batch"], y=dr["accuracy"],
                    mode="markers", name="Retrained",
                    marker=dict(color="red", size=12, symbol="star")))
                fig3.update_layout(title="Accuracy Over Time (★ = Retrain)",
                    yaxis_title="Accuracy", xaxis_title="Batch",
                    height=350, template="plotly_dark")
                st.plotly_chart(fig3, use_container_width=True)

            if len(st.session_state.class_dist_history) > 1:
                st.subheader("🌀 Storm Class Distribution Per Batch")
                cd_df   = pd.DataFrame(st.session_state.class_dist_history).fillna(0)
                cls_cols = [c for c in cd_df.columns if c not in ("batch","concept")]
                fig_cd  = go.Figure()
                colors  = px.colors.qualitative.Plotly
                for i, cls_name in enumerate(cls_cols):
                    fig_cd.add_trace(go.Bar(x=cd_df["batch"], y=cd_df[cls_name],
                        name=cls_name, marker_color=colors[i % len(colors)]))
                fig_cd.update_layout(barmode="stack",
                    title="Storm Type Count per Batch (stacked)",
                    xaxis_title="Batch", yaxis_title="Count",
                    height=400, template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig_cd, use_container_width=True)
        else:
            st.info("Generate batches to see drift timeline.")

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("🔮 Predictive Drift Warnings")
        if len(st.session_state.drift_history) > 5:
            df_d = pd.DataFrame(st.session_state.drift_history)
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=df_d["batch"], y=df_d["warning_score"],
                mode="lines+markers", name="Warning Score",
                line=dict(color="orange", width=3),
                fill="tozeroy", fillcolor="rgba(255,165,0,0.2)"))
            fig_p.add_trace(go.Scatter(x=df_d["batch"], y=df_d["drift_prob"],
                mode="lines+markers", name="Actual Drift",
                line=dict(color="red", width=2)))
            pred = df_d[df_d["drift_predicted"]]
            if len(pred):
                fig_p.add_trace(go.Scatter(x=pred["batch"], y=pred["warning_score"],
                    mode="markers", name="Prediction Alert",
                    marker=dict(color="yellow", size=15, symbol="diamond")))
            fig_p.add_hline(y=0.4, line_dash="dash", line_color="orange",
                annotation_text="Warning Threshold (0.4)")
            fig_p.update_layout(title="Predictive Warnings vs Actual Drift",
                xaxis_title="Batch", yaxis_title="Score",
                height=420, template="plotly_dark")
            st.plotly_chart(fig_p, use_container_width=True)

            wc1, wc2 = st.columns(2)
            with wc1:
                if st.session_state.accuracy_history:
                    fr = go.Figure()
                    fr.add_trace(go.Scatter(y=st.session_state.accuracy_history,
                        mode="lines+markers", name="Accuracy",
                        line=dict(color="cyan", width=2)))
                    fr.update_layout(title="Accuracy Trend",
                        yaxis_title="Accuracy", height=300, template="plotly_dark")
                    st.plotly_chart(fr, use_container_width=True)
            with wc2:
                if st.session_state.error_rate_history:
                    fe = go.Figure()
                    fe.add_trace(go.Scatter(y=st.session_state.error_rate_history,
                        mode="lines+markers", name="Error Rate",
                        line=dict(color="red", width=2), fill="tozeroy"))
                    fe.update_layout(title="Error Rate Trend",
                        height=300, template="plotly_dark")
                    st.plotly_chart(fe, use_container_width=True)
                    if len(st.session_state.error_rate_history) >= 5:
                        trend = np.polyfit(
                            range(5), st.session_state.error_rate_history[-5:], 1)[0]
                        if trend > 0.01:
                            st.error(f"⚠️ Error rate increasing: {trend:.3f}/batch")
                        else:
                            st.success("✓ Error rate stable")
        else:
            st.info("Generate more batches to see predictive analysis.")

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        if st.session_state.model_trained and \
           len(st.session_state.data_buffer) >= window_size * 1.5:
            dl    = list(st.session_state.data_buffer)
            ref_w = st.session_state.reference_window
            cur_w = np.array([d["X"] for d in dl[-window_size:]])

            attr_df = explainer.feature_attribution(ref_w, cur_w, feat_names)
            st.subheader("📌 Per-Feature Drift Attribution (KS Statistic)")
            fig4 = px.bar(attr_df, x="feature", y="drift_score",
                color="drift_score", color_continuous_scale="Reds",
                title="Which features have drifted most?",
                template="plotly_dark")
            fig4.update_layout(height=360)
            st.plotly_chart(fig4, use_container_width=True)

            psi_vals = explainer.calculate_psi(ref_w, cur_w)
            psi_df   = pd.DataFrame({
                "Feature":        feat_names,
                "PSI":            psi_vals,
                "Interpretation": [explainer.psi_interpretation(v) for v in psi_vals],
            })

            st.subheader("📊 Population Stability Index (PSI)")
            cx, cy = st.columns(2)
            with cx:
                fp = go.Figure()
                colors_psi = [
                    "#00cc00" if v < 0.10 else "#ffa500" if v < 0.20 else "#ff4b4b"
                    for v in psi_vals
                ]
                fp.add_trace(go.Bar(x=feat_names, y=psi_vals,
                    marker_color=colors_psi, name="PSI"))
                fp.add_hline(y=0.10, line_dash="dash", line_color="yellow",
                    annotation_text="Moderate (0.10)")
                fp.add_hline(y=0.20, line_dash="dash", line_color="red",
                    annotation_text="Significant (0.20)")
                fp.update_layout(title="PSI per Feature",
                    template="plotly_dark", height=360)
                st.plotly_chart(fp, use_container_width=True)
            with cy:
                st.markdown("**PSI Legend**")
                st.markdown("🟢 PSI < 0.10 — Stable")
                st.markdown("🟡 0.10–0.20  — Moderate shift")
                st.markdown("🔴 PSI > 0.20  — Significant → retrain")
                st.metric("Mean PSI",             f"{float(np.mean(psi_vals)):.3f}")
                st.metric("Max PSI",              f"{float(np.max(psi_vals)):.3f}")
                st.metric("Most Shifted Feature", feat_names[int(np.argmax(psi_vals))])

            st.dataframe(psi_df.round(4), use_container_width=True)

            top_idx = int(np.argmax(psi_vals))
            st.subheader(f"Distribution Overlay — {feat_names[top_idx]} (most shifted)")
            fig_dist2 = go.Figure()
            fig_dist2.add_trace(go.Histogram(x=ref_w[:, top_idx], name="Reference",
                opacity=0.6, marker_color="steelblue", nbinsx=30))
            fig_dist2.add_trace(go.Histogram(x=cur_w[:, top_idx], name="Current",
                opacity=0.6, marker_color="#ff4b4b", nbinsx=30))
            fig_dist2.update_layout(barmode="overlay", template="plotly_dark",
                height=300, title=f"Reference vs Current — {feat_names[top_idx]}")
            st.plotly_chart(fig_dist2, use_container_width=True)

            if st.session_state.model is not None:
                st.subheader("🌲 Random Forest Classifier — Feature Importance (Gini)")
                importances = st.session_state.model.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature":    feat_names,
                    "Importance": importances,
                }).sort_values("Importance", ascending=False)
                fig_fi = px.bar(fi_df, x="Feature", y="Importance",
                    color="Importance", color_continuous_scale="Greens",
                    template="plotly_dark")
                fig_fi.update_layout(height=340)
                st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Need more data for feature analysis.")

    # ── Tab 4 ─────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("🌲 Isolation Forest — Unsupervised Novelty Detection")
        if st.session_state.novelty_history:
            nov_df = pd.DataFrame(st.session_state.novelty_history)
            nv1, nv2, nv3 = st.columns(3)
            with nv1: st.metric("Current Novelty Rate", f"{nov_df['novelty_rate'].iloc[-1]:.1%}")
            with nv2: st.metric("Mean Novelty Rate",    f"{nov_df['novelty_rate'].mean():.1%}")
            with nv3: st.metric("Peak Novelty Rate",    f"{nov_df['novelty_rate'].max():.1%}")

            fn1, fn2 = st.columns(2)
            with fn1:
                fig_nov = go.Figure()
                fig_nov.add_trace(go.Scatter(
                    x=nov_df["batch"], y=nov_df["novelty_rate"],
                    mode="lines+markers", name="Novelty Rate",
                    line=dict(color="#ff8c00", width=3),
                    fill="tozeroy", fillcolor="rgba(255,140,0,0.2)"))
                fig_nov.add_hline(y=0.15, line_dash="dash", line_color="red",
                    annotation_text="High Novelty Alert (15%)")
                fig_nov.update_layout(title="Novelty Rate per Batch",
                    height=340, template="plotly_dark")
                st.plotly_chart(fig_nov, use_container_width=True)
            with fn2:
                fig_as = go.Figure()
                fig_as.add_trace(go.Scatter(
                    x=nov_df["batch"], y=nov_df["avg_score"],
                    mode="lines+markers", name="Avg Anomaly Score",
                    line=dict(color="yellow", width=2)))
                fig_as.update_layout(title="Avg Anomaly Score per Batch",
                    height=340, template="plotly_dark")
                st.plotly_chart(fig_as, use_container_width=True)

            if len(st.session_state.drift_history) > 0:
                df_d   = pd.DataFrame(st.session_state.drift_history)
                merged = pd.merge(nov_df,
                                  df_d[["batch","drift_prob","drift_detected"]],
                                  on="batch", how="inner")
                if len(merged) > 2:
                    fig_cross = go.Figure()
                    fig_cross.add_trace(go.Scatter(
                        x=merged["novelty_rate"], y=merged["drift_prob"],
                        mode="markers",
                        marker=dict(
                            color=merged["drift_detected"].astype(int),
                            colorscale=[[0,"steelblue"],[1,"red"]],
                            size=10, showscale=True,
                            colorbar=dict(title="Drift Detected")),
                        text=merged["concept"]))
                    fig_cross.update_layout(
                        title="Novelty Rate vs Drift Probability",
                        xaxis_title="Novelty Rate",
                        yaxis_title="Drift Probability",
                        height=340, template="plotly_dark")
                    st.plotly_chart(fig_cross, use_container_width=True)
        else:
            st.info("Generate batches to see Isolation Forest analysis.")

    # ── Tab 5 ─────────────────────────────────────────────────────────────────
    with tab5:
        st.subheader("📊 Classification Performance")
        if st.session_state.accuracy_history:
            pc1, pc2, pc3, pc4 = st.columns(4)
            with pc1: st.metric("Current Accuracy",    f"{st.session_state.accuracy_history[-1]:.1%}")
            with pc2: st.metric("Current F1 Macro",    f"{st.session_state.f1_macro_history[-1]:.3f}")
            with pc3: st.metric("Current F1 Weighted", f"{st.session_state.f1_weighted_history[-1]:.3f}")
            with pc4: st.metric("Mean Accuracy",        f"{np.mean(st.session_state.accuracy_history):.1%}")

            ra, rb = st.columns(2)
            with ra:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(y=st.session_state.accuracy_history,
                    mode="lines+markers", name="Accuracy",
                    fill="tozeroy", line=dict(color="#00ccff", width=3)))
                fig_acc.update_layout(title="Accuracy per Batch",
                    yaxis_title="Accuracy", height=300, template="plotly_dark")
                st.plotly_chart(fig_acc, use_container_width=True)
            with rb:
                fig_f1 = go.Figure()
                fig_f1.add_trace(go.Scatter(y=st.session_state.f1_macro_history,
                    mode="lines+markers", name="F1 Macro",
                    fill="tozeroy", line=dict(color="#00ff88", width=3)))
                fig_f1.add_trace(go.Scatter(y=st.session_state.f1_weighted_history,
                    mode="lines", name="F1 Weighted",
                    line=dict(color="#ffd700", width=2, dash="dash")))
                fig_f1.update_layout(title="F1 Score per Batch",
                    height=300, template="plotly_dark")
                st.plotly_chart(fig_f1, use_container_width=True)

            if len(st.session_state.no_retrain_acc) > 1:
                st.subheader("⚡ With vs Without Drift Detection")
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    y=st.session_state.no_retrain_acc,
                    mode="lines", name="Frozen RF (no retraining)",
                    line=dict(color="orange", width=2, dash="dash")))
                fig_comp.add_trace(go.Scatter(
                    y=st.session_state.with_retrain_acc,
                    mode="lines+markers", name="Adaptive RF (with drift detection)",
                    line=dict(color="#00ccff", width=3),
                    fill="tonexty", fillcolor="rgba(0,200,255,0.1)"))
                fig_comp.update_layout(
                    title="Adaptive vs Frozen Model Accuracy",
                    yaxis_title="Accuracy", xaxis_title="Batch",
                    height=380, template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig_comp, use_container_width=True)

                no_m  = np.mean(st.session_state.no_retrain_acc)
                wi_m  = np.mean(st.session_state.with_retrain_acc)
                imp   = ((wi_m - no_m) / (no_m + 1e-9)) * 100
                r1, r2c, r3c = st.columns(3)
                with r1:  st.metric("Avg Acc — Frozen",   f"{no_m:.1%}")
                with r2c: st.metric("Avg Acc — Adaptive", f"{wi_m:.1%}")
                with r3c: st.metric("Improvement",        f"{imp:+.1f}%",
                                    delta="Better" if imp > 0 else "No gain yet")
        else:
            st.info("No performance data yet.")

    # ── Tab 6 ─────────────────────────────────────────────────────────────────
    with tab6:
        st.subheader("🎯 Per EVENT_TYPE — Precision / Recall / F1")
        if st.session_state.pr_f1_history:
            prf_df  = pd.DataFrame(st.session_state.pr_f1_history)
            f1_cols = [c for c in prf_df.columns if c.startswith("f1_") and
                       c not in ("f1_macro","f1_weighted")]
            if f1_cols:
                fig_cls = go.Figure()
                colors  = px.colors.qualitative.Plotly
                for i, col_n in enumerate(f1_cols):
                    label = col_n.replace("f1_", "")
                    fig_cls.add_trace(go.Scatter(
                        x=prf_df["batch"], y=prf_df[col_n],
                        name=label, mode="lines+markers",
                        line=dict(color=colors[i % len(colors)], width=2)))
                fig_cls.update_layout(
                    title="F1 Score per EVENT_TYPE Over Time",
                    xaxis_title="Batch", yaxis_title="F1",
                    yaxis=dict(range=[0, 1.05]),
                    height=450, template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig_cls, use_container_width=True)

            st.subheader("Latest Batch — Per-Class F1")
            latest    = prf_df.iloc[-1]
            f1_latest = {c.replace("f1_",""):latest[c] for c in f1_cols if c in latest}
            if f1_latest:
                fig_f1b = go.Figure()
                fig_f1b.add_trace(go.Bar(
                    x=list(f1_latest.keys()), y=list(f1_latest.values()),
                    marker_color=colors[:len(f1_latest)]))
                fig_f1b.update_layout(
                    title="F1 by EVENT_TYPE — Latest Batch",
                    yaxis=dict(range=[0,1.1]),
                    xaxis_tickangle=-45,
                    height=380, template="plotly_dark")
                st.plotly_chart(fig_f1b, use_container_width=True)

            display_cols = ["batch","concept","accuracy","f1_macro","f1_weighted"]
            st.dataframe(
                prf_df[display_cols].round(3).rename(columns={
                    "batch":"Batch","concept":"Period",
                    "accuracy":"Accuracy","f1_macro":"F1 Macro",
                    "f1_weighted":"F1 Weighted"}),
                use_container_width=True, height=220)
        else:
            st.info("Generate batches to see per-class metrics.")

    # ── Tab 7 ─────────────────────────────────────────────────────────────────
    with tab7:
        st.subheader("🧠 Drift Memory & Recurrence Detection")
        if st.session_state.drift_memory:
            st.success(f"📝 Total Drift Events Recorded: {len(st.session_state.drift_memory)}")
            dm_df = pd.DataFrame(st.session_state.drift_memory,
                                 columns=["KS","MMD","Severity"])
            fig7 = px.scatter_3d(dm_df, x="KS", y="MMD", z="Severity",
                color="Severity", color_continuous_scale="Viridis",
                title="Drift Signature Space (3D)", template="plotly_dark")
            fig7.update_layout(height=480)
            st.plotly_chart(fig7, use_container_width=True)

            if len(st.session_state.drift_memory) > 1:
                latest = np.array(st.session_state.drift_memory[-1])
                sims   = [
                    np.dot(latest, np.array(s)) /
                    (np.linalg.norm(latest) * np.linalg.norm(np.array(s)) + 1e-10)
                    for s in st.session_state.drift_memory[:-1]
                ]
                max_sim = max(sims)
                st.metric("Max Recurrence Similarity", f"{max_sim:.3f}")
                if max_sim > 0.9:
                    st.success("♻️ Recurring drift pattern detected!")
                else:
                    st.info("🆕 Novel drift pattern")
        else:
            st.info("No drift events yet.")

    # ── Tab 8 — RF REGRESSORS (NEW) ───────────────────────────────────────────
    with tab_reg:
        st.subheader("📉 Random Forest Regressors — Continuous Target Prediction")
        st.markdown("""
<div class="reg-box">
Three separate <b>RandomForestRegressor</b> models predict continuous storm impact metrics:
<b>Property Damage ($K)</b>, <b>Storm Duration (hrs)</b>, and <b>Direct Injuries</b>.
Each regressor uses all other features as input, and is retrained together with the classifier
whenever concept drift is detected.
</div>
""", unsafe_allow_html=True)

        if not st.session_state.regressors_trained:
            st.info("Stream enough batches to train the regressors (needs ≥ window_size samples).")
        else:
            # ── Per-regressor metric history ──────────────────────────────────
            if st.session_state.reg_history:
                rh_df = pd.DataFrame(st.session_state.reg_history)

                # Summary metric tiles
                for t, meta in REGRESSION_TARGETS.items():
                    r2_col   = f"{t}_r2"
                    mae_col  = f"{t}_mae"
                    rmse_col = f"{t}_rmse"
                    if r2_col not in rh_df.columns:
                        continue

                    st.markdown(f"### 🎯 {meta['label']}")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(f"R² (latest)",
                                  f"{rh_df[r2_col].iloc[-1]:.4f}",
                                  delta=f"{rh_df[r2_col].iloc[-1] - rh_df[r2_col].iloc[0]:+.4f} vs baseline"
                                  if len(rh_df) > 1 else None)
                    with c2:
                        st.metric(f"MAE (latest)",
                                  f"{rh_df[mae_col].iloc[-1]:.2f} {meta['unit']}")
                    with c3:
                        st.metric(f"RMSE (latest)",
                                  f"{rh_df[rmse_col].iloc[-1]:.2f} {meta['unit']}")

                    fig_r2 = go.Figure()
                    fig_r2.add_trace(go.Scatter(
                        x=rh_df["batch"], y=rh_df[r2_col],
                        mode="lines+markers", name="R²",
                        line=dict(color=meta["color"], width=3),
                        fill="tozeroy", fillcolor=f"rgba(167,139,250,0.15)"))
                    fig_r2.add_hline(y=0, line_dash="dash", line_color="gray",
                        annotation_text="R²=0 baseline")
                    fig_r2.update_layout(
                        title=f"R² Score Over Time — {meta['label']}",
                        xaxis_title="Batch", yaxis_title="R²",
                        height=280, template="plotly_dark")
                    st.plotly_chart(fig_r2, use_container_width=True)

                    f_mae, f_rmse = st.columns(2)
                    with f_mae:
                        fig_mae = go.Figure()
                        fig_mae.add_trace(go.Scatter(
                            x=rh_df["batch"], y=rh_df[mae_col],
                            mode="lines+markers", name="MAE",
                            line=dict(color="#fbbf24", width=2)))
                        fig_mae.update_layout(title=f"MAE — {meta['label']}",
                            yaxis_title=f"MAE ({meta['unit']})",
                            height=240, template="plotly_dark")
                        st.plotly_chart(fig_mae, use_container_width=True)
                    with f_rmse:
                        fig_rmse = go.Figure()
                        fig_rmse.add_trace(go.Scatter(
                            x=rh_df["batch"], y=rh_df[rmse_col],
                            mode="lines+markers", name="RMSE",
                            line=dict(color="#f87171", width=2)))
                        fig_rmse.update_layout(title=f"RMSE — {meta['label']}",
                            yaxis_title=f"RMSE ({meta['unit']})",
                            height=240, template="plotly_dark")
                        st.plotly_chart(fig_rmse, use_container_width=True)

                    st.markdown("---")

            # ── Feature importances per regressor ─────────────────────────────
            st.subheader("🌲 Regressor Feature Importances (Gini)")
            if st.session_state.regressors:
                ri_cols = st.columns(len(REGRESSION_TARGETS))
                for (t, meta), col_ui in zip(REGRESSION_TARGETS.items(), ri_cols):
                    reg = st.session_state.regressors.get(t)
                    if reg is None or reg.model is None:
                        continue
                    fi = reg.feature_importances
                    if fi is None:
                        continue
                    fi_df = pd.DataFrame(
                        {"Feature": list(fi.keys()), "Importance": list(fi.values())}
                    ).sort_values("Importance", ascending=True)
                    with col_ui:
                        fig_fi = go.Figure()
                        fig_fi.add_trace(go.Bar(
                            x=fi_df["Importance"], y=fi_df["Feature"],
                            orientation="h",
                            marker_color=meta["color"]))
                        fig_fi.update_layout(
                            title=f"{meta['label']}",
                            height=320, template="plotly_dark",
                            margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(fig_fi, use_container_width=True)

            # ── Prediction vs Actual scatter (latest window) ───────────────────
            st.subheader("🎯 Predicted vs Actual — Latest Window")
            if st.session_state.regressors and \
               len(st.session_state.data_buffer) >= batch_size:
                dl     = list(st.session_state.data_buffer)
                X_last = np.array([d["X"] for d in dl[-batch_size:]])
                y_reg_last = {t: np.array([d[f"y_reg_{t}"] for d in dl[-batch_size:]])
                              for t in REGRESSION_TARGETS}

                pa_cols = st.columns(len(REGRESSION_TARGETS))
                for (t, meta), col_ui in zip(REGRESSION_TARGETS.items(), pa_cols):
                    reg = st.session_state.regressors.get(t)
                    if reg is None or reg.model is None:
                        continue
                    metrics = reg.evaluate(X_last, y_reg_last)
                    y_true  = metrics.get("y_true", np.array([]))
                    y_pred  = metrics.get("y_pred", np.array([]))
                    if len(y_true) == 0:
                        continue
                    # Clip extreme outliers for display clarity
                    p99 = np.percentile(y_true, 99)
                    mask = y_true <= p99
                    with col_ui:
                        fig_pa = go.Figure()
                        fig_pa.add_trace(go.Scatter(
                            x=y_true[mask], y=y_pred[mask],
                            mode="markers",
                            marker=dict(color=meta["color"], size=5, opacity=0.6),
                            name="Prediction"))
                        lim = max(y_true[mask].max(), y_pred[mask].max()) * 1.05
                        fig_pa.add_trace(go.Scatter(
                            x=[0, lim], y=[0, lim],
                            mode="lines", name="Perfect fit",
                            line=dict(color="white", dash="dash", width=1)))
                        fig_pa.update_layout(
                            title=f"{meta['label']}  R²={metrics.get('r2',0):.3f}",
                            xaxis_title="Actual", yaxis_title="Predicted",
                            height=320, template="plotly_dark",
                            margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(fig_pa, use_container_width=True)

            # ── Regression retrain log ─────────────────────────────────────────
            if st.session_state.reg_retrain_log:
                st.subheader("🔁 Regressor Retrain Log (triggered by drift)")
                rr_df = pd.DataFrame(st.session_state.reg_retrain_log)
                # Show MAE gain per target
                gain_cols = [c for c in rr_df.columns if c.endswith("_mae_gain")]
                gain_labels = {
                    f"{t}_mae_gain": f"MAE Δ {REGRESSION_TARGETS[t]['label']}"
                    for t in REGRESSION_TARGETS if f"{t}_mae_gain" in rr_df.columns
                }
                fig_rr = go.Figure()
                colors_rr = ["#f59e0b", "#a78bfa", "#f87171"]
                for i, (gc, label) in enumerate(gain_labels.items()):
                    fig_rr.add_trace(go.Bar(
                        x=rr_df["retrain_no"], y=rr_df[gc],
                        name=label,
                        marker_color=colors_rr[i % len(colors_rr)]))
                fig_rr.add_hline(y=0, line_dash="dash", line_color="white")
                fig_rr.update_layout(
                    barmode="group",
                    title="MAE Reduction per Regressor Retrain (↑ = better)",
                    xaxis_title="Retrain #", yaxis_title="ΔMAE",
                    height=340, template="plotly_dark")
                st.plotly_chart(fig_rr, use_container_width=True)

                display_log = ["retrain_no", "batch", "concept", "cost_ms"]
                for t in REGRESSION_TARGETS:
                    for suffix in ["_pre_mae", "_post_mae", "_pre_r2", "_post_r2"]:
                        col_name = f"{t}{suffix}"
                        if col_name in rr_df.columns:
                            display_log.append(col_name)
                st.dataframe(
                    rr_df[[c for c in display_log if c in rr_df.columns]].round(3),
                    use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # BACKTESTING
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.header("🧪 Backtesting & Model Validation")

    bm1, bm2, bm3, bm4 = st.columns(4)
    with bm1: st.metric("Walk-Forward Folds", len(st.session_state.backtest_results))
    with bm2:
        if st.session_state.pr_f1_history:
            st.metric("Mean F1 Macro",
                      f"{np.mean([r['f1_macro'] for r in st.session_state.pr_f1_history]):.3f}")
    with bm3:
        if st.session_state.backtest_results:
            st.metric("Avg Accuracy Gain",
                      f"{np.mean([r['acc_gain'] for r in st.session_state.backtest_results]):+.1%}")
    with bm4:
        st.metric("Total Retrain Cost", f"{st.session_state.total_retrain_cost_ms:.1f} ms")

    bTab1, bTab2 = st.tabs(["📊 Walk-Forward Validation", "💰 Retraining Cost & Gain"])

    with bTab1:
        if st.session_state.backtest_results:
            wf_df = pd.DataFrame(st.session_state.backtest_results)
            fig_wf = go.Figure()
            fig_wf.add_trace(go.Bar(x=wf_df["fold"], y=wf_df["pre_acc"],
                name="Accuracy Before Retrain", marker_color="#ff8c00"))
            fig_wf.add_trace(go.Bar(x=wf_df["fold"], y=wf_df["post_acc"],
                name="Accuracy After Retrain",  marker_color="#00cc00"))
            fig_wf.update_layout(barmode="group",
                title="Walk-Forward: Before vs After Retraining",
                xaxis_title="Fold", yaxis_title="Accuracy",
                yaxis=dict(range=[0,1.1]),
                height=380, template="plotly_dark")
            st.plotly_chart(fig_wf, use_container_width=True)

            gain_colors = ["#00cc00" if v >= 0 else "#ff4b4b" for v in wf_df["acc_gain"]]
            fig_gain = go.Figure()
            fig_gain.add_trace(go.Bar(x=wf_df["fold"], y=wf_df["acc_gain"],
                marker_color=gain_colors))
            fig_gain.add_hline(y=0, line_dash="dash", line_color="white")
            fig_gain.update_layout(title="Accuracy Gain per Retrain",
                xaxis_title="Fold", yaxis_title="ΔAccuracy",
                height=300, template="plotly_dark")
            st.plotly_chart(fig_gain, use_container_width=True)

            st.dataframe(
                wf_df[["fold","batch","concept","pre_acc","post_acc",
                        "acc_gain","pre_f1m","post_f1m","cost_ms"]]
                  .round(3)
                  .rename(columns={
                      "fold":"Fold","batch":"Batch","concept":"Period",
                      "pre_acc":"Acc Before","post_acc":"Acc After",
                      "acc_gain":"ΔAcc","pre_f1m":"F1m Before",
                      "post_f1m":"F1m After","cost_ms":"Cost (ms)"}),
                use_container_width=True, height=220)
        else:
            st.info("Walk-forward results appear here after each drift-triggered retrain.")

    with bTab2:
        if st.session_state.retrain_log:
            rl_df = pd.DataFrame(st.session_state.retrain_log)
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1: st.metric("Total Retrains",    len(rl_df))
            with rc2: st.metric("Total Cost",         f"{rl_df['cost_ms'].sum():.1f} ms")
            with rc3: st.metric("Avg Cost / Retrain", f"{rl_df['cost_ms'].mean():.1f} ms")
            with rc4: st.metric("Avg Acc Gain",        f"{rl_df['acc_gain'].mean():+.1%}")

            fig_cost = go.Figure()
            fig_cost.add_trace(go.Bar(x=rl_df["retrain_no"], y=rl_df["cost_ms"],
                marker_color="cyan"))
            fig_cost.update_layout(title="Cost per Retrain (ms)",
                height=300, template="plotly_dark")
            st.plotly_chart(fig_cost, use_container_width=True)

            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(
                x=rl_df["severity"], y=rl_df["acc_gain"],
                mode="markers+text",
                text=[f"R{n}" for n in rl_df["retrain_no"]],
                textposition="top center",
                marker=dict(size=12, color=rl_df["acc_gain"],
                            colorscale="RdYlGn", showscale=True)))
            fig_scat.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_scat.update_layout(
                title="Drift Severity vs Accuracy Gain",
                xaxis_title="Drift Severity", yaxis_title="ΔAccuracy",
                height=360, template="plotly_dark")
            st.plotly_chart(fig_scat, use_container_width=True)

            st.dataframe(
                rl_df[["retrain_no","batch","concept","cost_ms",
                        "acc_before","acc_after","acc_gain",
                        "f1m_before","f1m_after","drift_prob","severity"]]
                  .round(4)
                  .rename(columns={
                      "retrain_no":"#","batch":"Batch","concept":"Period",
                      "cost_ms":"Cost (ms)","acc_before":"Acc Before",
                      "acc_after":"Acc After","acc_gain":"ΔAcc",
                      "f1m_before":"F1m Before","f1m_after":"F1m After",
                      "drift_prob":"Drift P","severity":"Severity"}),
                use_container_width=True)
        else:
            st.info("Retraining log appears after the first drift event.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
**System Status**
- Source: Real NOAA CSVs (1950+)
- Target (clf): EVENT_TYPE
- Targets (reg): Damage, Duration, Injuries
- Classes: {n_classes}
- Model: RF Classifier + 3× RF Regressor
- Window: {window_size}
- Threshold: {drift_threshold:.0%}
- Batches: {st.session_state.batch_count}
- Retrains: {len(st.session_state.retrain_log)}
- Cost: {st.session_state.total_retrain_cost_ms:.1f} ms
""")

    if auto_mode and not st.session_state.stream_exhausted:
        time.sleep(speed)
        st.rerun()


if __name__ == "__main__":
    main()