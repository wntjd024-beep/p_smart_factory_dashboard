
# build_operational_dashboard_v2.py
# Robust builder with correct CLUSTER handling and cleaner KPIs.
import os, warnings, re
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

def read_any(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p), pd.read_csv(p)
    return None, None

def valid_small_integer_clusters(s):
    """Return True if s looks like small integer cluster labels (e.g., 1..41)."""
    vals = pd.Series(s.dropna().unique())
    if len(vals) < 3 or len(vals) > 80:
        return False
    # near-integer check
    near_int = (np.abs(vals - np.round(vals)) < 1e-6).all()
    return bool(near_int and vals.min() >= 0)

def build_row_cluster(df):
    """
    Determine a row-level 'CLUSTER' column using the best available signal:
    1) If integer-like small-range 'CLUSTER' exists -> normalize to int and use it
    2) If columns like CLUSTER_1..CLUSTER_41 exist -> argmax -> 1..N
    3) Else compute KMeans on X_* subset (200 cols max) -> 6..sqrt(N) clusters
    """
    # 1) Existing usable 'CLUSTER'
    cluster_cols = [c for c in df.columns if c.upper() == "CLUSTER"]
    if cluster_cols and valid_small_integer_clusters(df[cluster_cols[0]]):
        out = df.copy()
        out["CLUSTER"] = (
                            df[cluster_cols[0]]
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(df[cluster_cols[0]].median())
                            .round()
                            .astype(int)
                        )
        return out, "existing_int"

    # 2) Argmax over CLUSTER_*
    ccols = [c for c in df.columns if re.match(r'^CLUSTER_\d+$', c)]
    if len(ccols) >= 3:
        # Take idxmax among those columns -> numeric id
        idx = df[ccols].astype(float).replace([np.inf,-np.inf], np.nan).fillna(0.0).idxmax(axis=1)
        # Extract the number part
        out = df.copy()
        out["CLUSTER"] = idx.str.extract(r'CLUSTER_(\d+)').astype(int)
        return out, "argmax_from_columns"

    # 3) KMeans fallback on X_* features
    if KMeans is not None:
        xcols = [c for c in df.columns if c.startswith("X_")]
        if xcols:
            use = xcols[:min(200, len(xcols))]
            X = df[use].replace([np.inf,-np.inf], np.nan).fillna(df[use].median())
            n = max(6, int(np.sqrt(len(df))))
            n = min(n, 60)
            km = KMeans(n_clusters=n, n_init="auto", random_state=42)
            lab = km.fit_predict(X)
            out = df.copy()
            out["CLUSTER"] = lab + 1  # 1-indexed
            return out, f"kmeans_{n}"
    out = df.copy()
    out["CLUSTER"] = 1
    return out, "single"

def compute_feature_importance(df, top_k=50):
    fi_path = Path("feature_importances_lightgbm.csv")
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        # Normalize header names to ["Feature","Importance"]
        cols_lower = {c.lower(): c for c in fi.columns}
        if "feature" in cols_lower and "importance" in cols_lower:
            fi = fi.rename(columns={cols_lower["feature"]:"Feature", cols_lower["importance"]:"Importance"})
        else:
            # Heuristic: take first 2 columns
            fi = fi.rename(columns={fi.columns[0]:"Feature", fi.columns[1]:"Importance"})
        # Keep only numeric importance
        fi["Importance"] = pd.to_numeric(fi["Importance"], errors="coerce").fillna(0.0)
        # Normalize (0..1) for better visuals
        if fi["Importance"].max() > 0:
            fi["Importance"] = fi["Importance"] / fi["Importance"].max()
        fi = fi.sort_values("Importance", ascending=False).head(top_k).reset_index(drop=True)
        return fi, "provided_normalized"
    # Fallback: empty
    return pd.DataFrame({"Feature":[], "Importance":[]}), "empty"

def build_dashboard_master(df, fi=None, trend_top_n=12):
    # Status buckets are left out (we'll do dynamic thresholds in app)
    # Choose trend columns from FI if provided
    if fi is not None and not fi.empty:
        trend_cols = [f for f in fi["Feature"].tolist() if f in df.columns][:trend_top_n]
    else:
        trend_cols = [c for c in df.columns if c.startswith("X_")][:trend_top_n]

    base_cols = [c for c in ["PRODUCT_ID","TIMESTAMP","LINE","PRODUCT_CODE","CLUSTER","Y_Class","Y_Quality"] if c in df.columns]
    out = df[base_cols + trend_cols].copy()
    # Ensure TIMESTAMP parsed if exists
    if "TIMESTAMP" in out.columns:
        try:
            out["TIMESTAMP"] = pd.to_datetime(out["TIMESTAMP"])
        except Exception:
            pass
    out.to_csv(DATA_DIR / "dashboard_master.csv", index=False)
    return out, trend_cols

def build_cluster_summary(df):
    if "CLUSTER" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("CLUSTER", dropna=False).agg(
        samples=("Y_Quality","count"),
        yq_mean=("Y_Quality","mean"),
        yq_std=("Y_Quality","std"),
        good_ratio=("Y_Quality", lambda s: float((s > s.quantile(0.7)).mean()))  # relative threshold
    ).reset_index()
    grp["samples"] = grp["samples"].astype(int)
    grp = grp.sort_values("yq_mean", ascending=False)
    grp.to_csv(DATA_DIR / "cluster_summary.csv", index=False)
    return grp

def main():
    # Load base
    base_path, df = read_any(["master_merged.csv","train_with_core_features.csv"])
    if df is None:
        raise FileNotFoundError("Place 'master_merged.csv' or 'train_with_core_features.csv' in the working directory.")
    print(f"Loaded {base_path.name} with shape {df.shape}")

    # Build robust row-level CLUSTER
    df2, how = build_row_cluster(df)
    print(f"CLUSTER mode: {how}  | unique={df2['CLUSTER'].nunique()}")

    # Feature importance
    fi, mode = compute_feature_importance(df2, top_k=50)
    print(f"Feature importance: {mode}, rows={len(fi)}")
    (DATA_DIR / "feature_importance.csv").write_text("")  # ensure file exists
    if not fi.empty:
        fi.to_csv(DATA_DIR / "feature_importance.csv", index=False)

    # Master export
    out, trend_cols = build_dashboard_master(df2, fi=fi, trend_top_n=12)
    print(f"dashboard_master.csv saved. trend_cols={trend_cols}")

    # Cluster summary
    cs = build_cluster_summary(df2)
    if not cs.empty:
        print(f"cluster_summary.csv saved. shape={cs.shape}")

    print("Done. Files in ./data")

if __name__ == "__main__":
    main()
