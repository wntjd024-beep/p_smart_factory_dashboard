# app_pro_visual_final.py
# Smart Factory Quality Dashboard — Full Feature + Enhanced UI (No feature change)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from inspect import signature

# Plotly global defaults (일관된 폰트/팔레트/테마)
px.defaults.template = "plotly_dark"      # density heatmap 등 다크배경 유지
px.defaults.color_discrete_sequence = px.colors.qualitative.Set2
px.defaults.width = None
px.defaults.height = 420

def polish_fig(fig, title=None):
    fig.update_layout(
        title=title or fig.layout.title.text,
        font=dict(size=13),
        hoverlabel=dict(bgcolor="rgba(15,23,42,0.9)", font_size=12),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)")
    )
    return fig

# ===== Optional ML/SHAP Imports (graceful fallback) =====
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

# -------------------------------
# Page Config & Theme
# -------------------------------
st.set_page_config(page_title="Smart Factory — Visual Pro Dashboard", page_icon="🏭", layout="wide")
PRIMARY = "#00ADB5"; GOOD = "#10B981"; WARN = "#F59E0B"; BAD = "#EF4444"; GRAY = "#94A3B8"

# -------------------------------
# Global Styles (UI only)
# -------------------------------
st.markdown("""
<style>
/* ---------- Theme Vars: 'Soft Dark + Light Cards' ---------- */
:root{
  --sb-bg:#111827;        /* 이전 #0b1220 보다 밝음 */
  --sb-border:#253046;
  --text-main:#0f172a;
  --muted:#475569;

  --card-bg:#F8FAFC;      /* KPI/표 카드: 거의 흰색 */
  --card-border:#E5E7EB;
  --kpi-label:#64748B;
  --kpi-value:#00ADB5;

  --ai-bg:#0F172A;        /* Insight 카드(다크 유지) */
  --ai-border:#334155;
}

/* 폰트 */
html, body, [class*="css"] { font-family: 'Pretendard', sans-serif; }

/* ===== Sidebar (소프트 다크) ===== */
section[data-testid="stSidebar"]{
  background-color:var(--sb-bg);
  border-right:1px solid var(--sb-border);
}
section[data-testid="stSidebar"] *{ color:#E2E8F0 !important; }
.sidebar-title{
  font-size:18px;font-weight:600;color:var(--kpi-value) !important;margin-bottom:6px;
}

/* ===== Main text/heading (라이트 배경용) ===== */
.big-title{
  font-size:28px;font-weight:700;text-align:center;
  background:linear-gradient(90deg,#00ADB5,#00C7C5);
  -webkit-background-clip:text;color:transparent;margin-bottom:6px;
}
.sub-cap{ text-align:center; color:var(--muted); margin-bottom:10px; }
h1,h2,h3,h4,.section-title{ color:var(--text-main) !important; }
.section-title{
  font-size:22px;font-weight:600;border-left:6px solid var(--kpi-value);
  padding-left:10px;margin-top:24px;margin-bottom:10px;
}

/* ===== KPI 카드 (라이트 카드형) ===== */
.kpi-card{
  background:var(--card-bg);
  border:1px solid var(--card-border);
  border-radius:12px; padding:12px; text-align:center;
  box-shadow:0 1px 2px rgba(15,23,42,0.04);
}
.kpi-label{ font-size:13px; color:var(--kpi-label); }
.kpi-value{ font-size:24px; font-weight:700; color:var(--kpi-value); }

/* ===== Insight 카드 (다크 유지로 시각적 대비) ===== */
.ai-card{
  background:var(--ai-bg);
  border:1px solid var(--ai-border);
  border-radius:10px; padding:14px;
  color:#E2E8F0; font-size:15px; margin-bottom:8px;
}       
            
/* Plotly는 기존 template='plotly_dark'을 써도 본문 가독성엔 영향 없음 */
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ===== Sidebar = KPI 카드와 동일한 라이트 톤 ===== */
/* KPI 카드에서 쓰던 변수와 동일: --card-bg, --card-border, --text-main, --kpi-label, --kpi-value */
section[data-testid="stSidebar"]{
  background: var(--card-bg) !important;     /* = KPI 카드 배경 */
  border-right: 1px solid var(--card-border) !important;
}
/* 사이드바 기본 글자색을 다크 텍스트로 */
section[data-testid="stSidebar"] *{
  color: var(--text-main) !important;
}

/* 섹션 타이틀(좌측 라벨) 톤 통일 */
.sidebar-title{ color: var(--kpi-value) !important; }

/* ====== 사이드바 위젯 가독성 보정 ====== */
/* 입력/셀렉트/멀티셀렉트/파일업로더 등 공통 박스 */
section[data-testid="stSidebar"] .stTextInput > div > div,
section[data-testid="stSidebar"] .stNumberInput > div > div,
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stMultiSelect > div > div,
section[data-testid="stSidebar"] .stDateInput > div > div,
section[data-testid="stSidebar"] .uploadedFile,
section[data-testid="stSidebar"] [data-baseweb="file-uploader"] {
  background: #FFFFFF !important;
  border: 1px solid var(--card-border) !important;
  color: var(--text-main) !important;
  border-radius: 10px !important;
}

/* 멀티셀렉트 선택된 칩 */
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"]{
  background: #E6FFFB !important;            /* 아주 연한 민트 */
  border-color: #B2F5EA !important;
  color: #065F46 !important;
}

/* 숫자 인풋 +/− 버튼, 토글, 체크박스 아이콘 컬러 */
section[data-testid="stSidebar"] svg{
  color: var(--kpi-value) !important;
  fill: var(--kpi-value) !important;
}

/* 파일 업로더 텍스트 대비 향상 */
section[data-testid="stSidebar"] [data-baseweb="file-uploader"] *{
  color: var(--text-main) !important;
}

/* 구분선 톤 다운 */
section[data-testid="stSidebar"] hr { border-color: var(--card-border) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* 본문 최대 폭 제한 + 가운데 정렬 */
.block-container { max-width: 1200px; padding-top: 0.5rem; }

/* 카드/섹션 간 여백 살짝 축소해 밀도 개선 */
.section-title{ margin-top:22px !important; margin-bottom:10px !important; }
.kpi-card{ padding:14px !important; }

/* 축/범례 폰트 기본값 */
svg.main-svg text{ font-family: 'Pretendard', sans-serif !important; }
</style>
""", unsafe_allow_html=True)



# -------------------------------
# Utils (unchanged)
# -------------------------------
def to_datetime_safe(df, col="TIMESTAMP"):
    if col in df.columns:
        try: df[col] = pd.to_datetime(df[col])
        except Exception: pass
    return df

@st.cache_data
def load_csv(path):
    if not Path(path).exists(): return None
    return to_datetime_safe(pd.read_csv(path))

@st.cache_data
def load_best_available_base():
    for p in ["master_merged.csv", "train_with_core_features.csv"]:
        if Path(p).exists():
            return to_datetime_safe(pd.read_csv(p)), p
    return None, None

def kpi(label, value):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def progress_block(y_mean, target):
    prog = 0 if np.isnan(y_mean) else min(max(y_mean/target, 0), 1)
    st.markdown("#### 🎯 Goal Achievement")
    st.progress(prog)
    if not np.isnan(y_mean):
        st.write(f"**현재 평균 품질지수:** {y_mean:.3f} / 목표: {target:.2f} ({prog*100:.1f}% 달성)")

def control_chart(df, ts_col="TIMESTAMP", y_col="Y_Quality",
                  sigma_level=3.0, title=None):

    def polish_fig(fig, title=None):
        fig.update_layout(
            title=title or fig.layout.title.text,
            font=dict(size=13),
            hoverlabel=dict(bgcolor="rgba(15,23,42,0.9)", font_size=12),
            xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)")
        )
        return fig
    """
    Shewhart Control Chart (±kσ). k = sigma_level (e.g., 2, 3, 4, 6 ...)
    """
    s = df.dropna(subset=[y_col]).copy()
    if ts_col in s.columns:
        s = s.sort_values(ts_col); x = s[ts_col]
    else:
        s = s.reset_index(drop=True); x = s.index

    y = s[y_col]
    mu, sigma = y.mean(), y.std(ddof=0)  # 전체 표준편차
    k = float(sigma_level)
    ucl, lcl = mu + k*sigma, mu - k*sigma

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Y_Quality",
                             line=dict(color=PRIMARY)))
    fig.add_trace(go.Scatter(x=x, y=[mu]*len(s), mode="lines", name="Mean",
                             line=dict(dash="dash", color=GRAY)))
    fig.add_trace(go.Scatter(x=x, y=[ucl]*len(s), mode="lines",
                             name=f"+{k:.1f}σ", line=dict(dash="dot", color=BAD)))
    fig.add_trace(go.Scatter(x=x, y=[lcl]*len(s), mode="lines",
                             name=f"-{k:.1f}σ", line=dict(dash="dot", color=BAD)))

    mask = (y>ucl)|(y<lcl)
    if mask.any():
        fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="markers",
                                 name="Out-of-Control",
                                 marker=dict(size=8, color=BAD, symbol="circle")))
    ttl = title or f"Process Stability (±{k:.1f}σ)"
    fig.update_layout(title=ttl, template="plotly_dark",
                      height=420, legend=dict(orientation="h", y=-0.3))

    # 부가정보 반환(표/요약용)
    summary = {
        "mean": mu, "std": sigma, "sigma_level": k,
        "UCL": ucl, "LCL": lcl, "ooc_ratio": float(mask.mean())
    }
    return fig, summary

# -------------------------------
# Data load (unchanged logic)
# -------------------------------
st.sidebar.markdown('<div class="sidebar-title">📂 Data Configuration</div>', unsafe_allow_html=True)
up_dash = st.sidebar.file_uploader("Upload dashboard_master.csv (optional)", type=["csv"])
dashboard = pd.read_csv(up_dash) if up_dash is not None else load_csv("./data/dashboard_master.csv")
if dashboard is None or len(dashboard)==0:
    st.warning("⚠️ './data/dashboard_master.csv'가 없습니다. 먼저 빌더를 실행하세요: python build_operational_dashboard_v2.py")
    st.stop()
dashboard = to_datetime_safe(dashboard)

raw_full, raw_name = load_best_available_base()
if raw_full is None:
    raw_full = dashboard.copy(); raw_name = "dashboard_master.csv (fallback)"

# -------------------------------
# Sidebar filters & target (unchanged logic; UI labels only)
# -------------------------------
lines = sorted(dashboard["LINE"].dropna().unique().tolist()) if "LINE" in dashboard.columns else []
prods = sorted(dashboard["PRODUCT_CODE"].dropna().unique().tolist()) if "PRODUCT_CODE" in dashboard.columns else []
clusters = sorted(dashboard["CLUSTER"].dropna().unique().astype(int).tolist()) if "CLUSTER" in dashboard.columns else []

st.sidebar.markdown('<div class="sidebar-title">🎛 Filters</div>', unsafe_allow_html=True)
sel_lines = st.sidebar.multiselect("🏭 Line", options=lines, default=lines[:2] if lines else [])
sel_prods = st.sidebar.multiselect("🧩 Product Code", options=prods, default=prods[:2] if prods else [])
sel_cluster = st.sidebar.selectbox("🔹 Cluster", options=(["All"]+clusters) if clusters else ["All"], index=0)
st.sidebar.divider()
target_quality = st.sidebar.number_input("🎯 Target Y_Quality", value=0.60, step=0.01, format="%.2f")

if "TIMESTAMP" in dashboard.columns:
    mind, maxd = dashboard["TIMESTAMP"].min(), dashboard["TIMESTAMP"].max()
    start_d, end_d = st.sidebar.date_input("📅 Date Range", value=(mind.date(), maxd.date()))
else:
    start_d = end_d = None

# Apply filters (unchanged)
f = dashboard.copy()
if sel_lines: f = f[f["LINE"].isin(sel_lines)]
if sel_prods: f = f[f["PRODUCT_CODE"].isin(sel_prods)]
if sel_cluster!="All": f = f[f["CLUSTER"]==int(sel_cluster)]
if start_d and end_d and "TIMESTAMP" in f.columns:
    f = f[(f["TIMESTAMP"].dt.date>=start_d) & (f["TIMESTAMP"].dt.date<=end_d)]

# -------------------------------
# Header + KPI (unchanged metrics, styled)
# -------------------------------
st.markdown('<div class="big-title">🏭 Smart Factory — AI Quality Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-cap">AI 기반 품질 예측 + SHAP 해석 + 공정 안정성 모니터링</div>', unsafe_allow_html=True)

y = f["Y_Quality"] if "Y_Quality" in f.columns else pd.Series(dtype=float)
y_mean = float(y.mean()) if len(y) else np.nan
y_std  = float(y.std()) if len(y) else np.nan
q30, q70 = ((float(y.quantile(0.3)), float(y.quantile(0.7))) if len(y) else (np.nan, np.nan))
top30_ratio = float((y>q70).mean()) if len(y) else np.nan

c1,c2,c3,c4 = st.columns(4)
with c1: kpi("Avg. Y_Quality", f"{y_mean:.3f}" if not np.isnan(y_mean) else "N/A")
with c2: kpi("Quality Std", f"{y_std:.3f}" if not np.isnan(y_std) else "N/A")
with c3: kpi("Top30% Ratio", f"{top30_ratio*100:.1f}%" if not np.isnan(top30_ratio) else "N/A")
with c4: kpi("Samples", f"{len(f):,}")
progress_block(y_mean, target_quality)

# -------------------------------
# Tabs (same structure)
# -------------------------------
tabs = st.tabs([
    "📈 Trend / Control",
    "🧠 ML Prediction",
    "🔍 FI + SHAP",
    "📈 PDP",
    "⚙️ Stability",
    "🤖 Insight"
])

# ===== Trend / Control =====
with tabs[0]:
    st.markdown('<div class="section-title">📈 Rolling Trend of Y_Quality</div>', unsafe_allow_html=True)

    if "TIMESTAMP" in f.columns and len(f):
        # 1) 롤링 트렌드
        fs = f.sort_values("TIMESTAMP").copy()
        fs["RollingMean"] = fs["Y_Quality"].rolling(window=10, min_periods=1).mean()

        fig_tr = px.line(
            fs, x="TIMESTAMP", y=["Y_Quality", "RollingMean"],
            labels={"value": "Y_Quality"}, title="Y_Quality & Rolling Mean",
            template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set2
        )
        # 보기 좋게: 원 데이터는 라인+마커 / 롤링 평균은 두껍게
        fig_tr.update_traces(mode="lines+markers", marker=dict(size=3), selector=dict(name="Y_Quality"))
        fig_tr.update_traces(line=dict(width=3), selector=dict(name="RollingMean"))
        st.plotly_chart(fig_tr, use_container_width=True)

        # 2) 가변 시그마 Control Chart
        st.markdown('<div class="section-title">🛡️ Control Chart (±σ 설정)</div>', unsafe_allow_html=True)
        sigma_k = st.slider("Sigma level (±kσ)", min_value=1.0, max_value=6.0, value=3.0, step=0.5,
                            help="관리한계 계산에 사용할 σ 배수(k)")

        # control_chart가 (fig, summary) 또는 fig 단독 반환 둘 다 지원
        res = control_chart(fs, sigma_level=sigma_k)  # <- 함수는 이전에 정의되어 있어야 합니다.
        if isinstance(res, tuple):
            fig_cc, cc_sum = res
        else:
            fig_cc = res
            # 구버전 호환: 요약값 직접 계산
            y = fs["Y_Quality"].dropna()
            mu = y.mean()
            sigma = y.std(ddof=0)
            ucl, lcl = mu + sigma_k * sigma, mu - sigma_k * sigma
            ooc_ratio = float(((y > ucl) | (y < lcl)).mean())
            cc_sum = {"mean": mu, "std": sigma, "sigma_level": sigma_k, "UCL": ucl, "LCL": lcl, "ooc_ratio": ooc_ratio}

        st.plotly_chart(fig_cc, use_container_width=True)

        # 3) 관리한계 요약 KPI
        c_u, c_m, c_l = st.columns(3)
        c_u.metric("UCL", f"{cc_sum['UCL']:.4f}")
        c_m.metric("Mean (μ)", f"{cc_sum['mean']:.4f}")
        c_l.metric("LCL", f"{cc_sum['LCL']:.4f}")
        st.caption(
            f"Out-of-control 비율: {cc_sum['ooc_ratio']*100:.1f}% • σ={cc_sum['std']:.5f} • k={cc_sum['sigma_level']:.1f}"
        )
    else:
        st.info("Y_Quality / TIMESTAMP 데이터가 충분하지 않습니다.")


# ===== ML Prediction =====
with tabs[1]:
    st.markdown('<div class="section-title">🧠 Y_Quality Prediction (LightGBM/RandomForest)</div>', unsafe_allow_html=True)
    feat_cols = [c for c in raw_full.columns if c.startswith("X_")]
    if "Y_Quality" in raw_full.columns and len(feat_cols)>=5:
        rf = raw_full.copy()
        if sel_lines and "LINE" in rf.columns: rf = rf[rf["LINE"].isin(sel_lines)]
        if sel_prods and "PRODUCT_CODE" in rf.columns: rf = rf[rf["PRODUCT_CODE"].isin(sel_prods)]
        if sel_cluster!="All" and "CLUSTER" in rf.columns: rf = rf[rf["CLUSTER"]==int(sel_cluster)]

        X = rf[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(rf[feat_cols].median())
        y_full = rf["Y_Quality"]
        X_train, X_val, y_train, y_val = train_test_split(X, y_full, test_size=0.2, random_state=42)

        model = lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42
        ) if HAS_LGBM else RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=42)

        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        # RMSE 호환 처리 (unchanged logic)
        if "squared" in signature(mean_squared_error).parameters:
            rmse = mean_squared_error(y_val, pred, squared=False)
        else:
            rmse = np.sqrt(mean_squared_error(y_val, pred))
        r2 = r2_score(y_val, pred)

        cA,cB,cC = st.columns(3)
        with cA: kpi("RMSE", f"{rmse:.4f}")
        with cB: kpi("R²", f"{r2:.3f}")
        with cC: kpi("Model", "LightGBM" if HAS_LGBM else "RandomForest")

        comp = pd.DataFrame({"y_true": y_val.values, "y_pred": pred})
        try:
            fig_sc = px.scatter(comp, x="y_true", y="y_pred",
                                title="Validation — y_true vs y_pred", trendline="ols",
                                template="plotly_dark")
        except Exception:
            fig_sc = px.scatter(comp, x="y_true", y="y_pred",
                                title="Validation — y_true vs y_pred",
                                template="plotly_dark")
        fig_sc.update_layout(height=420)
        st.plotly_chart(fig_sc, use_container_width=True)

        # Native feature importance (Top 30)
        try:
            fi = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
            fi = fi.sort_values("Importance", ascending=False).head(30)
            fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                            title="Model Feature Importance (Top 30)",
                            template="plotly_dark", color="Importance",
                            color_continuous_scale="Blues")
            fig_fi.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception:
            fi = None
            st.info("모델 중요도를 계산할 수 없습니다.")
    else:
        st.info("모델 학습에 충분한 X_* 피처가 없습니다. (원본 CSV를 루트에 두면 자동 사용)")

st.session_state["model"] = model
st.session_state["X"] = X
st.session_state["fi"] = fi if 'fi' in locals() else None

# ===== FI + SHAP =====
with tabs[2]:
    st.markdown('<div class="section-title">🔍 Feature Importance & SHAP Summary</div>', unsafe_allow_html=True)

    # 세션에서 안전하게 가져오기 (탭 이동해도 유지)
    model = st.session_state.get("model", None)
    X = st.session_state.get("X", None)
    fi = st.session_state.get("fi", None)

    # --- Feature Importance (표/차트) ---
    if fi is not None and not fi.empty:
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            fig_imp = px.bar(
                fi.head(10), x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues_r",
                title="Top 10 Important Features", template="plotly_dark"
            )
            fig_imp.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_imp, use_container_width=True)
        with col2:
            st.dataframe(fi.head(10), use_container_width=True, height=320)
    else:
        st.info("먼저 'ML Prediction' 탭에서 모델을 학습하세요.")

    # --- SHAP Summary Plot ---
    st.markdown("#### 📈 SHAP Summary Plot (Top 10)")
    if not HAS_SHAP:
        st.info("SHAP 라이브러리가 필요합니다. `pip install shap` 후 다시 실행하세요.")
    elif model is None or X is None:
        st.info("학습된 모델 또는 입력 피처(X)가 없습니다. 'ML Prediction' 탭을 먼저 실행하세요.")
    else:
        try:
            # 계산량 절약을 위해 샘플링
            Xs = X.sample(min(800, len(X)), random_state=42) if len(X) > 800 else X

            # 트리 기반 모델에 최적화된 Explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Xs)

            # --- 회귀/다중출력 대응: shap_values가 list인 경우 첫 클래스 사용 (필요 시 abs-sum으로 변경 가능) ---
            if isinstance(shap_values, list):
                # 회귀라면 보통 배열, 다중 클래스면 list. 여기서는 대표로 첫 항목 사용.
                shap_values_use = shap_values[0]
            else:
                shap_values_use = shap_values

            import matplotlib.pyplot as plt
            plt.close("all")  # 깨끗한 상태

            # 신/구 버전 모두 호환: summary_plot은 내부에서 figure를 만들 수 있으므로 gcf로 받기
            shap.summary_plot(shap_values_use, Xs, plot_type="dot", show=False, max_display=10)
            fig = plt.gcf()  # 방금 그려진 현재 figure를 가져옴
            st.pyplot(fig, bbox_inches='tight')
            plt.close(fig)

        except Exception as e1:
            # 새 API (beeswarm) 시도
            try:
                import matplotlib.pyplot as plt
                plt.close("all")
                fig2, ax2 = plt.subplots()
                # shap.plots.beeswarm는 Explanation 형태를 선호하므로 변환 시도
                # 최신 shap는 shap.Explanation 사용, 구버전은 바로 가능
                if hasattr(shap, "plots") and hasattr(shap.plots, "beeswarm"):
                    # 가능하면 Explanation로 감싸기 (실패해도 beeswarm은 종종 ndarray로 작동)
                    try:
                        from shap import Explanation
                        exp = Explanation(values=shap_values_use, data=Xs.values,
                                          feature_names=Xs.columns)
                        shap.plots.beeswarm(exp, show=False, max_display=10)
                    except Exception:
                        shap.plots.beeswarm(shap_values_use, show=False, max_display=10)
                    st.pyplot(plt.gcf(), bbox_inches='tight')
                    plt.close("all")
                else:
                    raise e1
            except Exception as e2:
                st.warning(f"SHAP 시각화 중 문제가 발생했습니다: {e2}")


# ===== PDP =====
with tabs[3]:
    st.markdown('<div class="section-title">📈 Partial Dependence Plot (Top 3 features)</div>', unsafe_allow_html=True)
    if 'fi' in locals() and fi is not None and not fi.empty and 'model' in locals():
        top3 = fi["Feature"].head(3).tolist()
        for fcol in top3:
            try:
                figp, axp = plt.subplots()
                PartialDependenceDisplay.from_estimator(model, X, [fcol], ax=axp)
                st.pyplot(figp); plt.close(figp)
            except Exception as e:
                st.warning(f"{fcol} PDP 생성 중 오류: {e}")
    else:
        st.info("모델 및 중요도 정보가 필요합니다.")

# ========= Helper: NaN 셀 회색 패치 =========
def _add_nan_patches(fig, pivot, nan_fill="#ECEFF4", nan_edge="#CBD5E1"):
    """imshow 결과 위에 NaN 셀을 연회색 패치로 덮어 결측을 명확히 보이게."""
    rows, cols = pivot.shape
    xs = list(range(cols))
    ys = list(range(rows))
    for r in range(rows):
        for c in range(cols):
            if not np.isfinite(pivot.iat[r, c]):  # NaN/inf
                fig.add_shape(
                    type="rect",
                    x0=c-0.5, x1=c+0.5, y0=r-0.5, y1=r+0.5,
                    xref="x", yref="y",
                    line=dict(color=nan_edge, width=1, dash="dot"),
                    fillcolor=nan_fill, layer="above"
                )
    return fig

# ===== Stability — Clean Default + Advanced Options =====
with tabs[4]:
    st.markdown('<div class="section-title">⚙️ Process Stability & Heatmap</div>', unsafe_allow_html=True)

    if all(col in f.columns for col in ["LINE", "CLUSTER", "Y_Quality"]):
        # 1) 집계 + CV%
        stab = (f.groupby(["LINE", "CLUSTER"])["Y_Quality"]
                  .agg(["mean", "std", "count"]).reset_index())
        stab["cv_percent"] = (stab["std"] / stab["mean"]) * 100

        # 2) IQR 클리핑(극단값 영향 완화)
        q1, q3 = stab["cv_percent"].quantile([0.25, 0.75])
        iqr = max(q3 - q1, 1e-9)
        low_clip = max(q1 - 1.5 * iqr, 0)
        high_clip = q3 + 1.5 * iqr
        stab["cv_clip"] = stab["cv_percent"].clip(lower=low_clip, upper=high_clip)

        # 3) 피벗(정규화 없음, 절대 CV%)
        pivot = (stab
                 .pivot_table(index="CLUSTER", columns="LINE", values="cv_clip", aggfunc="mean")
                 .sort_index(ascending=True)
                 .sort_index(axis=1, ascending=True))

        # ---- Soft heatmap (always with labels) ----
        zmin = float(np.nanmin(pivot.values)); zmax = float(np.nanmax(pivot.values))
        rng = max(zmax - zmin, 1e-9)

        # 부드러운 저채도 팔레트 (초록-중립-레드, 파스텔)
        soft_scale = [
            [0.00, "#B7E4C7"],   # very light green
            [0.50, "#F5F5F5"],   # near white (중립)
            [1.00, "#F5B7B1"]    # very light red
        ]

        fig_hm = px.imshow(
            pivot,
            origin="lower", aspect="auto",
            zmin=zmin, zmax=zmax,
            color_continuous_scale=soft_scale,
            title="Stability Heatmap — CV% (clipped, soft)",
            template="simple_white",             # 라이트 배경으로 전환
            labels=dict(color="CV%")
        )

        fig_hm.update_layout(
            xaxis_title="LINE", yaxis_title="CLUSTER",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
            coloraxis_colorbar=dict(title="%", ticksuffix="%", len=0.85)
        )
        fig_hm.update_traces(
            hovertemplate="LINE=%{x}<br>CLUSTER=%{y}<br>CV=%{z:.1f}%<extra></extra>"
        )

        # 라벨: 항상 표시(부드러운 투명도, 배경 대비 자동)
        for (r, c), val in np.ndenumerate(pivot.values):
            if np.isfinite(val):
                norm = (val - zmin)/rng
                txt = f"{val:.1f}"
                # 배경이 진하면 흰색, 밝으면 검정 + 약간 투명
                color = "rgba(255,255,255,0.9)" if (norm >= 0.70 or norm <= 0.30) else "rgba(0,0,0,0.75)"
                fig_hm.add_annotation(
                    x=pivot.columns[c], y=pivot.index[r],
                    text=txt, showarrow=False,
                    font=dict(color=color, size=12),
                    xanchor="center", yanchor="middle"
                )

        st.plotly_chart(fig_hm, use_container_width=True)


        # ---- Advanced: 필요할 때만 켜는 옵션들 ----
        with st.expander("Advanced options (optional)"):
            c1, c2 = st.columns(2)
            use_threshold_bins = c1.toggle("임계값 구간색(8/12%)", value=False)
            show_cell_text     = c2.toggle("셀 값 라벨 표시", value=False)

            if use_threshold_bins or show_cell_text:
                # 구간색(선택) 적용
                if use_threshold_bins:
                    bins = [0, 8, 10, 12, 15, 1000]
                    colors = ["#2e7d32", "#8bc34a", "#fff176", "#ff9800", "#d32f2f"]
                    Z = pivot.copy()
                    Z[:] = np.digitize(pivot.values, bins, right=False) - 1
                    fig_adv = px.imshow(
                        Z, origin="lower", aspect="auto",
                        color_continuous_scale=[
                            [i/(len(colors)-1), c] for i, c in enumerate(colors)
                        ],
                        zmin=0, zmax=len(colors)-1,
                        title="Stability Heatmap — CV% (threshold bins)",
                        template="plotly_dark"
                    )
                    tickvals = list(range(len(colors)))
                    ticktext = ["<8", "8–10", "10–12", "12–15", "≥15"]
                    fig_adv.update_layout(
                        xaxis_title="LINE", yaxis_title="CLUSTER",
                        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
                        coloraxis_colorbar=dict(title="CV%", tickvals=tickvals, ticktext=ticktext, len=0.85)
                    )
                    # 라벨은 원 값 기준으로 찍기 위해 밑에서 pivot 사용
                    base_fig = fig_adv
                else:
                    # 연속색 유지 + 라벨만
                    base_fig = px.imshow(
                        pivot, origin="lower", aspect="auto",
                        zmin=zmin, zmax=zmax,
                        color_continuous_scale=[[0.00,"#1a9850"],[0.50,"#fee08b"],[1.00,"#d73027"]],
                        title="Stability Heatmap — CV% (clipped)",
                        template="plotly_dark",
                        labels=dict(color="CV%")
                    )
                    base_fig.update_layout(
                        xaxis_title="LINE", yaxis_title="CLUSTER",
                        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
                        coloraxis_colorbar=dict(title="%", ticksuffix="%", len=0.85)
                    )

                # 라벨(선택): NaN/inf 숨기고, 극단부만 흰 글자
                if show_cell_text:
                    z = pivot.values
                    rng = max(zmax - zmin, 1e-9)
                    for (r, c), val in np.ndenumerate(z):
                        if np.isfinite(val):
                            norm = (val - zmin)/rng
                            txt = f"{val:.0f}"
                            color = "white" if (norm >= 0.70 or norm <= 0.30) else "black"
                            base_fig.add_annotation(
                                x=pivot.columns[c], y=pivot.index[r],
                                text=txt, showarrow=False,
                                font=dict(color=color, size=11),
                                xanchor="center", yanchor="middle"
                            )

                st.plotly_chart(base_fig, use_container_width=True)

        # 요약 표
        st.markdown("#### 🔎 Top Unstable Segments (by CV%)")
        topN = (stab.sort_values("cv_percent", ascending=False)
                     .loc[:, ["LINE","CLUSTER","cv_percent","count"]]
                     .head(8)
                     .rename(columns={"cv_percent":"CV(%)"}))
        st.dataframe(topN, use_container_width=True, height=240)

        # 라인별 롤링 (그대로)
        st.markdown('<div class="section-title">📉 Rolling Trend by Line (Top 3)</div>', unsafe_allow_html=True)
        for line in topN["LINE"].unique()[:3]:
            sub = f[f["LINE"]==line].sort_values("TIMESTAMP")
            if len(sub) > 30:
                sub["rolling_mean"] = sub["Y_Quality"].rolling(window=10).mean()
                fig_line = px.line(
                    sub, x="TIMESTAMP", y="rolling_mean",
                    title=f"Rolling Mean — {line}",
                    color_discrete_sequence=["#2E86DE"], template="plotly_dark"
                )
                st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("LINE / CLUSTER / Y_Quality 컬럼이 필요합니다.")

# ===== Insight =====
with tabs[5]:
    st.markdown('<div class="section-title">🤖 AI Insight 2.0 — 자동 요약</div>', unsafe_allow_html=True)
    def insight(df_):
        out=[]
        if len(df_)==0 or "Y_Quality" not in df_.columns: return ["데이터가 부족합니다."]
        m, s = df_["Y_Quality"].mean(), df_["Y_Quality"].std()
        out.append(f"📊 평균 품질지수 **{m:.3f}**, 표준편차 **{s:.3f}**.")
        if "CLUSTER" in df_.columns and len(df_["CLUSTER"].unique())>1:
            g = df_.groupby("CLUSTER")["Y_Quality"].mean()
            hi, lo = int(g.idxmax()), int(g.idxmin()); gap = g.max()-g.min()
            out.append(f"📈 최고 클러스터 **#{hi}({g.max():.3f})**, 최저 **#{lo}({g.min():.3f})**, 격차 **{gap:.3f}**.")
        if m < 0.52: out.append("⚠️ 전반적 품질이 낮습니다. 핵심 센서 변동/이상 신호 점검 필요.")
        elif m < 0.55: out.append("🔍 보통 수준입니다. 변동성 관리와 상위 변수 튜닝이 필요합니다.")
        else: out.append("✅ 목표 수준 이상으로 안정적입니다.")
        if 'fi' in locals() and fi is not None and not fi.empty:
            out.append(f"🏭 핵심 변수는 **{fi.iloc[0]['Feature']}** 입니다. 민감도(PDP/SHAP)를 참고해 제어 구간을 설정하세요.")
        return out
    for line in insight(f):
        st.markdown(f'<div class="ai-card">{line}</div>', unsafe_allow_html=True)

# Footer
st.caption(f"Model base: {raw_name} • Dashboard data: ./data/dashboard_master.csv")
