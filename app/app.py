# ============================================================
# ROADSENSE INDIA — Streamlit Dashboard
# File: app/app.py
# Run with: streamlit run app/app.py
# ============================================================

import time
import sys
import os

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Path fix so imports work from project root ───────────────
sys.path.insert(0, os.path.dirname(__file__))
from iot_simulator import generate_batch, get_alert_color
from ml_model import train_model


# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    **RoadSense India 🚦**

    A data analytics dashboard to explore road accident patterns across India.

    👤 Created by: Akshith  
    🔗 GitHub: https://github.com/PENAKAAKSHITH/RoadSense-India  
    """
)

# Main app
st.title("🚦 RoadSense India Dashboard")

df = pd.read_csv("data/cleaned.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Accidents by State")
state_data = df.groupby('state')['accidents'].sum().sort_values(ascending=False)

st.bar_chart(state_data)

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = 'RoadSense India',
    page_icon   = '🛣️',
    layout      = 'wide',
    initial_sidebar_state = 'expanded'
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #E8593C;
        margin-bottom: 8px;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #E8593C; }
    .metric-label { font-size: 13px; color: #aaa; margin-top: 2px; }
    .section-title {
        font-size: 20px; font-weight: 600;
        border-bottom: 2px solid #E8593C;
        padding-bottom: 6px; margin: 20px 0 14px;
    }
    .alert-box {
        padding: 10px 16px; border-radius: 8px;
        margin-bottom: 6px; font-size: 14px;
    }
    div[data-testid="stSidebarContent"] { background-color: #111827; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned.csv')
    if not os.path.exists(path):
        st.error("❌ cleaned.csv not found. Run the EDA notebook first to generate it.")
        st.stop()
    df = pd.read_csv(path)
    # Ensure numeric
    for col in ['accidents', 'killed', 'injured', 'fatality_rate', 'severity_score', 'year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

@st.cache_resource
def load_model(df):
    model, feature_cols, r2, rmse, fig, coef_fig = train_model(df)
    return model, feature_cols, r2, rmse, fig, coef_fig

df_full = load_data()
all_states = sorted(df_full['state'].unique().tolist())


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png',
             width=80)
    st.markdown("## 🛣️ RoadSense India")
    st.markdown("Real-Time Road Safety & Accident Intelligence")
    st.divider()

    selected_states = st.multiselect(
        '🗺️ Filter by State',
        options=all_states,
        default=all_states[:8],
        help='Select one or more states to filter all charts'
    )

    year_range = st.slider(
        '📅 Year Range',
        min_value=int(df_full['year'].min()),
        max_value=int(df_full['year'].max()),
        value=(int(df_full['year'].min()), int(df_full['year'].max()))
    )
    st.divider()

    st.markdown("### ℹ️ About")
    st.markdown("""
**Project:** RoadSense India  
**Type:** Data Science Internship Project  
**Stack:** Python · Pandas · Plotly · Streamlit · Scikit-learn  

**Data Source:**  
[Road Accidents in India — Kaggle](https://www.kaggle.com/datasets/saurabhshahane/road-accident-india)

**Built by:** *(your name here)*  
**GitHub:** *(your repo link)*
    """)

# ── Apply filters ────────────────────────────────────────────
if not selected_states:
    selected_states = all_states

df = df_full[
    (df_full['state'].isin(selected_states)) &
    (df_full['year'].between(year_range[0], year_range[1]))
].copy()


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("# 🛣️ RoadSense India")
st.markdown("**Real-Time Road Safety & Accident Hotspot Intelligence System**")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '📊 Overview',
    '🗺️ State Analysis',
    '🤖 ML Predictions',
    '📡 IoT Live Feed',
    '📖 Insights'
])


# ════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('🚨 Total Accidents',  f"{int(df['accidents'].sum()):,}")
    with col2:
        st.metric('💀 Total Killed',     f"{int(df['killed'].sum()):,}")
    with col3:
        st.metric('🏥 Total Injured',    f"{int(df['injured'].sum()):,}")
    with col4:
        st.metric('📉 Avg Fatality Rate', f"{df['fatality_rate'].mean():.2f}%")

    st.divider()

    # ── Chart 1: Top 15 States ───────────────────────────────
    st.markdown('<div class="section-title">Top States by Total Accidents</div>',
                unsafe_allow_html=True)

    state_agg = (
        df.groupby('state')
        .agg(total_accidents=('accidents','sum'),
             avg_fatality_rate=('fatality_rate','mean'))
        .reset_index()
        .sort_values('total_accidents', ascending=False)
        .head(15)
    )

    fig1 = px.bar(
        state_agg,
        x='total_accidents', y='state', orientation='h',
        color='avg_fatality_rate', color_continuous_scale='OrRd',
        labels={'total_accidents': 'Total Accidents', 'state': 'State',
                'avg_fatality_rate': 'Fatality Rate (%)'},
        text='total_accidents'
    )
    fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig1.update_layout(yaxis={'categoryorder':'total ascending'}, height=480,
                       margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Year-over-Year Trend ───────────────────────
    st.markdown('<div class="section-title">Year-over-Year Trend</div>',
                unsafe_allow_html=True)

    yearly = (
        df.groupby('year')
        .agg(accidents=('accidents','sum'), killed=('killed','sum'))
        .reset_index().sort_values('year')
    )

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=yearly['year'], y=yearly['accidents'],
                              name='Accidents', mode='lines+markers',
                              line=dict(color='#E8593C', width=2.5)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=yearly['year'], y=yearly['killed'],
                              name='Killed', mode='lines+markers',
                              line=dict(color='#3B8BD4', width=2, dash='dot')), secondary_y=True)
    fig2.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10),
                       legend=dict(orientation='h', y=1.1))
    fig2.update_yaxes(title_text='Total Accidents', secondary_y=False)
    fig2.update_yaxes(title_text='Persons Killed',  secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — STATE ANALYSIS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">State-Level Deep Dive</div>',
                unsafe_allow_html=True)

    # ── Fatality Rate per State ──────────────────────────────
    fat_state = (
        df.groupby('state')['fatality_rate']
        .mean().reset_index()
        .sort_values('fatality_rate', ascending=False)
        .head(20)
    )
    fig3 = px.bar(
        fat_state, x='state', y='fatality_rate',
        color='fatality_rate', color_continuous_scale='Reds',
        title='Average Fatality Rate by State (deaths per 100 accidents)',
        labels={'fatality_rate': 'Fatality Rate (%)', 'state': 'State'},
        text='fatality_rate'
    )
    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig3.update_layout(xaxis_tickangle=-40, height=450,
                       margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Top 5 State Trends ───────────────────────────────────
    st.markdown('<div class="section-title">Accident Trend — Selected States</div>',
                unsafe_allow_html=True)

    top5_in_selection = (
        df.groupby('state')['accidents'].sum()
        .sort_values(ascending=False).head(5).index.tolist()
    )
    state_year = (
        df[df['state'].isin(top5_in_selection)]
        .groupby(['state','year'])['accidents'].sum()
        .reset_index().sort_values('year')
    )
    fig4 = px.line(
        state_year, x='year', y='accidents', color='state',
        markers=True,
        title='Year-over-Year Accident Trend — Top 5 States (from selection)',
        labels={'accidents':'Total Accidents','year':'Year','state':'State'}
    )
    fig4.update_layout(height=400, margin=dict(l=10,r=10,t=40,b=10),
                       legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig4, use_container_width=True)

    # ── Severity Distribution ────────────────────────────────
    st.markdown('<div class="section-title">Severity Score Distribution</div>',
                unsafe_allow_html=True)

    fig5 = px.box(
        df[df['state'].isin(top5_in_selection)],
        x='state', y='severity_score', color='state',
        title='Severity Score Distribution — Top 5 States',
        labels={'severity_score':'Severity Score','state':'State'}
    )
    fig5.update_layout(height=400, showlegend=False,
                       margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig5, use_container_width=True)

    # ── Correlation Heatmap ──────────────────────────────────
    st.markdown('<div class="section-title">Feature Correlation Heatmap</div>',
                unsafe_allow_html=True)

    import seaborn as sns
    import matplotlib.pyplot as plt

    numeric_df = df[['accidents','killed','injured','fatality_rate','severity_score']].dropna()
    corr = numeric_df.corr()

    fig_h, ax = plt.subplots(figsize=(7, 5))
    fig_h.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, vmin=-1, vmax=1, square=True,
                annot_kws={'color':'white'})
    ax.tick_params(colors='white')
    plt.setp(ax.get_xticklabels(), color='white', fontsize=10)
    plt.setp(ax.get_yticklabels(), color='white', fontsize=10)
    ax.set_title('Correlation Heatmap', color='white', fontsize=13, pad=10)
    plt.tight_layout()
    st.pyplot(fig_h)


# ════════════════════════════════════════════════════════════
# TAB 3 — ML PREDICTIONS
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Linear Regression — Accident Prediction</div>',
                unsafe_allow_html=True)

    with st.spinner('Training model on cleaned data...'):
        model, feature_cols, r2, rmse, ml_fig, coef_fig = load_model(df_full)

    col1, col2 = st.columns(2)
    col1.metric('R² Score',  str(r2),       help='Closer to 1.0 = better fit')
    col2.metric('RMSE',      f'{rmse:,.0f}', help='Lower = fewer prediction errors')

    st.plotly_chart(ml_fig,   use_container_width=True)
    st.plotly_chart(coef_fig, use_container_width=True)

    # ── Single Prediction Form ───────────────────────────────
    st.markdown('<div class="section-title">Try a Prediction</div>',
                unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        p_state = st.selectbox('State', all_states)
    with pc2:
        p_year  = st.number_input('Year', min_value=2000, max_value=2030, value=2022)
    with pc3:
        p_killed   = st.number_input('Est. Killed',  min_value=0, value=500)
        p_injured  = st.number_input('Est. Injured', min_value=0, value=2000)

    if st.button('🔮 Predict Accidents', type='primary'):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(df_full['state'].astype(str))
        try:
            state_enc = le.transform([p_state])[0]
        except ValueError:
            state_enc = 0

        row = {
            'state_encoded': state_enc,
            'year'         : p_year,
            'killed'       : p_killed,
            'injured'      : p_injured,
        }
        if 'road_type_encoded' in feature_cols:
            row['road_type_encoded'] = 0

        import pandas as _pd
        X_pred = _pd.DataFrame([row])[feature_cols]
        pred   = max(0, round(model.predict(X_pred)[0]))
        st.success(f"🚗 Predicted accident count: **{pred:,}**")


# ════════════════════════════════════════════════════════════
# TAB 4 — IoT LIVE FEED
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">📡 Live Highway IoT Sensor Feed</div>',
                unsafe_allow_html=True)
    st.caption('Simulated real-time data from highway IoT sensors across India.')

    col_a, col_b = st.columns([1, 3])
    with col_a:
        n_sensors  = st.slider('Sensors per refresh', 5, 30, 10)
        auto_refresh = st.checkbox('Auto-refresh (3s)', value=False)

    refresh_btn = col_b.button('🔄 Refresh Feed', type='primary')

    # Session state for feed history
    if 'iot_history' not in st.session_state:
        st.session_state.iot_history = generate_batch(n_sensors)

    if refresh_btn or auto_refresh:
        new_batch = generate_batch(n_sensors)
        st.session_state.iot_history = pd.concat(
            [new_batch, st.session_state.iot_history]
        ).head(50)

    feed = st.session_state.iot_history

    # ── Live KPIs ────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Active Sensors',    len(feed))
    k2.metric('Avg Speed (km/h)',  f"{feed['speed_kmh'].mean():.1f}")
    k3.metric('Incidents Detected',
              int((feed['incident'] != 'No Incident').sum()))
    k4.metric('Avg Risk Score',    f"{feed['risk_score'].mean():.1f}")

    st.divider()

    # ── Alert cards for critical readings ───────────────────
    critical = feed[feed['risk_level'].str.contains('Critical|High')]
    if not critical.empty:
        st.markdown('#### ⚠️ High-Risk Alerts')
        for _, row in critical.head(5).iterrows():
            color = get_alert_color(row['risk_level'])
            st.markdown(
                f'<div class="alert-box" style="border-left:4px solid {color};'
                f'background:#1a1a2e;">'
                f'<b>{row["risk_level"]}</b> &nbsp;|&nbsp; {row["highway"]} &nbsp;|&nbsp;'
                f'{row["weather"]} &nbsp;|&nbsp; {row["speed_kmh"]} km/h &nbsp;|&nbsp;'
                f'{row["incident"]} &nbsp;|&nbsp; Risk: {row["risk_score"]}</div>',
                unsafe_allow_html=True
            )

    # ── Live Table ───────────────────────────────────────────
    st.markdown('#### All Readings')
    st.dataframe(
        feed[['timestamp','highway','vehicle','weather',
              'speed_kmh','incident','risk_score','risk_level']],
        use_container_width=True,
        height=320
    )

    # ── Charts ───────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        risk_counts = feed['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        fig_r = px.pie(risk_counts, names='Risk Level', values='Count',
                       title='Risk Level Distribution',
                       color_discrete_sequence=['#27AE60','#F39C12','#E67E22','#C0392B'])
        fig_r.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_r, use_container_width=True)
    with c2:
        fig_w = px.histogram(feed, x='weather', color='risk_level',
                             title='Weather Conditions in Feed',
                             color_discrete_sequence=['#27AE60','#F39C12','#E67E22','#C0392B'])
        fig_w.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_w, use_container_width=True)

    if auto_refresh:
        time.sleep(3)
        st.rerun()


# ════════════════════════════════════════════════════════════
# TAB 5 — INSIGHTS
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 📖 Key Findings & Insights")

    st.markdown("""
### 🔴 Finding 1 — Raw counts mislead; fatality rates reveal truth
States like Tamil Nadu and Uttar Pradesh dominate total accident counts,
but smaller north-eastern states (Nagaland, Manipur) have fatality rates
**2–3× the national average**. Policy should prioritise fatality rate, not volume.

---
### 📉 Finding 2 — Accidents peaked in 2016 and are declining
India's road accidents peaked around **2016–2018** and have been on a downward trend.
The sharp 2020 dip is COVID-19 lockdown effect — not a safety improvement.
Post-pandemic recovery shows a partial uptick, suggesting structural interventions are needed.

---
### 🌧️ Finding 3 — Weather is the top IoT risk amplifier
From the live IoT simulation: **Foggy** and **Stormy** conditions raise incident
probability by **5–7×** compared to clear weather. Speed compliance in poor weather
is the single most impactful intervention point.

---
### 🤖 Finding 4 — ML model confirms: injuries predict accidents better than kills
The Linear Regression feature importance shows `injured` has a higher coefficient
than `killed` in predicting total accident count. This suggests accidents with
high injury rates are better leading indicators for hotspot identification.

---
### 🏥 Finding 5 — Emergency response gap in rural states
High fatality rates in low-accident states point to a **trauma care gap**, not
a road design problem. Urban states with more accidents but better hospitals
have proportionally far fewer deaths per accident.
    """)

    st.divider()
    st.markdown("#### 📊 Dataset Quick Stats (current filter)")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric('States in view', df['state'].nunique())
    sc2.metric('Years in view',  f"{int(df['year'].min())}–{int(df['year'].max())}")
    sc3.metric('Rows analysed',  f"{len(df):,}")