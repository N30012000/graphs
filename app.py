import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import anthropic
import json
from datetime import datetime

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafetyOS Executive Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #09090b;
    color: #fafafa;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #18181b !important;
    border-right: 1px solid #27272a;
}
[data-testid="stSidebar"] * {
    color: #fafafa !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Metric cards */
.metric-card {
    background: #1c1c1f;
    border: 1px solid #27272a;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 4px;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #a1a1aa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Sora', sans-serif;
    font-size: 26px;
    font-weight: 800;
}
.metric-sub {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #a1a1aa;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #a1a1aa;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 12px 0 8px 0;
    padding-left: 4px;
}

/* AI insight box */
.ai-insight {
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(129,140,248,0.08));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 16px 18px;
    margin: 8px 0;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #c7d2fe;
    line-height: 1.7;
}
.ai-insight-header {
    font-family: 'Sora', sans-serif;
    font-size: 12px;
    font-weight: 700;
    color: #818cf8;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Streamlit button override */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    padding: 8px 20px !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
}

/* Text inputs */
.stNumberInput input, .stTextInput input, .stTextArea textarea {
    background: #09090b !important;
    border: 1px solid #27272a !important;
    color: #fafafa !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #1c1c1f !important;
    border: 1px solid #27272a !important;
    border-radius: 10px !important;
    color: #fafafa !important;
    font-family: 'Sora', sans-serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #18181b;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #a1a1aa !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #fafafa !important;
}

/* Divider */
hr { border-color: #27272a !important; }

/* Success/info/warning */
.stSuccess { background: rgba(34,197,94,0.1) !important; border: 1px solid rgba(34,197,94,0.3) !important; }
.stInfo { background: rgba(99,102,241,0.1) !important; border: 1px solid rgba(99,102,241,0.3) !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border: 1px solid rgba(245,158,11,0.3) !important; }

/* Chat messages */
.stChatMessage {
    background: #1c1c1f !important;
    border: 1px solid #27272a !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE DEFAULTS ────────────────────────────────────────────────────
def init_state():
    defaults = {
        "data": {
            "jan": {"volume": 42, "highSev": 8, "medSev": 14, "lowSev": 20, "auditRate": 61, "reactive": 38, "proactive": 14},
            "feb": {"volume": 57, "highSev": 6, "medSev": 16, "lowSev": 22, "auditRate": 74, "reactive": 35, "proactive": 22},
            "mar": {"volume": 71, "highSev": 4, "medSev": 17, "lowSev": 28, "auditRate": 83, "reactive": 30, "proactive": 31},
        },
        "chat_history": [],
        "last_ai_summary": None,
        "api_key": "",
        "page": "Overview",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── COLORS ───────────────────────────────────────────────────────────────────
C = {
    "bg": "#09090b", "surface": "#18181b", "card": "#1c1c1f", "border": "#27272a",
    "text": "#fafafa", "muted": "#a1a1aa", "accent": "#6366f1", "accent_light": "#818cf8",
    "green": "#22c55e", "red": "#ef4444", "amber": "#f59e0b", "cyan": "#06b6d4",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color=C["muted"], size=10),
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=C["muted"])),
    xaxis=dict(gridcolor=C["border"], linecolor="rgba(0,0,0,0)", tickcolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor=C["border"], linecolor="rgba(0,0,0,0)", tickcolor="rgba(0,0,0,0)"),
)

# ─── LINEAR REGRESSION FORECAST ───────────────────────────────────────────────
def forecast(vals, n=3):
    x = np.arange(len(vals))
    slope, intercept = np.polyfit(x, vals, 1)
    return [max(0, round(intercept + slope * (len(vals) + i))) for i in range(n)]

def get_data_summary():
    d = st.session_state.data
    months = ["jan", "feb", "mar"]
    return {
        "reporting_volume": {m: d[m]["volume"] for m in months},
        "high_severity": {m: d[m]["highSev"] for m in months},
        "medium_severity": {m: d[m]["medSev"] for m in months},
        "low_severity": {m: d[m]["lowSev"] for m in months},
        "audit_closure_rate": {m: d[m]["auditRate"] for m in months},
        "reactive_reports": {m: d[m]["reactive"] for m in months},
        "proactive_hazard_ids": {m: d[m]["proactive"] for m in months},
        "forecasts": {
            "volume_apr_jun": forecast([d[m]["volume"] for m in months]),
            "high_sev_apr_jun": forecast([d[m]["highSev"] for m in months]),
            "audit_rate_apr_jun": forecast([d[m]["auditRate"] for m in months]),
            "proactive_apr_jun": forecast([d[m]["proactive"] for m in months]),
        }
    }

# ─── ANTHROPIC AI ─────────────────────────────────────────────────────────────
def get_ai_client():
    key = st.session_state.get("api_key", "")
    if key:
        return anthropic.Anthropic(api_key=key)
    try:
        return anthropic.Anthropic()
    except Exception:
        return None

def ai_generate_summary():
    client = get_ai_client()
    if not client:
        return "⚠️ No API key configured. Add your Anthropic API key in the sidebar settings."
    
    summary = get_data_summary()
    prompt = f"""You are a Senior Safety Performance Analyst providing an executive briefing.

Here is the Q1 2025 safety data:
{json.dumps(summary, indent=2)}

Write a concise executive summary (4-6 sentences) covering:
1. Overall safety culture trajectory
2. Key wins and concerning trends
3. Top 2 recommended actions for Q2
4. One forward-looking prediction based on the forecast data

Use professional but direct language. Include specific numbers. No bullet points — flowing paragraphs only."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

def ai_chat_response(user_message):
    client = get_ai_client()
    if not client:
        return "⚠️ No API key configured. Add your Anthropic API key in the sidebar settings."
    
    summary = get_data_summary()
    system_prompt = f"""You are SafetyOS AI, an expert safety performance analyst embedded in an executive dashboard.

Current Q1 2025 data context:
{json.dumps(summary, indent=2)}

Answer questions about this safety data with:
- Specific numbers and percentages
- Trend analysis and context
- Actionable recommendations
- Professional but conversational tone
- Keep responses concise (2-4 sentences unless a detailed breakdown is requested)"""

    messages = []
    for msg in st.session_state.chat_history[-8:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

def ai_predict_metric(metric_name, values):
    client = get_ai_client()
    if not client:
        return "Configure API key for AI insights."
    
    pred = forecast(values)
    prompt = f"""For the safety metric '{metric_name}' with Q1 values Jan={values[0]}, Feb={values[1]}, Mar={values[2]}, 
and forecast Apr={pred[0]}, May={pred[1]}, Jun={pred[2]}: 
Write ONE sentence of GM-level insight about this trend and what it means for Q2. Be specific and direct."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception:
        return None

# ─── CHART BUILDERS ───────────────────────────────────────────────────────────
def chart_volume():
    d = st.session_state.data
    months = ["Jan", "Feb", "Mar"]
    vals = [d["jan"]["volume"], d["feb"]["volume"], d["mar"]["volume"]]
    pred_months = ["Apr", "May", "Jun"]
    preds = forecast(vals)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=vals, mode="lines+markers", name="Actual",
        line=dict(color=C["accent"], width=3),
        marker=dict(size=8, color=C["accent"]),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.12)",
        hovertemplate="<b>%{x}</b><br>Reports: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=["Mar"] + pred_months, y=[vals[-1]] + preds,
        mode="lines+markers", name="Forecast",
        line=dict(color=C["accent_light"], width=2, dash="dot"),
        marker=dict(size=6, color=C["accent_light"], symbol="circle-open"),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y}<extra></extra>"
    ))
    fig.update_layout(**PLOT_LAYOUT, height=230,
        xaxis=dict(**PLOT_LAYOUT["xaxis"], showgrid=True),
        yaxis=dict(**PLOT_LAYOUT["yaxis"], showgrid=True))
    return fig

def chart_severity():
    d = st.session_state.data
    months = ["Jan", "Feb", "Mar"]
    pred_months = ["Apr", "May", "Jun"]
    high = [d[m]["highSev"] for m in ["jan","feb","mar"]]
    med  = [d[m]["medSev"]  for m in ["jan","feb","mar"]]
    low  = [d[m]["lowSev"]  for m in ["jan","feb","mar"]]
    hp, mp, lp = forecast(high), forecast(med), forecast(low)

    fig = go.Figure()
    for label, vals, color in [("High", high, C["red"]), ("Medium", med, C["amber"]), ("Low", low, C["green"])]:
        fig.add_trace(go.Bar(x=months, y=vals, name=label, marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>"))
    for label, vals, color in [("High Fcst", hp, C["red"]+"88"), ("Med Fcst", mp, C["amber"]+"88"), ("Low Fcst", lp, C["green"]+"88")]:
        fig.add_trace(go.Bar(x=pred_months, y=vals, name=label, marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>"))
    fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=230,
        xaxis=dict(**PLOT_LAYOUT["xaxis"]),
        yaxis=dict(**PLOT_LAYOUT["yaxis"], showgrid=True))
    return fig

def chart_gauge(value):
    color = C["green"] if value >= 80 else C["amber"] if value >= 60 else C["red"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 42, "color": color, "family": "Sora"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)", "tickfont": {"color": C["muted"], "size": 9}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [{"range": [0, 100], "color": C["border"]}],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.75, "value": value},
            "shape": "angular",
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=20, r=20, t=20, b=0),
        font=dict(family="DM Mono, monospace", color=C["muted"]),
    )
    return fig

def chart_proactivity():
    d = st.session_state.data
    months = ["Jan", "Feb", "Mar"]
    pred_months = ["Apr", "May", "Jun"]
    reactive  = [d[m]["reactive"]  for m in ["jan","feb","mar"]]
    proactive = [d[m]["proactive"] for m in ["jan","feb","mar"]]
    rp, pp = forecast(reactive), forecast(proactive)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=reactive, mode="lines+markers", name="Reactive",
        line=dict(color=C["red"], width=3), marker=dict(size=8, color=C["red"]),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
        hovertemplate="<b>%{x}</b><br>Reactive: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=months, y=proactive, mode="lines+markers", name="Proactive",
        line=dict(color=C["cyan"], width=3), marker=dict(size=8, color=C["cyan"]),
        fill="tozeroy", fillcolor="rgba(6,182,212,0.1)",
        hovertemplate="<b>%{x}</b><br>Proactive: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=["Mar"] + pred_months, y=[reactive[-1]] + rp,
        mode="lines+markers", name="Reactive Fcst",
        line=dict(color=C["red"], width=1.5, dash="dot"),
        marker=dict(size=5, color=C["red"], symbol="circle-open"),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=["Mar"] + pred_months, y=[proactive[-1]] + pp,
        mode="lines+markers", name="Proactive Fcst",
        line=dict(color=C["cyan"], width=1.5, dash="dot"),
        marker=dict(size=5, color=C["cyan"], symbol="circle-open"),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y}<extra></extra>"
    ))
    fig.update_layout(**PLOT_LAYOUT, height=230,
        xaxis=dict(**PLOT_LAYOUT["xaxis"], showgrid=True),
        yaxis=dict(**PLOT_LAYOUT["yaxis"], showgrid=True))
    return fig

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid #27272a'>
        <div style='width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#6366f1,#818cf8);
            display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:800;color:#fff;flex-shrink:0'>S</div>
        <div>
            <div style='font-family:Sora,sans-serif;font-size:14px;font-weight:700;color:#fafafa'>SafetyOS</div>
            <div style='font-family:DM Mono,monospace;font-size:9px;color:#a1a1aa;letter-spacing:0.1em'>EXECUTIVE · v2.4</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Navigation</div>", unsafe_allow_html=True)
    nav_items = [
        ("🔷", "Overview"), ("⬡", "Incidents"), ("⬤", "Audits"),
        ("◉", "Hazards"), ("⊞", "Reports"), ("⊿", "Training"), ("⚙", "Settings")
    ]
    for icon, label in nav_items:
        is_active = st.session_state.page == label
        btn_style = "background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.3)" if is_active else "background:transparent;border:1px solid transparent"
        if st.button(f"{icon}  {label}", key=f"nav_{label}", help=f"Go to {label}",
                     use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.divider()
    st.markdown("<div class='section-header'>⚡ AI Configuration</div>", unsafe_allow_html=True)
    api_key_input = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-ant-...",
        help="Your Anthropic API key for AI features"
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    client_test = get_ai_client()
    if client_test:
        st.markdown("<div style='color:#22c55e;font-size:10px;font-family:DM Mono,monospace;margin-top:4px'>● AI Connected</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#f59e0b;font-size:10px;font-family:DM Mono,monospace;margin-top:4px'>○ No API Key</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='padding:12px;background:rgba(99,102,241,0.1);border-radius:10px;border:1px solid rgba(99,102,241,0.2)'>
        <div style='color:#818cf8;font-size:10px;font-family:DM Mono,monospace;margin-bottom:4px'>Q1 2025 PERIOD</div>
        <div style='color:#fafafa;font-size:12px;font-family:Sora,sans-serif;font-weight:600'>Jan · Feb · Mar</div>
        <div style='color:#a1a1aa;font-size:10px;font-family:DM Mono,monospace;margin-top:2px'>+3 month AI forecast active</div>
    </div>
    """, unsafe_allow_html=True)

# ─── PLACEHOLDER PAGES ────────────────────────────────────────────────────────
if st.session_state.page != "Overview":
    st.markdown(f"""
    <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:60vh;gap:16px'>
        <div style='font-size:60px;opacity:0.15'>⬡</div>
        <div style='font-family:Sora,sans-serif;font-size:24px;font-weight:700;color:#fafafa'>{st.session_state.page}</div>
        <div style='font-family:DM Mono,monospace;font-size:13px;color:#a1a1aa;text-align:center;max-width:340px;line-height:1.7'>
            This module is part of the full SafetyOS suite.<br>
            Navigate to <span style='color:#818cf8'>Overview</span> for live dashboard data.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("← Back to Overview", key="back_home"):
        st.session_state.page = "Overview"
        st.rerun()
    st.stop()

# ─── OVERVIEW PAGE ────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:20px'>
    <div>
        <div style='font-family:Sora,sans-serif;font-size:22px;font-weight:800;color:#fafafa'>Safety Performance</div>
        <div style='font-family:DM Mono,monospace;font-size:10px;color:#a1a1aa;letter-spacing:0.06em;margin-top:2px'>
            EXECUTIVE DASHBOARD · Q1 2025 · AI-POWERED FORECAST
        </div>
    </div>
    <div style='padding:5px 12px;border-radius:8px;background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.3);
        color:#22c55e;font-size:10px;font-family:DM Mono,monospace'>● LIVE</div>
</div>
""", unsafe_allow_html=True)

# ─── KPI STRIP ───────────────────────────────────────────────────────────────
d = st.session_state.data
total_reports = d["jan"]["volume"] + d["feb"]["volume"] + d["mar"]["volume"]
high_reduction = d["jan"]["highSev"] - d["mar"]["highSev"]
audit_avg = round((d["jan"]["auditRate"] + d["feb"]["auditRate"] + d["mar"]["auditRate"]) / 3)
proactive_uplift = d["mar"]["proactive"] - d["jan"]["proactive"]

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>◈ Total Reports (Q1)</div>
        <div class='metric-value' style='color:#818cf8'>{total_reports}</div>
        <div class='metric-sub'>reports submitted</div>
    </div>""", unsafe_allow_html=True)
with k2:
    color = C["green"] if high_reduction > 0 else C["red"]
    arrow = "↓" if high_reduction > 0 else "↑"
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>⬡ High Sev Reduction</div>
        <div class='metric-value' style='color:{color}'>{arrow} {abs(high_reduction)} events</div>
        <div class='metric-sub'>Jan vs Mar delta</div>
    </div>""", unsafe_allow_html=True)
with k3:
    color = C["green"] if audit_avg >= 80 else C["amber"] if audit_avg >= 60 else C["red"]
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>◎ Avg Audit Closure</div>
        <div class='metric-value' style='color:{color}'>{audit_avg}%</div>
        <div class='metric-sub'>Q1 average rate</div>
    </div>""", unsafe_allow_html=True)
with k4:
    color = C["green"] if proactive_uplift > 0 else C["red"]
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>◉ Proactive Uplift</div>
        <div class='metric-value' style='color:{color}'>+{proactive_uplift} IDs</div>
        <div class='metric-sub'>Jan → Mar growth</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin: 16px 0'></div>", unsafe_allow_html=True)

# ─── MAIN TABS ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Charts", "🤖  AI Insights", "✏️  Input Data"])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""<div class='metric-card' style='margin-bottom:16px'>
            <div class='metric-label'>◈ Metric 01 — Reporting Volume</div>
            <div style='color:#fafafa;font-size:14px;font-weight:600;font-family:Sora,sans-serif;margin-bottom:2px'>Monthly Report Submissions</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(chart_volume(), use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("""<div class='metric-card' style='margin-bottom:16px'>
            <div class='metric-label'>⬡ Metric 02 — Incident Severity</div>
            <div style='color:#fafafa;font-size:14px;font-weight:600;font-family:Sora,sans-serif;margin-bottom:2px'>Stacked Severity Distribution</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(chart_severity(), use_container_width=True, config={"displayModeBar": False})

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""<div class='metric-card' style='margin-bottom:16px'>
            <div class='metric-label'>◎ Metric 03 — Audit Closure Rate</div>
            <div style='color:#fafafa;font-size:14px;font-weight:600;font-family:Sora,sans-serif;margin-bottom:2px'>Radial Gauge — Q1 Average</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(chart_gauge(audit_avg), use_container_width=True, config={"displayModeBar": False})

        # Monthly breakdown
        mc1, mc2, mc3 = st.columns(3)
        for col, month, key in [(mc1, "Jan", "jan"), (mc2, "Feb", "feb"), (mc3, "Mar", "mar")]:
            v = d[key]["auditRate"]
            c = C["green"] if v >= 80 else C["amber"] if v >= 60 else C["red"]
            with col:
                st.markdown(f"""<div style='background:#09090b;border:1px solid #27272a;border-radius:10px;
                    padding:10px;text-align:center;'>
                    <div style='font-family:DM Mono,monospace;font-size:9px;color:#a1a1aa'>{month}</div>
                    <div style='font-family:Sora,sans-serif;font-size:18px;font-weight:800;color:{c}'>{v}%</div>
                </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("""<div class='metric-card' style='margin-bottom:16px'>
            <div class='metric-label'>◉ Metric 04 — Hazard Proactivity</div>
            <div style='color:#fafafa;font-size:14px;font-weight:600;font-family:Sora,sans-serif;margin-bottom:2px'>Reactive vs Proactive Trend</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(chart_proactivity(), use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

    ai_col1, ai_col2 = st.columns([1.2, 1])

    with ai_col1:
        st.markdown("""<div style='font-family:Sora,sans-serif;font-size:15px;font-weight:700;color:#fafafa;margin-bottom:8px'>
            🤖 AI Executive Summary
        </div>""", unsafe_allow_html=True)

        if st.button("⚡ Generate Executive Summary", key="gen_summary"):
            with st.spinner("Analysing Q1 safety data..."):
                st.session_state.last_ai_summary = ai_generate_summary()

        if st.session_state.last_ai_summary:
            st.markdown(f"""<div class='ai-insight'>
                <div class='ai-insight-header'>🛡️ GM Executive Briefing — Q1 2025</div>
                {st.session_state.last_ai_summary}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class='ai-insight' style='opacity:0.5;text-align:center;padding:28px'>
                Click "Generate Executive Summary" to get an AI-powered briefing on your Q1 safety data.
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin:16px 0 8px 0;font-family:Sora,sans-serif;font-size:15px;font-weight:700;color:#fafafa'>📈 Per-Metric AI Insights</div>", unsafe_allow_html=True)

        metrics_for_insight = [
            ("Reporting Volume", [d[m]["volume"] for m in ["jan","feb","mar"]]),
            ("High Severity Events", [d[m]["highSev"] for m in ["jan","feb","mar"]]),
            ("Audit Closure Rate (%)", [d[m]["auditRate"] for m in ["jan","feb","mar"]]),
            ("Proactive Hazard IDs", [d[m]["proactive"] for m in ["jan","feb","mar"]]),
        ]

        for metric_name, vals in metrics_for_insight:
            with st.expander(f"🔍 {metric_name}", expanded=False):
                pred = forecast(vals)
                trend = "↑" if vals[2] > vals[0] else "↓"
                pct = round(abs(vals[2] - vals[0]) / max(vals[0], 1) * 100)
                c_trend = C["green"] if (trend == "↑" and "Severity" not in metric_name) or (trend == "↓" and "Severity" in metric_name) else C["red"]

                mc, fc = st.columns(2)
                with mc:
                    st.markdown(f"""<div style='font-family:DM Mono,monospace;font-size:11px;color:#a1a1aa'>
                        Jan: {vals[0]} → Feb: {vals[1]} → Mar: {vals[2]}<br>
                        <span style='color:{c_trend}'>{trend} {pct}% Q1 change</span>
                    </div>""", unsafe_allow_html=True)
                with fc:
                    st.markdown(f"""<div style='font-family:DM Mono,monospace;font-size:11px;color:#a1a1aa'>
                        Apr: {pred[0]} · May: {pred[1]} · Jun: {pred[2]}<br>
                        <span style='color:#818cf8'>Linear forecast</span>
                    </div>""", unsafe_allow_html=True)

                if st.button(f"Get AI Insight", key=f"insight_{metric_name}"):
                    with st.spinner(""):
                        insight = ai_predict_metric(metric_name, vals)
                        if insight:
                            st.markdown(f"""<div class='ai-insight' style='margin-top:8px'>
                                <div class='ai-insight-header'>💡 GM Insight</div>
                                {insight}
                            </div>""", unsafe_allow_html=True)

    with ai_col2:
        st.markdown("""<div style='font-family:Sora,sans-serif;font-size:15px;font-weight:700;color:#fafafa;margin-bottom:8px'>
            💬 Chat with SafetyOS AI
        </div>""", unsafe_allow_html=True)

        # Chat history display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""<div style='background:#1c1c1f;border:1px solid #27272a;border-radius:12px;padding:20px;
                    text-align:center;font-family:DM Mono,monospace;font-size:11px;color:#a1a1aa;'>
                    Ask me anything about your safety data.<br><br>
                    Try: "What's the biggest risk this quarter?" or<br>"Which metrics need the most attention?"
                </div>""", unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history[-10:]:
                    role_color = "#818cf8" if msg["role"] == "assistant" else "#22c55e"
                    role_label = "🤖 SafetyOS AI" if msg["role"] == "assistant" else "👤 You"
                    bg = "#1c1c1f" if msg["role"] == "assistant" else "#18181b"
                    st.markdown(f"""<div style='background:{bg};border:1px solid #27272a;border-radius:10px;
                        padding:12px 14px;margin-bottom:8px;'>
                        <div style='color:{role_color};font-size:10px;font-family:DM Mono,monospace;margin-bottom:6px'>{role_label}</div>
                        <div style='color:#fafafa;font-size:13px;font-family:Sora,sans-serif;line-height:1.6'>{msg["content"]}</div>
                    </div>""", unsafe_allow_html=True)

        # Quick prompts
        st.markdown("<div style='margin-top:10px;margin-bottom:6px;font-family:DM Mono,monospace;font-size:9px;color:#a1a1aa;letter-spacing:0.1em'>QUICK PROMPTS</div>", unsafe_allow_html=True)
        qp_cols = st.columns(2)
        quick_prompts = [
            "What's our biggest Q2 risk?",
            "Summarise high severity trend",
            "Are we hitting proactive targets?",
            "What should the GM focus on?",
        ]
        for i, prompt in enumerate(quick_prompts):
            with qp_cols[i % 2]:
                if st.button(prompt, key=f"qp_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    with st.spinner(""):
                        reply = ai_chat_response(prompt)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

        # Chat input
        user_input = st.text_input("", placeholder="Ask about your safety data...", key="chat_input", label_visibility="collapsed")
        if st.button("Send →", key="send_chat"):
            if user_input.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.spinner("Thinking..."):
                    reply = ai_chat_response(user_input)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑 Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""<div style='font-family:Sora,sans-serif;font-size:15px;font-weight:700;color:#fafafa;margin-bottom:4px'>
        ✏️ Update Q1 Safety Data
    </div>
    <div style='font-family:DM Mono,monospace;font-size:11px;color:#a1a1aa;margin-bottom:18px'>
        Edit values below — charts and AI analysis will reflect changes immediately on save.
    </div>""", unsafe_allow_html=True)

    fields = [
        ("volume",    "📋 Reporting Volume",       "reports"),
        ("highSev",   "🔴 High Severity Events",    "events"),
        ("medSev",    "🟡 Medium Severity Events",  "events"),
        ("lowSev",    "🟢 Low Severity Events",     "events"),
        ("auditRate", "📊 Audit Closure Rate",      "%"),
        ("reactive",  "⚡ Reactive Reports",        "reports"),
        ("proactive", "✅ Proactive Hazard IDs",    "IDs"),
    ]

    form_data = {m: dict(d[m]) for m in ["jan", "feb", "mar"]}

    # Column headers
    hdr0, hdr1, hdr2, hdr3 = st.columns([2, 1, 1, 1])
    with hdr0: st.markdown("<div style='font-family:DM Mono,monospace;font-size:10px;color:#a1a1aa;padding:6px 0'>METRIC</div>", unsafe_allow_html=True)
    for col, label in [(hdr1,"January"), (hdr2,"February"), (hdr3,"March")]:
        with col: st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:10px;color:#818cf8;padding:6px 0;text-align:center'>{label}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:4px 0 12px 0'>", unsafe_allow_html=True)

    for key, label, unit in fields:
        c0, c1, c2, c3 = st.columns([2, 1, 1, 1])
        with c0:
            st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:11px;color:#a1a1aa;padding:10px 0'>{label} <span style='opacity:0.5'>({unit})</span></div>", unsafe_allow_html=True)
        for col, month in [(c1,"jan"), (c2,"feb"), (c3,"mar")]:
            with col:
                form_data[month][key] = st.number_input(
                    f"{label}_{month}", min_value=0, max_value=9999,
                    value=int(d[month][key]),
                    label_visibility="collapsed",
                    key=f"input_{month}_{key}"
                )

    st.markdown("<div style='margin:20px 0 8px 0'></div>", unsafe_allow_html=True)
    sv_col, _ = st.columns([1, 2])
    with sv_col:
        if st.button("💾 Save & Update All Charts", key="save_data"):
            st.session_state.data = form_data
            st.session_state.last_ai_summary = None  # reset summary to reflect new data
            st.success("✅ Data saved! Charts and AI analysis updated.")
            st.rerun()

    st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)

    # Data preview table
    with st.expander("📋 Preview Current Data Table"):
        rows = []
        for key, label, unit in fields:
            row = {"Metric": f"{label} ({unit})"}
            for m, mlabel in [("jan","Jan"), ("feb","Feb"), ("mar","Mar")]:
                row[mlabel] = d[m][key]
            vals = [d[m][key] for m in ["jan","feb","mar"]]
            preds = forecast(vals)
            row["Apr (Fcst)"] = preds[0]
            row["May (Fcst)"] = preds[1]
            row["Jun (Fcst)"] = preds[2]
            rows.append(row)
        df = pd.DataFrame(rows).set_index("Metric")
        st.dataframe(df, use_container_width=True)
