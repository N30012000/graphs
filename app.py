import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import anthropic
import json

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

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp { background-color: #09090b; color: #fafafa; }

[data-testid="stSidebar"] { background-color: #18181b !important; border-right: 1px solid #27272a; }

#MainMenu, footer, header { visibility: hidden; }

.metric-card {
    background: #1c1c1f; border: 1px solid #27272a;
    border-radius: 14px; padding: 18px 22px; margin-bottom: 4px;
}
.metric-label {
    font-family: 'DM Mono', monospace; font-size: 10px; color: #a1a1aa;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;
}
.metric-value { font-family: 'Sora', sans-serif; font-size: 26px; font-weight: 800; }
.metric-sub   { font-family: 'DM Mono', monospace; font-size: 10px; color: #a1a1aa; margin-top: 4px; }

.section-hdr {
    font-family: 'DM Mono', monospace; font-size: 10px; color: #a1a1aa;
    text-transform: uppercase; letter-spacing: 0.12em; margin: 12px 0 6px 4px;
}

.nav-btn {
    display: flex; align-items: center; gap: 10px; padding: 9px 12px;
    border-radius: 9px; margin-bottom: 3px; cursor: pointer;
    font-family: 'Sora', sans-serif; font-size: 13px; color: #a1a1aa;
    border: 1px solid transparent; transition: all 0.15s;
    text-decoration: none; width: 100%;
}
.nav-btn:hover { background: rgba(255,255,255,0.05); color: #fafafa; }
.nav-btn.active {
    background: rgba(99,102,241,0.2); border-color: rgba(99,102,241,0.3);
    color: #fafafa; font-weight: 600;
}
.nav-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #818cf8; margin-left: auto; flex-shrink: 0;
}

.ai-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(129,140,248,0.06));
    border: 1px solid rgba(99,102,241,0.3); border-radius: 12px;
    padding: 16px 18px; margin: 8px 0;
    font-family: 'DM Mono', monospace; font-size: 12px;
    color: #c7d2fe; line-height: 1.75;
}
.ai-box-hdr {
    font-family: 'Sora', sans-serif; font-size: 12px; font-weight: 700;
    color: #818cf8; margin-bottom: 8px;
}

.chat-msg {
    border: 1px solid #27272a; border-radius: 10px;
    padding: 12px 14px; margin-bottom: 8px;
}
.chat-role {
    font-size: 10px; font-family: 'DM Mono', monospace; margin-bottom: 6px;
}
.chat-text {
    color: #fafafa; font-size: 13px; font-family: 'Sora', sans-serif; line-height: 1.6;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important; font-weight: 700 !important;
    padding: 8px 20px !important; width: 100%;
}
.stButton > button:hover { opacity: 0.9 !important; }

.stNumberInput input, .stTextInput input, .stTextArea textarea {
    background: #09090b !important; border: 1px solid #27272a !important;
    color: #fafafa !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #18181b; border-radius: 10px; gap: 4px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #a1a1aa !important;
    border-radius: 8px !important; font-family: 'Sora', sans-serif !important; font-size: 13px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important; color: #fafafa !important;
}
div[data-testid="stDataFrame"] { background: #1c1c1f !important; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "data": {
        "jan": {"volume": 42, "highSev": 8,  "medSev": 14, "lowSev": 20, "auditRate": 61, "reactive": 38, "proactive": 14},
        "feb": {"volume": 57, "highSev": 6,  "medSev": 16, "lowSev": 22, "auditRate": 74, "reactive": 35, "proactive": 22},
        "mar": {"volume": 71, "highSev": 4,  "medSev": 17, "lowSev": 28, "auditRate": 83, "reactive": 30, "proactive": 31},
    },
    "chat_history": [],
    "last_ai_summary": None,
    "api_key": "",
    "page": "Overview",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── PALETTE ──────────────────────────────────────────────────────────────────
C = {
    "bg":      "#09090b",
    "surface": "#18181b",
    "card":    "#1c1c1f",
    "border":  "#27272a",
    "text":    "#fafafa",
    "muted":   "#a1a1aa",
    "accent":  "#6366f1",
    "alight":  "#818cf8",
    "green":   "#22c55e",
    "red":     "#ef4444",
    "amber":   "#f59e0b",
    "cyan":    "#06b6d4",
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def forecast(vals, n=3):
    x = np.arange(len(vals), dtype=float)
    slope, intercept = np.polyfit(x, vals, 1)
    return [max(0, round(intercept + slope * (len(vals) + i))) for i in range(n)]

def base_layout(height=240, barmode=None):
    """Return a clean plotly layout dict — no duplicate keys."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        font=dict(family="DM Mono, monospace", color=C["muted"], size=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=C["muted"]),
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",   x=0,
        ),
        xaxis=dict(
            gridcolor=C["border"],
            linecolor="rgba(0,0,0,0)",
            tickcolor="rgba(0,0,0,0)",
            tickfont=dict(family="DM Mono, monospace", color=C["muted"], size=10),
        ),
        yaxis=dict(
            gridcolor=C["border"],
            linecolor="rgba(0,0,0,0)",
            tickcolor="rgba(0,0,0,0)",
            tickfont=dict(family="DM Mono, monospace", color=C["muted"], size=10),
            showgrid=True,
        ),
    )
    if barmode:
        layout["barmode"] = barmode
    return layout

def get_data_summary():
    d = st.session_state.data
    months = ["jan", "feb", "mar"]
    return {
        "reporting_volume":    {m: d[m]["volume"]    for m in months},
        "high_severity":       {m: d[m]["highSev"]   for m in months},
        "medium_severity":     {m: d[m]["medSev"]    for m in months},
        "low_severity":        {m: d[m]["lowSev"]    for m in months},
        "audit_closure_pct":   {m: d[m]["auditRate"] for m in months},
        "reactive_reports":    {m: d[m]["reactive"]  for m in months},
        "proactive_hazard_ids":{m: d[m]["proactive"] for m in months},
        "forecasts": {
            "volume_apr_jun":   forecast([d[m]["volume"]    for m in months]),
            "high_sev_apr_jun": forecast([d[m]["highSev"]   for m in months]),
            "audit_apr_jun":    forecast([d[m]["auditRate"] for m in months]),
            "proactive_apr_jun":forecast([d[m]["proactive"] for m in months]),
        },
    }

# ─── AI HELPERS ───────────────────────────────────────────────────────────────
def get_client():
    key = st.session_state.get("api_key", "").strip()
    if key:
        return anthropic.Anthropic(api_key=key)
    try:
        return anthropic.Anthropic()   # picks up ANTHROPIC_API_KEY env var if set
    except Exception:
        return None

def ai_summary():
    client = get_client()
    if not client:
        return "⚠️ No API key configured — paste your Anthropic key in the sidebar."
    summary = get_data_summary()
    prompt = (
        "You are a Senior Safety Performance Analyst providing a GM executive briefing.\n\n"
        f"Q1 2025 safety data:\n{json.dumps(summary, indent=2)}\n\n"
        "Write a concise executive summary (4-6 sentences) covering:\n"
        "1. Overall safety culture trajectory\n"
        "2. Key wins and any concerning trends\n"
        "3. Top 2 recommended actions for Q2\n"
        "4. One forward-looking prediction based on forecast data\n\n"
        "Professional, direct language. Include specific numbers. No bullet points."
    )
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except Exception as e:
        return f"⚠️ AI Error: {e}"

def ai_chat(user_msg):
    client = get_client()
    if not client:
        return "⚠️ No API key configured — paste your Anthropic key in the sidebar."
    summary = get_data_summary()
    system = (
        "You are SafetyOS AI, an expert safety performance analyst.\n\n"
        f"Current Q1 2025 data:\n{json.dumps(summary, indent=2)}\n\n"
        "Answer concisely (2-4 sentences) with specific numbers and actionable advice."
    )
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.chat_history[-8:]]
    history.append({"role": "user", "content": user_msg})
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=system,
            messages=history,
        )
        return resp.content[0].text
    except Exception as e:
        return f"⚠️ AI Error: {e}"

def ai_metric_insight(name, vals):
    client = get_client()
    if not client:
        return "Configure API key for AI insights."
    pred = forecast(vals)
    prompt = (
        f"Safety metric '{name}': Jan={vals[0]}, Feb={vals[1]}, Mar={vals[2]}. "
        f"Forecast: Apr={pred[0]}, May={pred[1]}, Jun={pred[2]}. "
        "Write ONE direct GM-level sentence about this trend and Q2 implication."
    )
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except Exception:
        return None

# ─── CHART BUILDERS ───────────────────────────────────────────────────────────
def chart_volume():
    d = st.session_state.data
    months    = ["Jan", "Feb", "Mar"]
    vals      = [d["jan"]["volume"], d["feb"]["volume"], d["mar"]["volume"]]
    pred_months = ["Apr", "May", "Jun"]
    preds     = forecast(vals)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=vals, mode="lines+markers", name="Actual",
        line=dict(color=C["accent"], width=3),
        marker=dict(size=8, color=C["accent"]),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.12)",
        hovertemplate="<b>%{x}</b><br>Reports: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=["Mar"] + pred_months, y=[vals[-1]] + preds,
        mode="lines+markers", name="Forecast",
        line=dict(color=C["alight"], width=2, dash="dot"),
        marker=dict(size=6, color=C["alight"], symbol="circle-open"),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y}<extra></extra>",
    ))
    fig.update_layout(**base_layout(240))
    return fig

def chart_severity():
    d = st.session_state.data
    months      = ["Jan", "Feb", "Mar"]
    pred_months = ["Apr", "May", "Jun"]
    high = [d[m]["highSev"] for m in ["jan","feb","mar"]]
    med  = [d[m]["medSev"]  for m in ["jan","feb","mar"]]
    low  = [d[m]["lowSev"]  for m in ["jan","feb","mar"]]
    hp, mp, lp = forecast(high), forecast(med), forecast(low)

    fig = go.Figure()
    for label, vals, color in [
        ("High",     high, C["red"]),
        ("Medium",   med,  C["amber"]),
        ("Low",      low,  C["green"]),
    ]:
        fig.add_trace(go.Bar(
            x=months, y=vals, name=label, marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>",
        ))
    for label, vals, color in [
        ("High Fcst",   hp, "rgba(239,68,68,0.45)"),
        ("Med Fcst",    mp, "rgba(245,158,11,0.45)"),
        ("Low Fcst",    lp, "rgba(34,197,94,0.45)"),
    ]:
        fig.add_trace(go.Bar(
            x=pred_months, y=vals, name=label, marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(**base_layout(240, barmode="stack"))
    return fig

def chart_gauge(value):
    color = C["green"] if value >= 80 else C["amber"] if value >= 60 else C["red"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 40, "color": color, "family": "Sora, sans-serif"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 0,
                "tickcolor": "rgba(0,0,0,0)",
                "tickfont": {"color": C["muted"], "size": 9},
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [{"range": [0, 100], "color": C["border"]}],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": value,
            },
            "shape": "angular",
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=210,
        margin=dict(l=20, r=20, t=20, b=0),
        font=dict(family="DM Mono, monospace", color=C["muted"]),
    )
    return fig

def chart_proactivity():
    d = st.session_state.data
    months      = ["Jan", "Feb", "Mar"]
    pred_months = ["Apr", "May", "Jun"]
    reactive  = [d[m]["reactive"]  for m in ["jan","feb","mar"]]
    proactive = [d[m]["proactive"] for m in ["jan","feb","mar"]]
    rp, pp = forecast(reactive), forecast(proactive)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=reactive, mode="lines+markers", name="Reactive",
        line=dict(color=C["red"], width=3),
        marker=dict(size=8, color=C["red"]),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.10)",
        hovertemplate="<b>%{x}</b><br>Reactive: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=proactive, mode="lines+markers", name="Proactive",
        line=dict(color=C["cyan"], width=3),
        marker=dict(size=8, color=C["cyan"]),
        fill="tozeroy", fillcolor="rgba(6,182,212,0.10)",
        hovertemplate="<b>%{x}</b><br>Proactive: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=["Mar"] + pred_months, y=[reactive[-1]] + rp,
        mode="lines+markers", name="Reactive Fcst",
        line=dict(color=C["red"], width=1.5, dash="dot"),
        marker=dict(size=5, color=C["red"], symbol="circle-open"),
        hovertemplate="<b>%{x}</b><br>Fcst Reactive: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=["Mar"] + pred_months, y=[proactive[-1]] + pp,
        mode="lines+markers", name="Proactive Fcst",
        line=dict(color=C["cyan"], width=1.5, dash="dot"),
        marker=dict(size=5, color=C["cyan"], symbol="circle-open"),
        hovertemplate="<b>%{x}</b><br>Fcst Proactive: %{y}<extra></extra>",
    ))
    fig.update_layout(**base_layout(240))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
NAV_ITEMS = [
    ("🔷", "Overview"),
    ("⬡",  "Incidents"),
    ("⬤",  "Audits"),
    ("◉",  "Hazards"),
    ("⊞",  "Reports"),
    ("⊿",  "Training"),
    ("⚙",  "Settings"),
]

with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:20px;
                padding-bottom:16px;border-bottom:1px solid #27272a'>
        <div style='width:34px;height:34px;border-radius:10px;flex-shrink:0;
                    background:linear-gradient(135deg,#6366f1,#818cf8);
                    display:flex;align-items:center;justify-content:center;
                    font-size:15px;font-weight:800;color:#fff'>S</div>
        <div>
            <div style='font-family:Sora,sans-serif;font-size:14px;font-weight:700;color:#fafafa;line-height:1.1'>SafetyOS</div>
            <div style='font-family:DM Mono,monospace;font-size:9px;color:#a1a1aa;letter-spacing:0.1em'>EXECUTIVE · v2.4</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-hdr'>Navigation</div>", unsafe_allow_html=True)

    for icon, label in NAV_ITEMS:
        active = st.session_state.page == label
        active_cls = "active" if active else ""
        dot_html = "<span class='nav-dot'></span>" if active else ""
        # Render a styled button via st.button, but wrap in HTML for looks
        clicked = st.button(
            f"{icon}  {label}",
            key=f"nav_{label}",
            use_container_width=True,
        )
        if clicked:
            st.session_state.page = label
            st.rerun()

    st.divider()
    st.markdown("<div class='section-hdr'>⚡ AI Configuration</div>", unsafe_allow_html=True)

    new_key = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-ant-...",
        label_visibility="collapsed",
    )
    if new_key != st.session_state.api_key:
        st.session_state.api_key = new_key

    connected = bool(st.session_state.api_key.strip())
    dot_color = C["green"] if connected else C["amber"]
    dot_label = "AI Connected" if connected else "Add API Key above"
    st.markdown(
        f"<div style='color:{dot_color};font-size:10px;font-family:DM Mono,monospace;margin-top:4px'>● {dot_label}</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("""
    <div style='padding:12px;background:rgba(99,102,241,0.1);border-radius:10px;
                border:1px solid rgba(99,102,241,0.2)'>
        <div style='color:#818cf8;font-size:10px;font-family:DM Mono,monospace;margin-bottom:4px'>Q1 2025 PERIOD</div>
        <div style='color:#fafafa;font-size:12px;font-family:Sora,sans-serif;font-weight:600'>Jan · Feb · Mar</div>
        <div style='color:#a1a1aa;font-size:10px;font-family:DM Mono,monospace;margin-top:2px'>+3 month AI forecast</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: page header
# ══════════════════════════════════════════════════════════════════════════════
def page_header(title, subtitle):
    st.markdown(f"""
    <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:18px'>
        <div>
            <div style='font-family:Sora,sans-serif;font-size:22px;font-weight:800;color:#fafafa'>{title}</div>
            <div style='font-family:DM Mono,monospace;font-size:10px;color:#a1a1aa;letter-spacing:0.06em;margin-top:2px'>{subtitle}</div>
        </div>
    </div>""", unsafe_allow_html=True)

def card(content_html):
    st.markdown(f"<div class='metric-card' style='margin-bottom:12px'>{content_html}</div>", unsafe_allow_html=True)

def section(title):
    st.markdown(f"<div style='font-family:Sora,sans-serif;font-size:15px;font-weight:700;color:#fafafa;margin:18px 0 10px'>{title}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INCIDENTS PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Incidents":
    d = st.session_state.data
    page_header("Incident Management", "INCIDENTS · Q1 2025 · ALL SEVERITY LEVELS")

    # KPIs
    total_high = sum(d[m]["highSev"] for m in ["jan","feb","mar"])
    total_med  = sum(d[m]["medSev"]  for m in ["jan","feb","mar"])
    total_low  = sum(d[m]["lowSev"]  for m in ["jan","feb","mar"])
    total_all  = total_high + total_med + total_low

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Incidents Q1</div><div class='metric-value' style='color:#818cf8'>{total_all}</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='metric-card'><div class='metric-label'>High Severity</div><div class='metric-value' style='color:#ef4444'>{total_high}</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Medium Severity</div><div class='metric-value' style='color:#f59e0b'>{total_med}</div></div>", unsafe_allow_html=True)
    with k4: st.markdown(f"<div class='metric-card'><div class='metric-label'>Low Severity</div><div class='metric-value' style='color:#22c55e'>{total_low}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        section("Monthly Incident Trend by Severity")
        months = ["Jan","Feb","Mar"]
        fig = go.Figure()
        for label, key, color in [("High","highSev",C["red"]),("Medium","medSev",C["amber"]),("Low","lowSev",C["green"])]:
            vals = [d[m][key] for m in ["jan","feb","mar"]]
            fig.add_trace(go.Scatter(x=months, y=vals, mode="lines+markers", name=label,
                line=dict(color=color, width=2.5), marker=dict(size=7, color=color),
                hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>"))
        fig.update_layout(**base_layout(260))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        section("Severity Distribution — Q1 Total")
        labels = ["High", "Medium", "Low"]
        values = [total_high, total_med, total_low]
        colors = [C["red"], C["amber"], C["green"]]
        fig2 = go.Figure(go.Pie(labels=labels, values=values, hole=0.55,
            marker=dict(colors=colors, line=dict(color=C["card"], width=2)),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, margin=dict(l=8,r=8,t=8,b=8),
            font=dict(family="DM Mono,monospace", color=C["muted"], size=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"])))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    section("Incident Log — Q1 2025")
    inc_data = {
        "Month":    ["Jan","Jan","Jan","Feb","Feb","Feb","Mar","Mar","Mar"],
        "Severity": ["High","Medium","Low","High","Medium","Low","High","Medium","Low"],
        "Count":    [d["jan"]["highSev"], d["jan"]["medSev"], d["jan"]["lowSev"],
                     d["feb"]["highSev"], d["feb"]["medSev"], d["feb"]["lowSev"],
                     d["mar"]["highSev"], d["mar"]["medSev"], d["mar"]["lowSev"]],
        "Trend":    ["↓ Improving","→ Stable","↑ Growing","↓ Improving","↑ Growing","↑ Growing",
                     "↓ Improving","→ Stable","↑ Growing"],
        "Action Required": ["Yes","Monitor","No","Yes","Monitor","No","Review","Monitor","No"],
    }
    st.dataframe(pd.DataFrame(inc_data), use_container_width=True, hide_index=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# AUDITS PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Audits":
    d = st.session_state.data
    page_header("Audit Management", "AUDITS · Q1 2025 · CLOSURE TRACKING")

    rates = [d[m]["auditRate"] for m in ["jan","feb","mar"]]
    avg_rate = round(sum(rates)/3)
    trend_pct = round(((rates[2]-rates[0])/max(rates[0],1))*100)

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Q1 Avg Closure Rate</div><div class='metric-value' style='color:#f59e0b'>{avg_rate}%</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Jan Rate</div><div class='metric-value' style='color:#818cf8'>{rates[0]}%</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Mar Rate</div><div class='metric-value' style='color:#22c55e'>{rates[2]}%</div></div>", unsafe_allow_html=True)
    with k4:
        c = C["green"] if trend_pct > 0 else C["red"]
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Q1 Trend</div><div class='metric-value' style='color:{c}'>+{trend_pct}%</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        section("Closure Rate Trend + Forecast")
        months = ["Jan","Feb","Mar"]
        preds  = forecast(rates)
        pred_m = ["Apr","May","Jun"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=rates, mode="lines+markers", name="Actual",
            line=dict(color=C["amber"], width=3), marker=dict(size=8, color=C["amber"]),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.1)",
            hovertemplate="<b>%{x}</b><br>Rate: %{y}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=["Mar"]+pred_m, y=[rates[-1]]+preds,
            mode="lines+markers", name="Forecast",
            line=dict(color=C["alight"], width=2, dash="dot"),
            marker=dict(size=5, color=C["alight"], symbol="circle-open"),
            hovertemplate="<b>%{x}</b><br>Forecast: %{y}%<extra></extra>"))
        fig.add_hline(y=90, line_dash="dash", line_color=C["green"], opacity=0.5,
            annotation_text="90% Target", annotation_font_color=C["green"])
        fig.update_layout(**base_layout(260))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        section("Audit Categories — Simulated Breakdown")
        cats = ["Safety Inspections","Equipment Checks","Process Reviews","Documentation","Training Audits"]
        closed  = [18, 14, 9, 22, 11]
        open_   = [2,  4,  3, 1,  3]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=cats, y=closed, name="Closed", marker_color=C["green"]))
        fig2.add_trace(go.Bar(x=cats, y=open_,  name="Open",   marker_color=C["red"]))
        fig2.update_layout(**base_layout(260, barmode="stack"))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    section("Audit Schedule — Q2 2025 Forecast")
    preds = forecast(rates)
    sched = pd.DataFrame({
        "Month":        ["Apr","May","Jun"],
        "Forecast Rate":  [f"{p}%" for p in preds],
        "Target":         ["90%","90%","90%"],
        "Status":         ["On Track" if p>=80 else "At Risk" for p in preds],
        "Audits Planned": [24, 26, 28],
    })
    st.dataframe(sched, use_container_width=True, hide_index=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# HAZARDS PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Hazards":
    d = st.session_state.data
    page_header("Hazard Management", "HAZARDS · Q1 2025 · PROACTIVE vs REACTIVE")

    react  = [d[m]["reactive"]  for m in ["jan","feb","mar"]]
    proact = [d[m]["proactive"] for m in ["jan","feb","mar"]]
    ratio_jan = round(proact[0]/max(react[0],1)*100)
    ratio_mar = round(proact[2]/max(react[2],1)*100)

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Reactive Q1</div><div class='metric-value' style='color:#ef4444'>{sum(react)}</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Proactive Q1</div><div class='metric-value' style='color:#06b6d4'>{sum(proact)}</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Proactivity Ratio Jan</div><div class='metric-value' style='color:#a1a1aa'>{ratio_jan}%</div></div>", unsafe_allow_html=True)
    with k4:
        c = C["green"] if ratio_mar > ratio_jan else C["red"]
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Proactivity Ratio Mar</div><div class='metric-value' style='color:{c}'>{ratio_mar}%</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        section("Reactive vs Proactive — Monthly Trend")
        months = ["Jan","Feb","Mar"]
        rp, pp = forecast(react), forecast(proact)
        pred_m = ["Apr","May","Jun"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=react,  name="Reactive",  marker_color=C["red"],  opacity=0.85))
        fig.add_trace(go.Bar(x=months, y=proact, name="Proactive", marker_color=C["cyan"], opacity=0.85))
        fig.add_trace(go.Bar(x=pred_m, y=rp,  name="React Fcst",  marker_color="rgba(239,68,68,0.4)"))
        fig.add_trace(go.Bar(x=pred_m, y=pp,  name="Proact Fcst", marker_color="rgba(6,182,212,0.4)"))
        fig.update_layout(**base_layout(260, barmode="group"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        section("Proactivity Ratio Over Time")
        ratios = [round(proact[i]/max(react[i],1)*100) for i in range(3)]
        pred_ratios = forecast(ratios)
        months = ["Jan","Feb","Mar"]
        pred_m = ["Apr","May","Jun"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=months, y=ratios, mode="lines+markers", name="Ratio %",
            line=dict(color=C["cyan"], width=3), marker=dict(size=8, color=C["cyan"]),
            fill="tozeroy", fillcolor="rgba(6,182,212,0.1)",
            hovertemplate="<b>%{x}</b><br>Ratio: %{y}%<extra></extra>"))
        fig2.add_trace(go.Scatter(x=["Mar"]+pred_m, y=[ratios[-1]]+pred_ratios,
            mode="lines+markers", name="Forecast",
            line=dict(color=C["alight"], width=2, dash="dot"),
            marker=dict(size=5, color=C["alight"], symbol="circle-open")))
        fig2.add_hline(y=100, line_dash="dash", line_color=C["green"], opacity=0.5,
            annotation_text="1:1 Parity Target", annotation_font_color=C["green"])
        fig2.update_layout(**base_layout(260))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    section("Hazard Register — Q1 2025 Summary")
    haz_df = pd.DataFrame({
        "Hazard Category":  ["Manual Handling","Chemical Exposure","Electrical","Working at Height","Machinery","Ergonomics"],
        "Reactive Reports": [12, 8, 5, 9, 7, 11],
        "Proactive IDs":    [8, 14, 9, 6, 12, 18],
        "Risk Level":       ["Medium","High","High","High","Medium","Low"],
        "Status":           ["In Progress","Escalated","Resolved","In Progress","Resolved","Monitoring"],
    })
    st.dataframe(haz_df, use_container_width=True, hide_index=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# REPORTS PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Reports":
    d = st.session_state.data
    page_header("Reports & Submissions", "REPORTS · Q1 2025 · VOLUME & CULTURE TRACKING")

    vols = [d[m]["volume"] for m in ["jan","feb","mar"]]
    total = sum(vols)
    growth = round(((vols[2]-vols[0])/max(vols[0],1))*100)

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Reports Q1</div><div class='metric-value' style='color:#818cf8'>{total}</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Jan Volume</div><div class='metric-value' style='color:#a1a1aa'>{vols[0]}</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Mar Volume</div><div class='metric-value' style='color:#22c55e'>{vols[2]}</div></div>", unsafe_allow_html=True)
    with k4:
        c = C["green"] if growth > 0 else C["red"]
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Q1 Growth</div><div class='metric-value' style='color:{c}'>+{growth}%</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        section("Reporting Volume — Trend + Forecast")
        months = ["Jan","Feb","Mar"]
        preds  = forecast(vols)
        pred_m = ["Apr","May","Jun"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=vols, name="Actual",
            marker_color=C["accent"], opacity=0.9,
            hovertemplate="<b>%{x}</b><br>Reports: %{y}<extra></extra>"))
        fig.add_trace(go.Bar(x=pred_m, y=preds, name="Forecast",
            marker_color="rgba(99,102,241,0.4)",
            hovertemplate="<b>%{x}</b><br>Forecast: %{y}<extra></extra>"))
        fig.update_layout(**base_layout(260))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        section("Report Type Breakdown — Q1 Estimate")
        types  = ["Near Miss","Hazard ID","Incident","Unsafe Act","Unsafe Condition","Observation"]
        counts = [38, 45, 18, 22, 29, 18]
        fig2 = go.Figure(go.Pie(labels=types, values=counts, hole=0.5,
            marker=dict(colors=[C["amber"],C["cyan"],C["red"],C["green"],C["alight"],"#a855f7"],
                        line=dict(color=C["card"], width=2)),
            hovertemplate="<b>%{label}</b><br>%{value} reports<br>%{percent}<extra></extra>"))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, margin=dict(l=8,r=8,t=8,b=8),
            font=dict(family="DM Mono,monospace", color=C["muted"], size=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"], size=9)))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    section("Monthly Report Log")
    rpt_df = pd.DataFrame({
        "Month":       ["Jan","Jan","Feb","Feb","Mar","Mar"],
        "Type":        ["Near Miss","Hazard ID","Near Miss","Incident","Hazard ID","Observation"],
        "Count":       [14, 16, 18, 6, 22, 9],
        "Dept":        ["Operations","Maintenance","Logistics","Operations","Maintenance","All"],
        "Closed":      ["Yes","Yes","Yes","In Progress","Yes","Yes"],
    })
    st.dataframe(rpt_df, use_container_width=True, hide_index=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Training":
    page_header("Training & Competency", "TRAINING · Q1 2025 · COMPLETION & COMPLIANCE")

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Staff Trained Q1</div><div class='metric-value' style='color:#818cf8'>148</div><div class='metric-sub'>of 162 total</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Completion Rate</div><div class='metric-value' style='color:#22c55e'>91%</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Overdue Courses</div><div class='metric-value' style='color:#ef4444'>14</div></div>", unsafe_allow_html=True)
    with k4: st.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Score</div><div class='metric-value' style='color:#f59e0b'>84%</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        section("Monthly Training Completions")
        months = ["Jan","Feb","Mar"]
        completed = [44, 52, 52]
        target    = [50, 50, 50]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=completed, name="Completed", marker_color=C["green"]))
        fig.add_trace(go.Scatter(x=months, y=target, mode="lines", name="Target",
            line=dict(color=C["red"], width=2, dash="dash")))
        fig.update_layout(**base_layout(260))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        section("Completion by Course Type")
        courses = ["Manual Handling","Fire Safety","First Aid","Chemical Handling","PPE Use","Emergency Resp."]
        rates   = [95, 88, 72, 91, 98, 83]
        colors  = [C["green"] if r>=90 else C["amber"] if r>=75 else C["red"] for r in rates]
        fig2 = go.Figure(go.Bar(x=courses, y=rates, marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Completion: %{y}%<extra></extra>"))
        fig2.add_hline(y=90, line_dash="dash", line_color=C["green"], opacity=0.5,
            annotation_text="90% Target", annotation_font_color=C["green"])
        fig2.update_layout(**base_layout(260))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    section("Training Register — Q1 2025")
    train_df = pd.DataFrame({
        "Course":       ["Manual Handling","Fire Safety","First Aid","Chemical Handling","PPE Use","Emergency Response"],
        "Required":     [55, 55, 40, 30, 55, 45],
        "Completed":    [52, 48, 29, 27, 54, 37],
        "Completion %": ["95%","87%","73%","90%","98%","82%"],
        "Next Due":     ["Jul 2025","Apr 2025","Jun 2025","Jul 2025","Oct 2025","May 2025"],
        "Status":       ["✅ On Track","⚠ At Risk","❌ Behind","✅ On Track","✅ On Track","⚠ At Risk"],
    })
    st.dataframe(train_df, use_container_width=True, hide_index=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Settings":
    page_header("Settings", "CONFIGURATION · SAFETYOS DASHBOARD")

    section("🔑 AI Configuration")
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    new_key = st.text_input("Anthropic API Key", value=st.session_state.api_key,
        type="password", placeholder="sk-ant-...",
        help="Get your key at https://console.anthropic.com")
    if new_key != st.session_state.api_key:
        st.session_state.api_key = new_key
    connected = bool(st.session_state.api_key.strip())
    if connected:
        st.success("✅ API Key configured — all AI features are active.")
    else:
        st.warning("⚠️ No API key set. AI features (summary, chat, insights) require an Anthropic key.")
    st.markdown("</div>", unsafe_allow_html=True)

    section("📅 Reporting Period")
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        st.selectbox("Report Period", ["Q1 2025 (Jan–Mar)", "Q2 2025 (Apr–Jun)", "Q3 2025 (Jul–Sep)", "Q4 2025 (Oct–Dec)"], index=0)
    with sc2:
        st.selectbox("Organisation", ["Head Office", "Site A", "Site B", "All Sites"], index=3)
    st.markdown("</div>", unsafe_allow_html=True)

    section("🎯 KPI Targets")
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    with t1: st.number_input("Audit Closure Target (%)", min_value=0, max_value=100, value=90)
    with t2: st.number_input("Max High Severity Events", min_value=0, max_value=50, value=5)
    with t3: st.number_input("Min Proactive Reports", min_value=0, max_value=200, value=30)
    st.markdown("</div>", unsafe_allow_html=True)

    section("🗑️ Data Management")
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        if st.button("Reset to Default Data", key="reset_data"):
            st.session_state.data = {
                "jan": {"volume":42,"highSev":8,"medSev":14,"lowSev":20,"auditRate":61,"reactive":38,"proactive":14},
                "feb": {"volume":57,"highSev":6,"medSev":16,"lowSev":22,"auditRate":74,"reactive":35,"proactive":22},
                "mar": {"volume":71,"highSev":4,"medSev":17,"lowSev":28,"auditRate":83,"reactive":30,"proactive":31},
            }
            st.session_state.last_ai_summary = None
            st.session_state.chat_history = []
            st.success("✅ Data reset to defaults.")
            st.rerun()
    with r2:
        if st.button("Clear AI Chat History", key="clear_all_chat"):
            st.session_state.chat_history = []
            st.success("✅ Chat history cleared.")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW PAGE
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:18px'>
    <div>
        <div style='font-family:Sora,sans-serif;font-size:22px;font-weight:800;color:#fafafa'>
            Safety Performance
        </div>
        <div style='font-family:DM Mono,monospace;font-size:10px;color:#a1a1aa;letter-spacing:0.06em;margin-top:2px'>
            EXECUTIVE DASHBOARD · Q1 2025 · AI-POWERED FORECAST
        </div>
    </div>
    <div style='padding:5px 14px;border-radius:8px;background:rgba(34,197,94,0.12);
                border:1px solid rgba(34,197,94,0.3);color:#22c55e;
                font-size:10px;font-family:DM Mono,monospace'>● LIVE</div>
</div>
""", unsafe_allow_html=True)

# ─── KPI STRIP ────────────────────────────────────────────────────────────────
d  = st.session_state.data
total_v    = d["jan"]["volume"]    + d["feb"]["volume"]    + d["mar"]["volume"]
high_delta = d["jan"]["highSev"]   - d["mar"]["highSev"]
audit_avg  = round((d["jan"]["auditRate"] + d["feb"]["auditRate"] + d["mar"]["auditRate"]) / 3)
pro_delta  = d["mar"]["proactive"] - d["jan"]["proactive"]

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>◈ Total Reports (Q1)</div>
        <div class='metric-value' style='color:#818cf8'>{total_v}</div>
        <div class='metric-sub'>reports submitted</div>
    </div>""", unsafe_allow_html=True)
with k2:
    c = C["green"] if high_delta > 0 else C["red"]
    arrow = "↓" if high_delta > 0 else "↑"
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>⬡ High Sev Reduction</div>
        <div class='metric-value' style='color:{c}'>{arrow} {abs(high_delta)} events</div>
        <div class='metric-sub'>Jan vs Mar delta</div>
    </div>""", unsafe_allow_html=True)
with k3:
    c = C["green"] if audit_avg >= 80 else C["amber"] if audit_avg >= 60 else C["red"]
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>◎ Avg Audit Closure</div>
        <div class='metric-value' style='color:{c}'>{audit_avg}%</div>
        <div class='metric-sub'>Q1 average rate</div>
    </div>""", unsafe_allow_html=True)
with k4:
    c = C["green"] if pro_delta > 0 else C["red"]
    sign = "+" if pro_delta > 0 else ""
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>◉ Proactive Uplift</div>
        <div class='metric-value' style='color:{c}'>{sign}{pro_delta} IDs</div>
        <div class='metric-sub'>Jan → Mar growth</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_charts, tab_ai, tab_input = st.tabs(["📊  Charts", "🤖  AI Insights", "✏️  Input Data"])

# ════════════════════════════════════════
# TAB 1 — CHARTS
# ════════════════════════════════════════
with tab_charts:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""<div class='metric-card' style='margin-bottom:10px'>
            <div class='metric-label'>◈ Metric 01 — Reporting Volume</div>
            <div style='color:#fafafa;font-size:14px;font-weight:600;font-family:Sora,sans-serif'>
                Monthly Report Submissions</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(chart_volume(), use_container_width=True,
                        config={"displayModeBar": False})

    with c2:
        st.markdown("""<div class='metric-card' style='margin-bottom:10px'>
            <div class='metric-label'>⬡ Metric 02 — Incident Severity</div>
            <div style='color:#fafafa;font-size:14px;font-weight:600;font-family:Sora,sans-serif'>
                Stacked Severity Distribution</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(chart_severity(), use_container_width=True,
                        config={"displayModeBar": False})

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("""<div class='metric-card' style='margin-bottom:10px'>
            <div class='metric-label'>◎ Metric 03 — Audit Closure Rate</div>
            <div style='color:#fafafa;font-size:14px;font-weight:600;font-family:Sora,sans-serif'>
                Radial Gauge — Q1 Average</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(chart_gauge(audit_avg), use_container_width=True,
                        config={"displayModeBar": False})
        # monthly mini-cards
        m1, m2, m3 = st.columns(3)
        for col, lbl, key in [(m1,"Jan","jan"), (m2,"Feb","feb"), (m3,"Mar","mar")]:
            v = d[key]["auditRate"]
            c = C["green"] if v >= 80 else C["amber"] if v >= 60 else C["red"]
            with col:
                st.markdown(f"""
                <div style='background:#09090b;border:1px solid #27272a;border-radius:10px;
                            padding:10px;text-align:center'>
                    <div style='font-family:DM Mono,monospace;font-size:9px;color:#a1a1aa'>{lbl}</div>
                    <div style='font-family:Sora,sans-serif;font-size:17px;font-weight:800;color:{c}'>{v}%</div>
                </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown("""<div class='metric-card' style='margin-bottom:10px'>
            <div class='metric-label'>◉ Metric 04 — Hazard Proactivity</div>
            <div style='color:#fafafa;font-size:14px;font-weight:600;font-family:Sora,sans-serif'>
                Reactive vs Proactive Trend</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(chart_proactivity(), use_container_width=True,
                        config={"displayModeBar": False})

# ════════════════════════════════════════
# TAB 2 — AI INSIGHTS
# ════════════════════════════════════════
with tab_ai:
    st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
    ai_left, ai_right = st.columns([1.2, 1])

    # ── Left: summary + per-metric ──
    with ai_left:
        st.markdown("""<div style='font-family:Sora,sans-serif;font-size:15px;
                        font-weight:700;color:#fafafa;margin-bottom:8px'>
            🤖 AI Executive Summary</div>""", unsafe_allow_html=True)

        if st.button("⚡ Generate Executive Summary", key="btn_summary"):
            with st.spinner("Analysing Q1 safety data..."):
                st.session_state.last_ai_summary = ai_summary()

        if st.session_state.last_ai_summary:
            st.markdown(f"""<div class='ai-box'>
                <div class='ai-box-hdr'>🛡️ GM Executive Briefing — Q1 2025</div>
                {st.session_state.last_ai_summary}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class='ai-box' style='opacity:0.4;text-align:center;padding:28px'>
                Click "Generate Executive Summary" to get an AI-powered briefing.
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div style='font-family:Sora,sans-serif;font-size:15px;
                        font-weight:700;color:#fafafa;margin:16px 0 8px'>
            📈 Per-Metric AI Insights</div>""", unsafe_allow_html=True)

        METRICS = [
            ("Reporting Volume",        [d[m]["volume"]    for m in ["jan","feb","mar"]]),
            ("High Severity Events",    [d[m]["highSev"]   for m in ["jan","feb","mar"]]),
            ("Audit Closure Rate (%)",  [d[m]["auditRate"] for m in ["jan","feb","mar"]]),
            ("Proactive Hazard IDs",    [d[m]["proactive"] for m in ["jan","feb","mar"]]),
        ]
        for name, vals in METRICS:
            with st.expander(f"🔍  {name}"):
                pred = forecast(vals)
                pct_change = round(abs(vals[2] - vals[0]) / max(vals[0], 1) * 100)
                trend = "↑" if vals[2] > vals[0] else "↓"
                is_bad = (trend == "↑" and "Severity" in name) or (trend == "↓" and "Severity" not in name)
                trend_color = C["red"] if is_bad else C["green"]

                ca, cb = st.columns(2)
                with ca:
                    st.markdown(f"""<div style='font-family:DM Mono,monospace;font-size:11px;color:#a1a1aa;line-height:1.8'>
                        Jan: <b style='color:#fafafa'>{vals[0]}</b> →
                        Feb: <b style='color:#fafafa'>{vals[1]}</b> →
                        Mar: <b style='color:#fafafa'>{vals[2]}</b><br>
                        <span style='color:{trend_color}'>{trend} {pct_change}% Q1 change</span>
                    </div>""", unsafe_allow_html=True)
                with cb:
                    st.markdown(f"""<div style='font-family:DM Mono,monospace;font-size:11px;color:#a1a1aa;line-height:1.8'>
                        Apr: <b style='color:#818cf8'>{pred[0]}</b> ·
                        May: <b style='color:#818cf8'>{pred[1]}</b> ·
                        Jun: <b style='color:#818cf8'>{pred[2]}</b><br>
                        <span style='color:#818cf8'>Linear regression forecast</span>
                    </div>""", unsafe_allow_html=True)

                if st.button("Get AI Insight", key=f"ai_btn_{name}"):
                    with st.spinner(""):
                        insight = ai_metric_insight(name, vals)
                        if insight:
                            st.markdown(f"""<div class='ai-box' style='margin-top:8px'>
                                <div class='ai-box-hdr'>💡 GM Insight</div>
                                {insight}
                            </div>""", unsafe_allow_html=True)

    # ── Right: chat ──
    with ai_right:
        st.markdown("""<div style='font-family:Sora,sans-serif;font-size:15px;
                        font-weight:700;color:#fafafa;margin-bottom:8px'>
            💬 Chat with SafetyOS AI</div>""", unsafe_allow_html=True)

        # Messages
        if not st.session_state.chat_history:
            st.markdown("""<div style='background:#1c1c1f;border:1px solid #27272a;
                            border-radius:12px;padding:22px;text-align:center;
                            font-family:DM Mono,monospace;font-size:11px;color:#a1a1aa;line-height:1.8'>
                Ask me anything about your safety data.<br><br>
                <i>"What's the biggest Q2 risk?"</i><br>
                <i>"Are we hitting proactive targets?"</i>
            </div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history[-10:]:
                is_ai = msg["role"] == "assistant"
                role_color = C["alight"] if is_ai else C["green"]
                role_label = "🤖 SafetyOS AI" if is_ai else "👤 You"
                bg = "#1c1c1f" if is_ai else "#18181b"
                st.markdown(f"""<div class='chat-msg' style='background:{bg}'>
                    <div class='chat-role' style='color:{role_color}'>{role_label}</div>
                    <div class='chat-text'>{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)

        # Quick prompts
        st.markdown("<div style='margin-top:10px;font-family:DM Mono,monospace;font-size:9px;color:#a1a1aa;letter-spacing:0.1em;margin-bottom:4px'>QUICK PROMPTS</div>", unsafe_allow_html=True)
        qp1, qp2 = st.columns(2)
        QUICK = [
            "What's our biggest Q2 risk?",
            "Summarise high severity",
            "Are proactive targets met?",
            "Top action for the GM?",
        ]
        for i, prompt in enumerate(QUICK):
            with (qp1 if i % 2 == 0 else qp2):
                if st.button(prompt, key=f"qp_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    with st.spinner(""):
                        reply = ai_chat(prompt)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

        # Free-form input
        user_input = st.text_input(
            "chat_input_label",
            placeholder="Ask about your safety data...",
            key="chat_input",
            label_visibility="collapsed",
        )
        send_col, clear_col = st.columns([2, 1])
        with send_col:
            if st.button("Send →", key="send_btn"):
                if user_input.strip():
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    with st.spinner("Thinking..."):
                        reply = ai_chat(user_input)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()
        with clear_col:
            if st.session_state.chat_history:
                if st.button("🗑 Clear", key="clear_btn"):
                    st.session_state.chat_history = []
                    st.rerun()

# ════════════════════════════════════════
# TAB 3 — INPUT DATA
# ════════════════════════════════════════
with tab_input:
    st.markdown("""<div style='font-family:Sora,sans-serif;font-size:15px;font-weight:700;color:#fafafa;margin-bottom:2px'>
        ✏️ Update Q1 Safety Data</div>
    <div style='font-family:DM Mono,monospace;font-size:11px;color:#a1a1aa;margin-bottom:18px'>
        Edit values below — all charts and AI analysis update on save.</div>
    """, unsafe_allow_html=True)

    FIELDS = [
        ("volume",    "📋 Reporting Volume",      "reports"),
        ("highSev",   "🔴 High Severity Events",  "events"),
        ("medSev",    "🟡 Medium Severity Events", "events"),
        ("lowSev",    "🟢 Low Severity Events",    "events"),
        ("auditRate", "📊 Audit Closure Rate",     "%"),
        ("reactive",  "⚡ Reactive Reports",       "reports"),
        ("proactive", "✅ Proactive Hazard IDs",   "IDs"),
    ]

    # Copy current data into local form state
    form_data = {m: dict(d[m]) for m in ["jan", "feb", "mar"]}

    # Column headers
    h0, h1, h2, h3 = st.columns([2, 1, 1, 1])
    for col, txt, color in [
        (h0, "METRIC",   "#a1a1aa"),
        (h1, "JANUARY",  "#818cf8"),
        (h2, "FEBRUARY", "#818cf8"),
        (h3, "MARCH",    "#818cf8"),
    ]:
        with col:
            st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:10px;color:{color};padding:6px 0'>{txt}</div>",
                        unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#27272a;margin:4px 0 10px 0'>", unsafe_allow_html=True)

    for key, label, unit in FIELDS:
        c0, c1, c2, c3 = st.columns([2, 1, 1, 1])
        with c0:
            st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:11px;color:#a1a1aa;padding:10px 0'>"
                        f"{label} <span style='opacity:0.4'>({unit})</span></div>",
                        unsafe_allow_html=True)
        for col, month in [(c1,"jan"), (c2,"feb"), (c3,"mar")]:
            with col:
                form_data[month][key] = st.number_input(
                    f"_{key}_{month}",
                    min_value=0, max_value=9999,
                    value=int(d[month][key]),
                    label_visibility="collapsed",
                    key=f"inp_{month}_{key}",
                )

    st.markdown("<div style='margin:18px 0 6px'></div>", unsafe_allow_html=True)
    sv, _ = st.columns([1, 2])
    with sv:
        if st.button("💾  Save & Update All Charts", key="save_btn"):
            st.session_state.data = {m: dict(form_data[m]) for m in ["jan","feb","mar"]}
            st.session_state.last_ai_summary = None
            st.success("✅ Data saved! All charts and AI analysis updated.")
            st.rerun()

    st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

    with st.expander("📋  Preview Data + Forecasts Table"):
        rows = []
        for key, label, unit in FIELDS:
            vals = [d[m][key] for m in ["jan","feb","mar"]]
            preds = forecast(vals)
            rows.append({
                "Metric": f"{label} ({unit})",
                "Jan": vals[0], "Feb": vals[1], "Mar": vals[2],
                "Apr (Fcst)": preds[0], "May (Fcst)": preds[1], "Jun (Fcst)": preds[2],
            })
        df = pd.DataFrame(rows).set_index("Metric")
        st.dataframe(df, use_container_width=True)
