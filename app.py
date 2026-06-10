import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import anthropic
import json
from datetime import datetime, date
import io

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirSial SafetyOS",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0c0e14; color: #e2e8f0; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1117 0%, #111420 100%) !important;
    border-right: 1px solid #1e2235;
}

/* Metric Cards */
.kpi-card {
    background: linear-gradient(135deg, #141726 0%, #1a1f30 100%);
    border: 1px solid #242840;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-line, linear-gradient(90deg, #4f6ef7, #818cf8));
}
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #5a6380;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 30px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #3d4460;
    margin-top: 6px;
}

/* Section headers */
.section-title {
    font-size: 13px;
    font-weight: 700;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 22px 0 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2235;
}

/* AI Box */
.ai-box {
    background: linear-gradient(135deg, rgba(79,110,247,0.08), rgba(129,140,248,0.04));
    border: 1px solid rgba(79,110,247,0.25);
    border-radius: 14px;
    padding: 20px 22px;
    color: #c4ccf0;
    font-size: 13.5px;
    line-height: 1.7;
}
.ai-box-hdr {
    color: #818cf8;
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 10px;
}

/* Chat messages */
.chat-msg {
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-size: 13px;
    line-height: 1.6;
}
.chat-role {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.chat-text { color: #c4ccf0; }

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.badge-green { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.badge-red { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-amber { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-blue { background: rgba(79,110,247,0.15); color: #818cf8; border: 1px solid rgba(79,110,247,0.3); }

/* Form styling */
.form-section {
    background: #141726;
    border: 1px solid #1e2235;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 16px;
}
.form-section-title {
    font-size: 11px;
    font-weight: 700;
    color: #4f6ef7;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e2235;
}

/* Table styles */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox select,
.stNumberInput input, .stDateInput input {
    background: #141726 !important;
    border: 1px solid #242840 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #4f6ef7 !important;
    box-shadow: 0 0 0 2px rgba(79,110,247,0.2) !important;
}

/* Sidebar styling */
.sidebar-logo {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 24px; padding-bottom: 20px;
    border-bottom: 1px solid #1e2235;
}
.sidebar-logo-icon {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #4f6ef7, #818cf8);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; font-weight: 900; color: #fff; flex-shrink: 0;
}
.sidebar-logo-text { line-height: 1.2; }
.sidebar-logo-name {
    font-size: 15px; font-weight: 800; color: #e2e8f0;
    letter-spacing: -0.02em;
}
.sidebar-logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #3d4460; letter-spacing: 0.12em;
    text-transform: uppercase; margin-top: 2px;
}
.section-hdr {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #3d4460;
    text-transform: uppercase; letter-spacing: 0.15em;
    margin: 16px 0 8px; padding-left: 2px;
}

/* Nav active state */
.nav-active { background: rgba(79,110,247,0.15) !important; color: #818cf8 !important; }

/* Page header */
.page-hdr {
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid #1e2235;
}
.page-title {
    font-size: 24px; font-weight: 800; color: #e2e8f0;
    letter-spacing: -0.03em; line-height: 1.1;
}
.page-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #3d4460;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-top: 4px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #141726;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2235;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #5a6380 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: #1e2235 !important;
    color: #818cf8 !important;
    font-weight: 600 !important;
}

.stButton button {
    background: #141726 !important;
    border: 1px solid #242840 !important;
    color: #8892b0 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    transition: all 0.15s !important;
}
.stButton button:hover {
    border-color: #4f6ef7 !important;
    color: #818cf8 !important;
    background: rgba(79,110,247,0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── COLOUR PALETTE ───────────────────────────────────────────────────────────
C = {
    "bg": "#0c0e14", "surface": "#141726", "card": "#1a1f30",
    "border": "#242840", "text": "#e2e8f0", "muted": "#5a6380",
    "accent": "#4f6ef7", "alight": "#818cf8",
    "green": "#22c55e", "red": "#ef4444", "amber": "#f59e0b", "cyan": "#06b6d4",
    "purple": "#a855f7",
}

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def load_excel_data():
    data = {}
    files = {
        "bird_hits": "BIRD_HITS.xlsx",
        "fsr": "FSR.xlsx",
        "hira": "HIRA.xlsx",
        "mor": "MOR.xlsx",
    }
    for key, fname in files.items():
        try:
            df = pd.read_excel(fname)
            df.columns = [str(c).strip() for c in df.columns]
            data[key] = df
        except Exception as e:
            data[key] = pd.DataFrame()
    return data

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_ai_summary" not in st.session_state:
    st.session_state.last_ai_summary = None
if "org_name" not in st.session_state:
    st.session_state.org_name = "Air Sial"
if "report_period" not in st.session_state:
    st.session_state.report_period = "2025–2026"
if "audit_target" not in st.session_state:
    st.session_state.audit_target = 90
if "max_high_sev" not in st.session_state:
    st.session_state.max_high_sev = 5
if "theme_accent" not in st.session_state:
    st.session_state.theme_accent = "Indigo"

excel_data = load_excel_data()
bird_df = excel_data["bird_hits"]
fsr_df  = excel_data["fsr"]
hira_df = excel_data["hira"]
mor_df  = excel_data["mor"]

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def forecast(vals, n=3):
    if len(vals) < 2:
        return [vals[-1]] * n if vals else [0]*n
    x = np.arange(len(vals), dtype=float)
    slope, intercept = np.polyfit(x, vals, 1)
    return [max(0, round(intercept + slope * (len(vals) + i))) for i in range(n)]

def base_layout(height=260, barmode=None):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        font=dict(family="JetBrains Mono, monospace", color=C["muted"], size=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=C["muted"]),
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        ),
        xaxis=dict(
            gridcolor="#1e2235", linecolor="rgba(0,0,0,0)",
            tickcolor="rgba(0,0,0,0)",
            tickfont=dict(family="JetBrains Mono, monospace", color=C["muted"], size=10),
        ),
        yaxis=dict(
            gridcolor="#1e2235", linecolor="rgba(0,0,0,0)",
            tickcolor="rgba(0,0,0,0)",
            tickfont=dict(family="JetBrains Mono, monospace", color=C["muted"], size=10),
            showgrid=True,
        ),
    )
    if barmode:
        layout["barmode"] = barmode
    return layout

def page_header(title, subtitle, badge=None):
    badge_html = f"<span class='badge badge-green' style='margin-left:12px'>{badge}</span>" if badge else ""
    st.markdown(f"""
    <div class='page-hdr'>
        <div style='display:flex;align-items:center'>
            <div class='page-title'>{title}</div>{badge_html}
        </div>
        <div class='page-subtitle'>{subtitle}</div>
    </div>""", unsafe_allow_html=True)

def kpi(label, value, color, sub="", accent="linear-gradient(90deg, #4f6ef7, #818cf8)"):
    st.markdown(f"""
    <div class='kpi-card' style='--accent-line: {accent}'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value' style='color:{color}'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)

# ─── AI HELPERS ───────────────────────────────────────────────────────────────
def get_client():
    key = st.session_state.get("api_key","").strip()
    if key:
        return anthropic.Anthropic(api_key=key)
    try:
        return anthropic.Anthropic()
    except:
        return None

def build_context():
    ctx = {}
    if not bird_df.empty:
        ctx["bird_hits_total"] = len(bird_df)
        ctx["bird_hit_phases"] = bird_df["PHASE OF FLIGHT"].value_counts().to_dict() if "PHASE OF FLIGHT" in bird_df.columns else {}
    if not fsr_df.empty:
        ctx["fsr_total"] = len(fsr_df)
        ctx["fsr_factors"] = fsr_df["AFFECTED FACTORS"].value_counts().head(8).to_dict() if "AFFECTED FACTORS" in fsr_df.columns else {}
    if not hira_df.empty:
        ctx["hira_total"] = len(hira_df)
        ctx["hira_risk_ratings"] = hira_df["int. risk rating"].value_counts().to_dict() if "int. risk rating" in hira_df.columns else {}
        ctx["hira_status"] = hira_df["status"].value_counts().to_dict() if "status" in hira_df.columns else {}
    if not mor_df.empty:
        ctx["mor_total"] = len(mor_df)
        ctx["mor_causes"] = mor_df["NATURE AND CAUSE"].value_counts().head(8).to_dict() if "NATURE AND CAUSE" in mor_df.columns else {}
    return ctx

def ai_summary():
    client = get_client()
    if not client:
        return "⚠️ No API key configured. Add your Anthropic key in Settings."
    ctx = build_context()
    prompt = (
        f"You are a Senior Aviation Safety Analyst for {st.session_state.org_name}, briefing the GM.\n\n"
        f"Operational safety data:\n{json.dumps(ctx, indent=2)}\n\n"
        "Write a concise executive summary (4-6 sentences) covering:\n"
        "1. Overall safety culture and reporting trajectory\n"
        "2. Most significant risk areas from the data\n"
        "3. Top 2 recommended actions for the next quarter\n"
        "4. One forward-looking prediction\n\n"
        "Professional, direct language. Include specific numbers. No bullet points."
    )
    try:
        resp = get_client().messages.create(
            model="claude-sonnet-4-20250514", max_tokens=500,
            messages=[{"role":"user","content":prompt}]
        )
        return resp.content[0].text
    except Exception as e:
        return f"⚠️ AI Error: {e}"

def ai_chat(user_msg):
    client = get_client()
    if not client:
        return "⚠️ No API key configured. Add your Anthropic key in Settings."
    ctx = build_context()
    system = (
        f"You are SafetyOS AI for {st.session_state.org_name}, an expert aviation safety analyst.\n\n"
        f"Safety data:\n{json.dumps(ctx, indent=2)}\n\n"
        "Answer concisely (2-4 sentences) with specific numbers and actionable aviation safety advice."
    )
    history = [{"role":m["role"],"content":m["content"]} for m in st.session_state.chat_history[-8:]]
    history.append({"role":"user","content":user_msg})
    try:
        resp = get_client().messages.create(
            model="claude-sonnet-4-20250514", max_tokens=600,
            system=system, messages=history
        )
        return resp.content[0].text
    except Exception as e:
        return f"⚠️ AI Error: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
NAV = [
    ("✈️", "Overview"),
    ("🐦", "Bird Hits"),
    ("📋", "FSR"),
    ("⚠️", "HIRA"),
    ("📊", "MOR"),
    ("🤖", "AI Insights"),
    ("⚙️", "Settings"),
]

with st.sidebar:
    st.markdown(f"""
    <div class='sidebar-logo'>
        <div class='sidebar-logo-icon'>✈</div>
        <div class='sidebar-logo-text'>
            <div class='sidebar-logo-name'>{st.session_state.org_name}</div>
            <div class='sidebar-logo-sub'>SafetyOS · v3.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-hdr'>Navigation</div>", unsafe_allow_html=True)
    for icon, label in NAV:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.divider()
    st.markdown("<div class='section-hdr'>AI Configuration</div>", unsafe_allow_html=True)
    new_key = st.text_input("API Key", value=st.session_state.api_key,
        type="password", placeholder="sk-ant-...", label_visibility="collapsed")
    if new_key != st.session_state.api_key:
        st.session_state.api_key = new_key
    connected = bool(st.session_state.api_key.strip())
    color = C["green"] if connected else C["amber"]
    label = "AI Connected" if connected else "Add key for AI features"
    st.markdown(f"<div style='color:{color};font-size:10px;font-family:JetBrains Mono,monospace;margin-top:4px'>● {label}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style='padding:12px;background:rgba(79,110,247,0.08);border-radius:10px;border:1px solid rgba(79,110,247,0.18)'>
        <div style='color:#4f6ef7;font-size:9px;font-family:JetBrains Mono,monospace;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px'>Period</div>
        <div style='color:#e2e8f0;font-size:13px;font-weight:700'>{st.session_state.report_period}</div>
        <div style='color:#3d4460;font-size:9px;font-family:JetBrains Mono,monospace;margin-top:3px'>
            Bird: {len(bird_df)} · FSR: {len(fsr_df)} · HIRA: {len(hira_df)} · MOR: {len(mor_df)}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# BIRD HITS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Bird Hits":
    page_header("Bird Strike Reports", "BIRD / ANIMAL HIT LOG · CAAF-067-MSXX-2.0", "Live Data")

    df = bird_df.copy()
    total = len(df)
    phases = df["PHASE OF FLIGHT"].value_counts() if "PHASE OF FLIGHT" in df.columns else pd.Series()
    locations = df["LOCATION"].value_counts() if "LOCATION" in df.columns else pd.Series()
    effects = df["EFFECT ON FLIGHT"].value_counts() if "EFFECT ON FLIGHT" in df.columns else pd.Series()

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Total Events", total, C["alight"], "all records", "linear-gradient(90deg,#4f6ef7,#818cf8)")
    with k2: kpi("On Airport", int(locations.get("ON AIRPORT",0)), C["amber"], "within 200ft AGL", "linear-gradient(90deg,#f59e0b,#fbbf24)")
    with k3: kpi("Near Airport", int(locations.get("NEAR AIRPORT",0)), C["cyan"], "201–1000ft AGL", "linear-gradient(90deg,#06b6d4,#67e8f9)")
    with k4: kpi("Off Airport", int(locations.get("OFF AIRPORT",0)), C["green"], ">1000ft AGL", "linear-gradient(90deg,#22c55e,#86efac)")

    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        section("Bird Strikes by Phase of Flight")
        if not phases.empty:
            top_phases = phases.head(8)
            fig = go.Figure(go.Bar(
                x=top_phases.values, y=top_phases.index,
                orientation="h",
                marker=dict(
                    color=top_phases.values,
                    colorscale=[[0,"#1e2235"],[1,"#4f6ef7"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{y}</b><br>%{x} events<extra></extra>",
            ))
            fig.update_layout(**base_layout(280))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with c2:
        section("Location Distribution")
        if not locations.empty:
            colors_pie = [C["amber"], C["cyan"], C["green"], C["alight"]]
            fig2 = go.Figure(go.Pie(
                labels=locations.index, values=locations.values,
                hole=0.6,
                marker=dict(colors=colors_pie, line=dict(color="#0c0e14", width=2)),
                hovertemplate="<b>%{label}</b><br>%{value} strikes<br>%{percent}<extra></extra>",
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=280, margin=dict(l=8,r=8,t=8,b=8),
                font=dict(family="JetBrains Mono,monospace", color=C["muted"], size=10),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"])),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    section("Parts of Aircraft Affected")
    if "PARTS OF AIRCRAFT AFFECTED" in df.columns:
        parts = df["PARTS OF AIRCRAFT AFFECTED"].dropna()
        all_parts = []
        for p in parts:
            for item in str(p).split(","):
                item = item.strip()
                if item and item.lower() not in ["nan","none","nil",""]:
                    all_parts.append(item.upper())
        if all_parts:
            from collections import Counter
            part_counts = Counter(all_parts)
            top = dict(sorted(part_counts.items(), key=lambda x: -x[1])[:10])
            fig3 = go.Figure(go.Bar(
                x=list(top.keys()), y=list(top.values()),
                marker_color=C["accent"], opacity=0.9,
                hovertemplate="<b>%{x}</b><br>%{y} strikes<extra></extra>",
            ))
            fig3.update_layout(**base_layout(200))
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

    section("Full Bird Hit Log")
    display_cols = [c for c in ["DATE OF OCCURRENCE","AIRCRAFT TYPE","FLT #","REGISTRATION","DESTINATION",
                                 "PHASE OF FLIGHT","LOCATION","HEIGHT (AGL)","PARTS OF AIRCRAFT AFFECTED","REMARKS"]
                    if c in df.columns]
    st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, hide_index=True)

    section("Submit New Bird Hit Report")
    with st.expander("➕ Open CAAF-067 Report Form"):
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        st.markdown("<div class='form-section-title'>Operator & Aircraft Information</div>", unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            new_operator = st.text_input("Operator Name", value="Air Sial", key="bh_operator")
            new_reg = st.text_input("Aircraft Registration", placeholder="AP-BOA", key="bh_reg")
        with fc2:
            new_ac_type = st.selectbox("Aircraft Type/Category", ["A320","Light","Medium","Heavy"], key="bh_type")
            new_flt = st.text_input("Flight No.", placeholder="PF-721", key="bh_flt")
        with fc3:
            new_dest = st.text_input("Destination", placeholder="OPKC", key="bh_dest")
            new_date = st.date_input("Date of Occurrence", key="bh_date")

        st.markdown("<div class='form-section-title' style='margin-top:16px'>Event Details</div>", unsafe_allow_html=True)
        fd1, fd2, fd3 = st.columns(3)
        with fd1:
            new_aero = st.text_input("Aerodrome Name", placeholder="OPLA", key="bh_aero")
            new_rwy = st.text_input("Runway Used", placeholder="36R", key="bh_rwy")
            new_time = st.text_input("Local/UTC Time of Bird Hit", placeholder="10:15 UTC", key="bh_time")
        with fd2:
            new_phase = st.selectbox("Phase of Flight", ["TAKEOFF","CLIMB","CRUISE","DESCENT","APPROACH","LANDING","LANDING ROLL","TAXI"], key="bh_phase")
            new_loc = st.selectbox("Location", ["ON AIRPORT","NEAR AIRPORT","OFF AIRPORT"], key="bh_loc")
            new_height = st.number_input("Height AGL (ft)", min_value=0, max_value=50000, key="bh_height")
        with fd3:
            new_speed = st.number_input("Speed IAS (kts)", min_value=0, max_value=400, key="bh_speed")
            new_status = st.selectbox("Bird Strike Status", ["Suspected","Confirmed"], key="bh_status")
            new_effect = st.selectbox("Effect on Flight", ["NONE","Aborted Take off","Precautionary Landing","Engines Shut down"], key="bh_effect")

        fe1, fe2 = st.columns(2)
        with fe1:
            new_weather = st.text_input("Weather Condition / METAR", key="bh_weather")
            new_species = st.text_input("Bird/Animal Species (if known)", key="bh_species")
            new_num = st.text_input("Number of Birds Seen/Struck", key="bh_num")
            new_size = st.selectbox("Size of Bird/Animal", ["Small","Medium","Large"], key="bh_size")
        with fe2:
            new_parts = st.text_area("Parts of Aircraft Affected", height=100, key="bh_parts")
            new_delay = st.text_input("Delay (if due to bird strike)", key="bh_delay")

        new_remarks = st.text_area("Remarks — describe damage, injuries, and other pertinent information", height=100, key="bh_remarks")

        if st.button("Submit Bird Hit Report", key="submit_bh"):
            st.success("✅ Bird hit report submitted. In production, this would save to your database and notify SQMS.")

        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# FSR PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "FSR":
    page_header("Flight Service Pre-Reports", "FSR · FLIGHT SERVICES DEPARTMENT · AIR SIAL", "Live Data")

    df = fsr_df.copy()
    total = len(df)
    factors = df["AFFECTED FACTORS"].value_counts() if "AFFECTED FACTORS" in df.columns else pd.Series()

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Total FSRs", total, C["alight"], "all periods")
    with k2: kpi("Medical", int(factors.get("MEDICAL",0)) + int(factors.get("MEDICAL EMERGENCY",0)), C["red"], "medical incidents", "linear-gradient(90deg,#ef4444,#f87171)")
    with k3: kpi("Disruptive Pax", int(factors.get("DISRUPTIVE PAX",0)), C["amber"], "disruptive passengers", "linear-gradient(90deg,#f59e0b,#fbbf24)")
    with k4: kpi("Engineering", int(factors.get("ENGINEERING",0)), C["cyan"], "engineering issues", "linear-gradient(90deg,#06b6d4,#67e8f9)")

    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        section("FSR Affected Factors — Frequency")
        if not factors.empty:
            top = factors.head(12)
            clrs = [C["red"] if "MED" in str(k).upper() else C["amber"] if "PAX" in str(k).upper()
                    else C["cyan"] if "ENG" in str(k).upper() else C["alight"] for k in top.index]
            fig = go.Figure(go.Bar(
                x=top.values, y=top.index, orientation="h",
                marker_color=clrs,
                hovertemplate="<b>%{y}</b><br>%{x} reports<extra></extra>",
            ))
            fig.update_layout(**base_layout(320))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with c2:
        section("Reports by Aircraft Registration")
        if "A/C REG" in df.columns:
            reg_cnt = df["A/C REG"].value_counts().head(10)
            fig2 = go.Figure(go.Bar(
                x=reg_cnt.index, y=reg_cnt.values,
                marker=dict(
                    color=reg_cnt.values,
                    colorscale=[[0,"#1e2235"],[1,"#818cf8"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x}</b><br>%{y} FSRs<extra></extra>",
            ))
            fig2.update_layout(**base_layout(320))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    section("FSR Log")
    display_cols = [c for c in ["DATE","A/C REG","FLIGHT NO.","AFFECTED FACTORS","INCIDENT","ACTION TAKEN"]
                    if c in df.columns]
    st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, hide_index=True)

    section("Submit New FSR")
    with st.expander("➕ Open Flight Service Pre-Report Form"):
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        st.markdown("<div class='form-section-title'>Flight Information</div>", unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            fsr_flt = st.text_input("Flight No.", placeholder="PF-717", key="fsr_flt")
            fsr_date = st.date_input("Date", key="fsr_date")
        with f2:
            fsr_reg = st.text_input("A/C REG", placeholder="AP-BOS", key="fsr_reg")
            fsr_lcc = st.text_input("LCC Name", key="fsr_lcc")
        with f3:
            fsr_actype = st.selectbox("A/C Type", ["A320","A321","ATR"], key="fsr_actype")
            fsr_sector = st.text_input("Sector / Time", placeholder="KHI-LHE / 08:30", key="fsr_sector")
        with f4:
            fsr_sno = st.text_input("S.No.", key="fsr_sno")

        st.markdown("<div class='form-section-title' style='margin-top:16px'>Affected Factors (tick all that apply)</div>", unsafe_allow_html=True)
        factors_list = ["Disruptive Pax","Delays","Hotel","Seating","Medical Emergency","Transport",
                        "Medical","Meal/Beverages","Logistics Items","Engineering","WCHS","Ramp",
                        "Scheduling","Medical Pouch","Boarding","Pax Complaint","Smoke","Others"]
        cols_fac = st.columns(4)
        selected_fac = []
        for i, fac in enumerate(factors_list):
            with cols_fac[i % 4]:
                if st.checkbox(fac, key=f"fsr_fac_{i}"):
                    selected_fac.append(fac)

        st.markdown("<div class='form-section-title' style='margin-top:16px'>Incident & Action</div>", unsafe_allow_html=True)
        fsr_incident = st.text_area("Incident Description", height=120, key="fsr_incident")
        fsr_action = st.text_area("Action Taken", height=80, key="fsr_action")
        fsr_sig = st.text_input("LCC Signature and AS Number", key="fsr_sig")

        if st.button("Submit FSR", key="submit_fsr"):
            st.success("✅ FSR submitted successfully.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# HIRA PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "HIRA":
    page_header("Hazard Identification & Risk Assessment", "HIRA · AS-CS-003 · CORPORATE SAFETY DEPT", "Live Data")

    df = hira_df.copy()
    total = len(df)
    risk_col = "int. risk rating" if "int. risk rating" in df.columns else None
    status_col = "status" if "status" in df.columns else None

    risk_counts = df[risk_col].value_counts() if risk_col else pd.Series()
    status_counts = df[status_col].value_counts() if status_col else pd.Series()

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Total Hazards", total, C["alight"], "all HIRA records")
    with k2: kpi("High Risk", int(risk_counts.get("HIGH",0)) + int(risk_counts.get("high",0)), C["red"], "intolerable", "linear-gradient(90deg,#ef4444,#f87171)")
    with k3: kpi("Medium Risk", int(risk_counts.get("medium",0)) + int(risk_counts.get("MEDIUM",0)), C["amber"], "tolerable", "linear-gradient(90deg,#f59e0b,#fbbf24)")
    with k4: kpi("Closed", int(status_counts.get("closed",0)), C["green"], "resolved hazards", "linear-gradient(90deg,#22c55e,#86efac)")

    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        section("Risk Rating Distribution")
        if risk_col and not risk_counts.empty:
            color_map = {"low":C["green"],"medium":C["amber"],"high":C["red"],
                         "LOW":C["green"],"MEDIUM":C["amber"],"HIGH":C["red"]}
            clrs = [color_map.get(str(k).strip(), C["alight"]) for k in risk_counts.index]
            fig = go.Figure(go.Bar(
                x=risk_counts.index.str.upper() if hasattr(risk_counts.index, "str") else risk_counts.index,
                y=risk_counts.values,
                marker_color=clrs,
                hovertemplate="<b>%{x}</b><br>%{y} hazards<extra></extra>",
            ))
            fig.update_layout(**base_layout(260))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with c2:
        section("Hazard Status Breakdown")
        if status_col and not status_counts.empty:
            sc = {C["green"] if "close" in str(k).lower() else C["amber"]: v
                  for k,v in status_counts.items()}
            sc_colors = [C["green"] if "close" in str(k).lower() else C["amber"]
                         for k in status_counts.index]
            fig2 = go.Figure(go.Pie(
                labels=[str(k).upper() for k in status_counts.index],
                values=status_counts.values, hole=0.6,
                marker=dict(colors=sc_colors, line=dict(color="#0c0e14", width=2)),
                hovertemplate="<b>%{label}</b><br>%{value}<br>%{percent}<extra></extra>",
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=260, margin=dict(l=8,r=8,t=8,b=8),
                font=dict(family="JetBrains Mono,monospace", color=C["muted"], size=10),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"])),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    section("HIRA Log")
    display_cols = [c for c in ["date of report","reporter name","department","location of hazard",
                                 "hazard description","int. risk assessment","int. risk rating",
                                 "corrective action plan","status"]
                    if c in df.columns]
    st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, hide_index=True)

    section("Submit New HIRA Report")
    with st.expander("➕ Open HIRA Form (AS-CS-003)"):
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        st.markdown("<div class='form-section-title'>Reporter Information</div>", unsafe_allow_html=True)
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            h_date = st.date_input("Date of Report", key="h_date")
            h_reporter = st.text_input("Reporter Name", key="h_reporter")
        with hc2:
            h_dept = st.selectbox("Department", ["Flight Operations","Engineering","Airport Services","Safety","Cabin Crew","Ground Handling","Other"], key="h_dept")
            h_report_no = st.text_input("Report No.", key="h_report_no")
        with hc3:
            h_location = st.text_input("Location of Hazard", placeholder="Ramp, Cockpit, Galley...", key="h_location")
            h_datetime_haz = st.text_input("Date/Time Hazard Identified", key="h_datetime_haz")

        st.markdown("<div class='form-section-title' style='margin-top:16px'>Hazard Description</div>", unsafe_allow_html=True)
        h_desc = st.text_area("Describe the hazard in detail", height=100, key="h_desc")

        st.markdown("<div class='form-section-title' style='margin-top:16px'>Initial Risk Assessment</div>", unsafe_allow_html=True)
        hm1, hm2 = st.columns(2)
        with hm1:
            h_prob = st.selectbox("Probability", ["A - Frequent","B - Occasional","C - Remote","D - Improbable","E - Extremely Improbable"], key="h_prob")
        with hm2:
            h_sev = st.selectbox("Severity", ["1 - Catastrophic","2 - Hazardous","3 - Major","4 - Minor","5 - Negligible"], key="h_sev")

        risk_matrix = {
            ("A - Frequent","1 - Catastrophic"):"1A - INTOLERABLE",
            ("A - Frequent","2 - Hazardous"):"2A - INTOLERABLE",
            ("B - Occasional","2 - Hazardous"):"2B - INTOLERABLE",
        }
        risk_key = (h_prob, h_sev)
        risk_label = risk_matrix.get(risk_key, "Review matrix for rating")
        if "INTOLERABLE" in risk_label:
            st.markdown(f"<div class='badge badge-red'>Risk Rating: {risk_label}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='badge badge-amber'>Risk Rating: {risk_label}</div>", unsafe_allow_html=True)

        st.markdown("<div class='form-section-title' style='margin-top:16px'>Root Cause & Corrective Action</div>", unsafe_allow_html=True)
        h_root = st.text_area("Root Cause Analysis", height=80, key="h_root")
        h_cap = st.text_area("Corrective Action Plan (CAP)", height=80, key="h_cap")

        hx1, hx2, hx3 = st.columns(3)
        with hx1:
            h_responsible = st.text_input("Responsible Person", key="h_responsible")
        with hx2:
            h_target = st.date_input("Target Date", key="h_target")
        with hx3:
            h_residual = st.selectbox("Residual Risk (After CAP)", ["Intolerable","Tolerable","Acceptable"], key="h_residual")

        h_remarks = st.text_area("Remarks by Snr. GMCS&S", height=60, key="h_remarks")

        if st.button("Submit HIRA Report", key="submit_hira"):
            st.success("✅ HIRA report submitted to Corporate Safety Department.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# MOR PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "MOR":
    page_header("Mandatory Occurrence Reports", "MOR · CAAF-001-SBXX-1.0 · SAFETY & INVESTIGATION BOARD", "Live Data")

    df = mor_df.copy()
    total = len(df)
    causes = df["NATURE AND CAUSE"].value_counts() if "NATURE AND CAUSE" in df.columns else pd.Series()
    damage = df["DAMAGE TO AIRCRAFT"].value_counts() if "DAMAGE TO AIRCRAFT" in df.columns else pd.Series()

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Total MORs", total, C["alight"], "mandatory reports")
    with k2: kpi("Aircraft Types", df["REGISTRATION"].nunique() if "REGISTRATION" in df.columns else 0, C["cyan"], "unique registrations", "linear-gradient(90deg,#06b6d4,#67e8f9)")
    with k3: kpi("Routes Affected", df["DESTINATION"].nunique() if "DESTINATION" in df.columns else 0, C["amber"], "unique destinations", "linear-gradient(90deg,#f59e0b,#fbbf24)")
    with k4: kpi("NIL Damage", int(damage.get("NIL",0)), C["green"], "no structural damage", "linear-gradient(90deg,#22c55e,#86efac)")

    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        section("MOR by Nature & Cause")
        if not causes.empty:
            top = causes.head(10)
            fig = go.Figure(go.Bar(
                x=top.values, y=[str(k)[:35]+"…" if len(str(k))>35 else str(k) for k in top.index],
                orientation="h",
                marker=dict(
                    color=top.values,
                    colorscale=[[0,"#1e2235"],[0.5,"#ef4444"],[1,"#f87171"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{y}</b><br>%{x} occurrences<extra></extra>",
            ))
            fig.update_layout(**base_layout(300))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with c2:
        section("MOR Distribution by Registration")
        if "REGISTRATION" in df.columns:
            reg = df["REGISTRATION"].value_counts().head(10)
            fig2 = go.Figure(go.Bar(
                x=reg.index, y=reg.values,
                marker=dict(
                    color=reg.values,
                    colorscale=[[0,"#1e2235"],[1,"#ef4444"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x}</b><br>%{y} MORs<extra></extra>",
            ))
            fig2.update_layout(**base_layout(300))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    section("MOR Log")
    display_cols = [c for c in ["DATE","FLIGHT NO.","REGISTRATION","REPORTER","ORIGIN","DESTINATION",
                                 "NATURE AND CAUSE","DAMAGE TO AIRCRAFT","WEATHER","SUMMARY"]
                    if c in df.columns]
    st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, hide_index=True)

    section("Submit New MOR")
    with st.expander("➕ Open MOR Form (CAAF-001-SBXX-1.0)"):
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        st.markdown("<div class='form-section-title'>Reporter & Airport</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            m_reporter_name = st.text_input("Name of Reporting Officer", key="m_rname")
            m_designation = st.text_input("Designation", key="m_desig")
            m_telephone = st.text_input("Telephone", key="m_tel")
        with m2:
            m_airport = st.text_input("Airport of Origin", placeholder="OPKC", key="m_airport")
            m_mor_no = st.text_input("MOR No.", key="m_mornum")
            m_year = st.text_input("Year", value=str(datetime.now().year), key="m_year")
        with m3:
            m_ac_type = st.selectbox("Aircraft Type", ["A320","A321","ATR-72","Other"], key="m_actype")
            m_date = st.date_input("Date of Occurrence", key="m_date")

        st.markdown("<div class='form-section-title' style='margin-top:16px'>Aircraft Information</div>", unsafe_allow_html=True)
        ma1, ma2 = st.columns(2)
        with ma1:
            m_ac_reg = st.text_input("Aircraft Registration & Flight No.", placeholder="AP-BOA / PF-714", key="m_acreg")
            m_operator = st.text_input("Owner / Operator", value="Air Sial", key="m_op")
            m_pilot = st.text_input("Name of Pilot", key="m_pilot")
        with ma2:
            m_dep = st.text_input("Last Point of Departure", placeholder="OPLA", key="m_dep")
            m_dest = st.text_input("Point of Intended Landing", placeholder="OERK", key="m_dest")
            m_nature_flight = st.text_input("Nature of Flight", placeholder="Scheduled Passenger", key="m_nat")

        st.markdown("<div class='form-section-title' style='margin-top:16px'>Occurrence Details</div>", unsafe_allow_html=True)
        mb1, mb2 = st.columns(2)
        with mb1:
            m_location = st.text_input("Location of Occurrence (geographical reference)", key="m_loc")
            m_persons = st.text_input("Total persons on board (pax + crew)", key="m_pers")
            m_killed = st.text_input("Number killed (if any)", key="m_kill")
        with mb2:
            m_injured = st.text_input("Number seriously injured", key="m_inj")
            m_weather = st.text_input("Weather Condition", key="m_wx")
            m_damage = st.text_area("Nature and extent of damage to A/C", height=60, key="m_dmg")

        m_cause = st.text_area("Nature and cause of incident/accident (as known)", height=80, key="m_cause")
        m_summary = st.text_area("Summary of the Occurrence", height=120, key="m_sum")
        m_dg = st.text_area("Presence and description of dangerous goods on board", height=60, key="m_dg")

        st.markdown("<div class='form-section-title' style='margin-top:16px'>Distribution</div>", unsafe_allow_html=True)
        st.markdown("This MOR will be distributed to: **PD (Reg) HQ CAA** · **President SIB HQ CAA** · **Dir SQMS** · **SO to DDG CAA**", unsafe_allow_html=False)
        mc1, mc2 = st.columns(2)
        with mc1:
            m_sig = st.text_input("Signature / Name", key="m_sig")
        with mc2:
            m_desig2 = st.text_input("Designation", key="m_desig2")

        if st.button("Submit MOR", key="submit_mor"):
            st.success("✅ MOR submitted. Report transmitted to PSIB, PD (Reg), and Dir SQMS.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# AI INSIGHTS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "AI Insights":
    page_header("AI Safety Intelligence", "POWERED BY CLAUDE · REAL OPERATIONAL DATA")

    ai_left, ai_right = st.columns([1.2, 1])

    with ai_left:
        section("Executive AI Summary")
        if st.button("⚡ Generate GM Briefing", key="btn_summary"):
            with st.spinner("Analysing operational safety data..."):
                st.session_state.last_ai_summary = ai_summary()

        if st.session_state.last_ai_summary:
            st.markdown(f"""<div class='ai-box'>
                <div class='ai-box-hdr'>🛡️ GM Executive Briefing — {st.session_state.org_name}</div>
                {st.session_state.last_ai_summary}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class='ai-box' style='opacity:0.35;text-align:center;padding:36px'>
                Click "Generate GM Briefing" for an AI-powered analysis of your operational data.
            </div>""", unsafe_allow_html=True)

        section("Data Highlights")
        ctx = build_context()
        cols = st.columns(2)
        items = [
            ("Bird Hits", str(ctx.get("bird_hits_total","—")), C["amber"]),
            ("FSR Events", str(ctx.get("fsr_total","—")), C["alight"]),
            ("HIRA Reports", str(ctx.get("hira_total","—")), C["cyan"]),
            ("MORs Filed", str(ctx.get("mor_total","—")), C["red"]),
        ]
        for i,(label, val, col) in enumerate(items):
            with cols[i%2]:
                st.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-label'>{label}</div>
                    <div class='kpi-value' style='color:{col}'>{val}</div>
                </div>""", unsafe_allow_html=True)

    with ai_right:
        section("Chat with SafetyOS AI")
        if not st.session_state.chat_history:
            st.markdown("""<div class='ai-box' style='text-align:center;padding:28px;opacity:0.5'>
                Ask anything about your safety data.<br><br>
                <em>"Which aircraft has the most MORs?"</em><br>
                <em>"What's the top bird strike risk area?"</em><br>
                <em>"What FSR factor needs urgent attention?"</em>
            </div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history[-10:]:
                is_ai = msg["role"] == "assistant"
                color = C["alight"] if is_ai else C["green"]
                role = "🤖 SafetyOS AI" if is_ai else "👤 You"
                bg = "#141726" if is_ai else "#111420"
                st.markdown(f"""<div class='chat-msg' style='background:{bg};border:1px solid #1e2235'>
                    <div class='chat-role' style='color:{color}'>{role}</div>
                    <div class='chat-text'>{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#3d4460;text-transform:uppercase;letter-spacing:0.12em;margin:12px 0 6px'>Quick Prompts</div>", unsafe_allow_html=True)
        qp_col1, qp_col2 = st.columns(2)
        QUICK = [
            "Top bird strike risk areas?",
            "Most critical MOR causes?",
            "FSR medical trend?",
            "Highest priority HIRA action?",
        ]
        for i, prompt in enumerate(QUICK):
            with (qp_col1 if i % 2 == 0 else qp_col2):
                if st.button(prompt, key=f"qp_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":prompt})
                    with st.spinner(""):
                        reply = ai_chat(prompt)
                    st.session_state.chat_history.append({"role":"assistant","content":reply})
                    st.rerun()

        user_input = st.text_input("Ask about your safety data...", key="chat_input", label_visibility="collapsed")
        sc1, sc2 = st.columns([2,1])
        with sc1:
            if st.button("Send →", key="send_btn"):
                if user_input.strip():
                    st.session_state.chat_history.append({"role":"user","content":user_input})
                    with st.spinner(""):
                        reply = ai_chat(user_input)
                    st.session_state.chat_history.append({"role":"assistant","content":reply})
                    st.rerun()
        with sc2:
            if st.session_state.chat_history:
                if st.button("🗑 Clear", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.rerun()
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS PAGE — Enhanced
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Settings":
    page_header("Settings & Configuration", "SAFETYOS PLATFORM · ORGANISATION PREFERENCES")

    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Organisation", "🤖 AI & Integration", "🎯 KPI Targets", "📂 Data Management"])

    with tab1:
        section("Organisation Profile")
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            new_org = st.text_input("Organisation Name", value=st.session_state.org_name)
            if new_org != st.session_state.org_name:
                st.session_state.org_name = new_org
            st.text_input("IATA Code", value="ER", placeholder="ER")
            st.text_input("CAA Licence No.", placeholder="PKA-AOC-2019-01")
        with s2:
            new_period = st.text_input("Report Period", value=st.session_state.report_period)
            if new_period != st.session_state.report_period:
                st.session_state.report_period = new_period
            st.selectbox("Primary Fleet Type", ["Airbus A320", "Boeing 737", "ATR-72", "Mixed Fleet"], index=0)
            st.text_input("Safety Email", value="safety@airsial.com")
        st.markdown("</div>", unsafe_allow_html=True)

        section("Regulatory Contacts")
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        with rc1:
            st.text_input("PSIB Tel", value="92-51-4472750")
            st.text_input("PSIB Cell", value="0300-8250472")
            st.text_input("PSIB Fax", value="92-21-34604305")
        with rc2:
            st.text_input("SQMS Email", value="jtd.sqms@caapakistan.com.pk")
            st.text_input("AAIB Email", value="paaib@caapakistan.com.pk")
            st.text_input("APS Email", value="director.aps@caapakistan.com.pk")
        st.markdown("</div>", unsafe_allow_html=True)

        section("Reporting Deadlines")
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        rd1, rd2, rd3 = st.columns(3)
        with rd1: st.number_input("Bird Hit Form Submission (hours)", value=24, min_value=1, max_value=72)
        with rd2: st.number_input("Bird Hit SQMS Report (hours)", value=48, min_value=1, max_value=168)
        with rd3: st.number_input("MOR Telephone Notification (minutes)", value=60, min_value=5, max_value=240)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        section("AI Configuration")
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            new_key = st.text_input("Anthropic API Key", value=st.session_state.api_key,
                type="password", placeholder="sk-ant-...",
                help="Get your key at https://console.anthropic.com")
            if new_key != st.session_state.api_key:
                st.session_state.api_key = new_key
        with a2:
            st.selectbox("AI Model", ["claude-sonnet-4-20250514 (Recommended)", "claude-opus-4-20250514 (Advanced)"])

        connected = bool(st.session_state.api_key.strip())
        if connected:
            st.success("✅ API Key configured — all AI features active.")
        else:
            st.warning("⚠️ No API key. AI summary, chat, and insights require an Anthropic key. Get one at console.anthropic.com")
        st.markdown("</div>", unsafe_allow_html=True)

        section("Notification Settings")
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        n1, n2 = st.columns(2)
        with n1:
            st.checkbox("Email alerts for new MOR submissions", value=True)
            st.checkbox("Email alerts for High Risk HIRA reports", value=True)
            st.checkbox("Daily summary report email", value=False)
        with n2:
            st.checkbox("Notify on bird strike with engine damage", value=True)
            st.checkbox("Alert on overdue HIRA CAP deadlines", value=True)
            st.checkbox("Weekly AI digest email", value=False)
        st.text_input("Alert Recipients (comma-separated emails)", placeholder="safety@airsial.com, gm@airsial.com")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        section("Safety KPI Targets")
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        with t1:
            new_audit = st.number_input("Audit Closure Target (%)", min_value=0, max_value=100,
                value=st.session_state.audit_target)
            st.session_state.audit_target = new_audit
            st.number_input("Max Bird Hits per Quarter", min_value=0, max_value=100, value=10)
        with t2:
            new_high = st.number_input("Max High Severity Events", min_value=0, max_value=50,
                value=st.session_state.max_high_sev)
            st.session_state.max_high_sev = new_high
            st.number_input("Min Proactive HIRA Reports / Month", min_value=0, max_value=200, value=5)
        with t3:
            st.number_input("Max Open MORs at Month End", min_value=0, max_value=50, value=3)
            st.number_input("FSR Submission Threshold (hours)", min_value=0, max_value=72, value=24)
        st.markdown("</div>", unsafe_allow_html=True)

        section("Risk Matrix Thresholds")
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        rm1, rm2, rm3 = st.columns(3)
        with rm1:
            st.markdown("🔴 **Intolerable** — Immediate action", unsafe_allow_html=False)
            st.text_input("Intolerable codes", value="1A, 2A, 1B, 2B, 1C, 1D, 1E", key="rm_intol")
        with rm2:
            st.markdown("🟡 **Tolerable** — Management decision", unsafe_allow_html=False)
            st.text_input("Tolerable codes", value="3A, 3B, 2C, 2D, 3C, 4A", key="rm_tol")
        with rm3:
            st.markdown("🟢 **Acceptable** — Monitor", unsafe_allow_html=False)
            st.text_input("Acceptable codes", value="4B, 4C, 4D, 5A-5E, 3D, 3E", key="rm_acc")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        section("Data Sources")
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Loaded Data Files**")
            for name, df_local in [("Bird Hits", bird_df), ("FSR", fsr_df), ("HIRA", hira_df), ("MOR", mor_df)]:
                rows = len(df_local)
                badge = "badge-green" if rows > 0 else "badge-red"
                st.markdown(f"<span class='badge {badge}'>{name}: {rows} records</span>&nbsp;", unsafe_allow_html=True)
        with d2:
            st.markdown("**Export Data**")
            for name, df_local in [("Bird Hits", bird_df), ("FSR", fsr_df), ("HIRA", hira_df), ("MOR", mor_df)]:
                if not df_local.empty:
                    csv = df_local.to_csv(index=False).encode("utf-8")
                    st.download_button(f"⬇ Download {name} CSV", csv, f"{name.lower().replace(' ','_')}.csv", "text/csv", key=f"dl_{name}")
        st.markdown("</div>", unsafe_allow_html=True)

        section("Danger Zone")
        st.markdown("<div class='form-section' style='border-color:rgba(239,68,68,0.3)'>", unsafe_allow_html=True)
        dz1, dz2 = st.columns(2)
        with dz1:
            if st.button("🗑 Clear Chat History", key="clear_all_chat"):
                st.session_state.chat_history = []
                st.session_state.last_ai_summary = None
                st.success("✅ Chat history cleared.")
                st.rerun()
        with dz2:
            if st.button("♻️ Reload Data from Files", key="reload_data"):
                st.cache_data.clear()
                st.success("✅ Data cache cleared. Reload the page to refresh.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW PAGE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='page-hdr' style='display:flex;align-items:flex-start;justify-content:space-between'>
    <div>
        <div class='page-title'>{st.session_state.org_name} Safety Dashboard</div>
        <div class='page-subtitle'>EXECUTIVE OVERVIEW · {st.session_state.report_period} · OPERATIONAL DATA</div>
    </div>
    <div>
        <span class='badge badge-green'>● LIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPIs
k1, k2, k3, k4 = st.columns(4)
with k1: kpi("Bird Strikes", len(bird_df), C["amber"], f"total recorded events", "linear-gradient(90deg,#f59e0b,#fbbf24)")
with k2: kpi("FSR Events", len(fsr_df), C["alight"], "flight service reports", "linear-gradient(90deg,#4f6ef7,#818cf8)")
with k3: kpi("HIRA Reports", len(hira_df), C["cyan"], "hazard identifications", "linear-gradient(90deg,#06b6d4,#67e8f9)")
with k4: kpi("MORs Filed", len(mor_df), C["red"], "mandatory occurrences", "linear-gradient(90deg,#ef4444,#f87171)")

st.markdown("<div style='margin:20px 0'></div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    section("Bird Strikes by Phase of Flight")
    if not bird_df.empty and "PHASE OF FLIGHT" in bird_df.columns:
        p = bird_df["PHASE OF FLIGHT"].value_counts().head(8)
        fig = go.Figure(go.Bar(
            x=p.values, y=p.index, orientation="h",
            marker=dict(
                color=p.values,
                colorscale=[[0,"#1e2235"],[1,"#f59e0b"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>%{x} strikes<extra></extra>",
        ))
        fig.update_layout(**base_layout(260))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

with c2:
    section("FSR — Top Affected Factors")
    if not fsr_df.empty and "AFFECTED FACTORS" in fsr_df.columns:
        f = fsr_df["AFFECTED FACTORS"].value_counts().head(8)
        clrs = [C["red"] if "MED" in str(k).upper() else C["amber"] if "PAX" in str(k).upper()
                else C["alight"] for k in f.index]
        fig2 = go.Figure(go.Bar(
            x=f.values, y=f.index, orientation="h",
            marker_color=clrs,
            hovertemplate="<b>%{y}</b><br>%{x} reports<extra></extra>",
        ))
        fig2.update_layout(**base_layout(260))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

c3, c4 = st.columns(2)

with c3:
    section("HIRA — Risk Rating Distribution")
    if not hira_df.empty and "int. risk rating" in hira_df.columns:
        r = hira_df["int. risk rating"].dropna().str.lower().value_counts()
        color_map = {"low": C["green"], "medium": C["amber"], "high": C["red"]}
        clrs = [color_map.get(str(k).strip(), C["alight"]) for k in r.index]
        fig3 = go.Figure(go.Pie(
            labels=[str(k).upper() for k in r.index], values=r.values,
            hole=0.55, marker=dict(colors=clrs, line=dict(color="#0c0e14", width=2)),
            hovertemplate="<b>%{label}</b><br>%{value}<br>%{percent}<extra></extra>",
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(l=8,r=8,t=8,b=8),
            font=dict(family="JetBrains Mono,monospace", color=C["muted"], size=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"])),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

with c4:
    section("MOR — Occurrence Types")
    if not mor_df.empty and "NATURE AND CAUSE" in mor_df.columns:
        m = mor_df["NATURE AND CAUSE"].value_counts().head(6)
        fig4 = go.Figure(go.Bar(
            x=[str(k)[:28]+"…" if len(str(k))>28 else str(k) for k in m.index],
            y=m.values,
            marker=dict(
                color=m.values,
                colorscale=[[0,"#1e2235"],[1,"#ef4444"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b><br>%{y} MORs<extra></extra>",
        ))
        fig4.update_layout(**base_layout(260))
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar":False})

section("Recent Activity — Across All Reports")
recent_rows = []
if not mor_df.empty:
    for _, row in mor_df.head(4).iterrows():
        recent_rows.append({
            "Report Type": "🔴 MOR",
            "Date": str(row.get("DATE",""))[:10],
            "Flight / Reg": f"{row.get('FLIGHT NO.','')} · {row.get('REGISTRATION','')}",
            "Details": str(row.get("NATURE AND CAUSE",""))[:60],
            "Route": f"{row.get('ORIGIN','')} → {row.get('DESTINATION','')}",
        })
if not fsr_df.empty:
    for _, row in fsr_df.head(3).iterrows():
        recent_rows.append({
            "Report Type": "🟡 FSR",
            "Date": str(row.get("DATE",""))[:10],
            "Flight / Reg": f"{row.get('FLIGHT NO.','')} · {row.get('A/C REG','')}",
            "Details": str(row.get("INCIDENT",""))[:60],
            "Route": str(row.get("AFFECTED FACTORS",""))[:30],
        })
if recent_rows:
    st.dataframe(pd.DataFrame(recent_rows), use_container_width=True, hide_index=True)
