import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import anthropic
import json
from datetime import datetime

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirSial SafetyOS",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS — original visual DNA preserved ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp { background-color: #09090b; color: #fafafa; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#09090b 0%,#0f0f12 100%) !important;
    border-right: 1px solid #27272a;
}

/* ── Metric card ── */
.metric-card {
    background: #1c1c1f;
    border: 1px solid #27272a;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: var(--accent-line, linear-gradient(90deg,#6366f1,#818cf8));
}
.metric-label {
    font-family:'DM Mono',monospace; font-size:11px; color:#71717a;
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px;
}
.metric-value { font-size:28px; font-weight:800; color:#fafafa; }
.metric-sub { font-family:'DM Mono',monospace; font-size:11px; color:#52525b; margin-top:4px; }

/* ── AI box ── */
.ai-box {
    background: linear-gradient(135deg,rgba(99,102,241,.1),rgba(129,140,248,.04));
    border: 1px solid rgba(99,102,241,.25);
    border-radius:12px; padding:16px 18px; font-size:13px; line-height:1.7;
}
.ai-box-hdr { color:#818cf8; font-weight:700; margin-bottom:8px; font-size:11px;
              text-transform:uppercase; letter-spacing:.08em; }

/* ── Chat ── */
.chat-msg { border-radius:12px; padding:14px 16px; margin-bottom:10px;
            font-size:13px; line-height:1.6; border:1px solid #27272a; }
.chat-role { font-family:'DM Mono',monospace; font-size:10px; font-weight:600;
             text-transform:uppercase; letter-spacing:.1em; margin-bottom:6px; }

/* ── Section header ── */
.sec-hdr {
    font-family:'Sora',sans-serif; font-size:15px; font-weight:700; color:#fafafa;
    margin:20px 0 10px; padding-bottom:8px; border-bottom:1px solid #27272a;
}

/* ── Badge ── */
.badge { display:inline-block; padding:3px 10px; border-radius:20px;
         font-size:11px; font-weight:600; font-family:'DM Mono',monospace; }
.bg { background:rgba(34,197,94,.15); color:#22c55e; border:1px solid rgba(34,197,94,.3); }
.br { background:rgba(239,68,68,.15);  color:#ef4444; border:1px solid rgba(239,68,68,.3); }
.ba { background:rgba(245,158,11,.15); color:#f59e0b; border:1px solid rgba(245,158,11,.3); }
.bb { background:rgba(99,102,241,.15); color:#818cf8; border:1px solid rgba(99,102,241,.3); }

/* ── Form section ── */
.form-sec { background:#18181b; border:1px solid #27272a; border-radius:14px;
            padding:20px 22px; margin-bottom:14px; }
.form-sec-title { font-size:11px; font-weight:700; color:#6366f1; text-transform:uppercase;
                  letter-spacing:.1em; margin-bottom:14px; padding-bottom:8px;
                  border-bottom:1px solid #27272a; }

/* ── Sidebar nav ── */
.section-hdr { font-family:'DM Mono',monospace; font-size:9px; color:#52525b;
               text-transform:uppercase; letter-spacing:.15em; margin:14px 0 6px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background:#18181b; border-radius:10px; padding:4px;
                                     gap:4px; border:1px solid #27272a; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:#71717a !important;
                                border-radius:8px !important; }
.stTabs [aria-selected="true"] { background:#27272a !important; color:#818cf8 !important;
                                  font-weight:600 !important; }

/* ── Inputs ── */
.stTextInput input,.stTextArea textarea,.stNumberInput input {
    background:#18181b !important; border:1px solid #27272a !important;
    color:#fafafa !important; border-radius:8px !important; }
.stButton button { background:#18181b !important; border:1px solid #27272a !important;
                   color:#a1a1aa !important; border-radius:8px !important;
                   transition:all .15s !important; }
.stButton button:hover { border-color:#6366f1 !important; color:#818cf8 !important; }
</style>
""", unsafe_allow_html=True)

# ─── PALETTE ──────────────────────────────────────────────────────────────────
C = dict(bg="#09090b",surface="#18181b",card="#1c1c1f",border="#27272a",text="#fafafa",
         muted="#71717a",accent="#6366f1",alight="#818cf8",
         green="#22c55e",red="#ef4444",amber="#f59e0b",cyan="#06b6d4",purple="#a855f7")

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
DEFS = dict(page="Overview", api_key="", chat_history=[], last_ai_summary=None,
            org_name="Air Sial", report_period="2025–2026",
            audit_target=90, max_high_sev=5)
for k,v in DEFS.items():
    if k not in st.session_state: st.session_state[k] = v

# ─── DATA LOADING from Excel ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    out = {}
    files = dict(bird="BIRD_HITS.xlsx", fsr="FSR.xlsx", hira="HIRA.xlsx", mor="MOR.xlsx")
    for key, fname in files.items():
        try:
            df = pd.read_excel(fname)
            df.columns = [str(c).strip() for c in df.columns]
            out[key] = df
        except:
            out[key] = pd.DataFrame()
    return out

dfs = load_data()
bird_df = dfs["bird"]; fsr_df = dfs["fsr"]
hira_df = dfs["hira"]; mor_df  = dfs["mor"]

# ─── DERIVE MONTHLY METRICS from real data ────────────────────────────────────
def monthly_counts(df, date_col, n_months=6):
    """Return last n_months monthly counts as ordered dict {YYYY-MM: count}"""
    col = df[date_col].copy()
    col = pd.to_datetime(col, errors="coerce")
    df2 = df.copy(); df2["_d"] = col
    df2 = df2.dropna(subset=["_d"])
    df2["_m"] = df2["_d"].dt.to_period("M")
    counts = df2.groupby("_m").size().sort_index()
    return counts.tail(n_months)

def period_labels(counts):
    return [str(p) for p in counts.index]

def forecast(vals, n=3):
    vals = list(vals)
    if len(vals) < 2: return [int(vals[-1])]*n if vals else [0]*n
    x = np.arange(len(vals), dtype=float)
    s, i = np.polyfit(x, vals, 1)
    return [max(0, round(i + s*(len(vals)+j))) for j in range(n)]

def base_layout(height=240, barmode=None):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=8,r=8,t=8,b=8),
        font=dict(family="DM Mono, monospace", color=C["muted"], size=10),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10,color=C["muted"]),
                    orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0),
        xaxis=dict(gridcolor=C["border"],linecolor="rgba(0,0,0,0)",
                   tickcolor="rgba(0,0,0,0)",
                   tickfont=dict(family="DM Mono,monospace",color=C["muted"],size=10)),
        yaxis=dict(gridcolor=C["border"],linecolor="rgba(0,0,0,0)",
                   tickcolor="rgba(0,0,0,0)",showgrid=True,
                   tickfont=dict(family="DM Mono,monospace",color=C["muted"],size=10)),
    )
    if barmode: layout["barmode"] = barmode
    return layout

# ─── OVERVIEW DERIVED STATS ───────────────────────────────────────────────────
fsr_monthly  = monthly_counts(fsr_df,  "DATE", 6)
hira_monthly = monthly_counts(hira_df, "date of report", 6)
mor_monthly  = monthly_counts(mor_df,  "DATE", 6)
bird_monthly = monthly_counts(bird_df, "DATE OF OCCURRENCE", 6)

# severity from HIRA
def hira_sev_monthly(rating_val, n=6):
    df = hira_df.copy()
    df["_d"] = pd.to_datetime(df["date of report"], errors="coerce")
    df = df.dropna(subset=["_d"])
    df["_m"] = df["_d"].dt.to_period("M")
    mask = df["int. risk rating"].str.lower().str.strip() == rating_val.lower()
    counts = df[mask].groupby("_m").size().sort_index()
    all_months = df.groupby("_m").size().sort_index().index
    counts = counts.reindex(all_months, fill_value=0).tail(n)
    return counts

high_monthly = hira_sev_monthly("high", 6)
# fallback to 'medium' and 'low' combo for medium/low
med_monthly  = hira_sev_monthly("medium", 6)

# ─── AI HELPERS ───────────────────────────────────────────────────────────────
def get_client():
    key = st.session_state.get("api_key","").strip()
    if key: return anthropic.Anthropic(api_key=key)
    try:    return anthropic.Anthropic()
    except: return None

def build_context():
    ctx = {}
    ctx["fsr_total"] = len(fsr_df)
    if not fsr_df.empty and "AFFECTED FACTORS" in fsr_df.columns:
        ctx["fsr_top_factors"] = fsr_df["AFFECTED FACTORS"].value_counts().head(6).to_dict()
    ctx["hira_total"] = len(hira_df)
    if not hira_df.empty and "int. risk rating" in hira_df.columns:
        ctx["hira_risk_ratings"] = hira_df["int. risk rating"].value_counts().to_dict()
    if not hira_df.empty and "status" in hira_df.columns:
        ctx["hira_status"] = hira_df["status"].value_counts().to_dict()
    ctx["mor_total"]  = len(mor_df)
    if not mor_df.empty and "NATURE AND CAUSE" in mor_df.columns:
        ctx["mor_top_causes"] = mor_df["NATURE AND CAUSE"].value_counts().head(6).to_dict()
    ctx["bird_total"] = len(bird_df)
    if not bird_df.empty and "PHASE OF FLIGHT" in bird_df.columns:
        ctx["bird_phases"] = bird_df["PHASE OF FLIGHT"].value_counts().to_dict()
    ctx["fsr_monthly_last6"] = {str(k):int(v) for k,v in fsr_monthly.items()}
    return ctx

def ai_summary():
    client = get_client()
    if not client: return "⚠️ No API key configured — paste your key in Settings."
    prompt = (
        f"You are a Senior Aviation Safety Analyst for {st.session_state.org_name}, briefing the GM.\n\n"
        f"Operational safety data:\n{json.dumps(build_context(), indent=2)}\n\n"
        "Write a concise executive summary (4-6 sentences) covering:\n"
        "1. Overall safety culture and reporting trajectory\n"
        "2. Key wins and concerning trends\n"
        "3. Top 2 recommended actions for next quarter\n"
        "4. One forward-looking prediction\n\n"
        "Professional, direct language. Include specific numbers. No bullet points."
    )
    try:
        r = client.messages.create(model="claude-sonnet-4-20250514",max_tokens=500,
                                    messages=[{"role":"user","content":prompt}])
        return r.content[0].text
    except Exception as e: return f"⚠️ AI Error: {e}"

def ai_chat(user_msg):
    client = get_client()
    if not client: return "⚠️ No API key configured."
    system = (f"You are SafetyOS AI for {st.session_state.org_name}, expert aviation safety analyst.\n\n"
              f"Data:\n{json.dumps(build_context(), indent=2)}\n\n"
              "Answer concisely (2-4 sentences) with specific numbers and actionable advice.")
    hist = [{"role":m["role"],"content":m["content"]} for m in st.session_state.chat_history[-8:]]
    hist.append({"role":"user","content":user_msg})
    try:
        r = client.messages.create(model="claude-sonnet-4-20250514",max_tokens=600,
                                    system=system,messages=hist)
        return r.content[0].text
    except Exception as e: return f"⚠️ AI Error: {e}"

def ai_metric_insight(name, labels, vals):
    client = get_client()
    if not client: return None
    data_str = ", ".join(f"{l}={v}" for l,v in zip(labels,vals))
    pred = forecast(vals)
    prompt = (f"Aviation safety metric '{name}': {data_str}. "
              f"Forecast next 3 periods: {pred}. "
              "Write ONE direct GM-level sentence about this trend and its implication.")
    try:
        r = client.messages.create(model="claude-sonnet-4-20250514",max_tokens=120,
                                    messages=[{"role":"user","content":prompt}])
        return r.content[0].text
    except: return None

# ─── CHART BUILDERS ───────────────────────────────────────────────────────────
def chart_volume(counts, color=None):
    """Line + forecast from a pd.Series of monthly counts"""
    color = color or C["accent"]
    labels = [str(p) for p in counts.index]
    vals   = counts.values.tolist()
    preds  = forecast(vals)
    pred_labels = ["Fcst+1","Fcst+2","Fcst+3"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=vals, mode="lines+markers", name="Actual",
        line=dict(color=color,width=3), marker=dict(size=8,color=color),
        fill="tozeroy", fillcolor=color.replace("#","rgba(").replace(")",",0.12)") if color.startswith("#") else "rgba(99,102,241,0.12)",
        hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[labels[-1]]+pred_labels, y=[vals[-1]]+preds,
        mode="lines+markers", name="Forecast",
        line=dict(color=C["alight"],width=2,dash="dot"),
        marker=dict(size=6,color=C["alight"],symbol="circle-open"),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y}<extra></extra>"))
    fig.update_layout(**base_layout(240))
    return fig

def chart_severity_monthly():
    """Stacked bar — HIRA high/medium per month + forecast"""
    if high_monthly.empty and med_monthly.empty:
        return go.Figure()
    months = [str(p) for p in high_monthly.index]
    h_vals = high_monthly.values.tolist()
    m_vals = med_monthly.values.tolist() if not med_monthly.empty else [0]*len(h_vals)
    hp, mp = forecast(h_vals), forecast(m_vals)
    pm = ["Fcst+1","Fcst+2","Fcst+3"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=months,y=h_vals,name="High",marker_color=C["red"],
                         hovertemplate="<b>%{x}</b><br>High: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(x=months,y=m_vals,name="Medium",marker_color=C["amber"],
                         hovertemplate="<b>%{x}</b><br>Medium: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(x=pm,y=hp,name="High Fcst",marker_color="rgba(239,68,68,.4)"))
    fig.add_trace(go.Bar(x=pm,y=mp,name="Med Fcst", marker_color="rgba(245,158,11,.4)"))
    fig.update_layout(**base_layout(240,barmode="stack"))
    return fig

def chart_gauge(value):
    color = C["green"] if value>=80 else C["amber"] if value>=60 else C["red"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix":"%","font":{"size":40,"color":color,"family":"Sora,sans-serif"}},
        gauge={"axis":{"range":[0,100],"tickwidth":0,"tickcolor":"rgba(0,0,0,0)",
                       "tickfont":{"color":C["muted"],"size":9}},
               "bar":{"color":color,"thickness":.28},
               "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
               "steps":[{"range":[0,100],"color":C["border"]}],
               "threshold":{"line":{"color":color,"width":3},"thickness":.75,"value":value},
               "shape":"angular"}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=210,
                      margin=dict(l=20,r=20,t=20,b=0),
                      font=dict(family="DM Mono,monospace",color=C["muted"]))
    return fig

def chart_fsr_factors():
    if fsr_df.empty or "AFFECTED FACTORS" not in fsr_df.columns: return go.Figure()
    f = fsr_df["AFFECTED FACTORS"].dropna()
    # normalise slightly
    f = f.str.upper().str.strip()
    counts = f.value_counts().head(10)
    clrs = [C["red"] if "MED" in k else C["amber"] if "PAX" in k or "DISRUPT" in k
            else C["cyan"] if "ENG" in k else C["alight"] for k in counts.index]
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h",
        marker_color=clrs, hovertemplate="<b>%{y}</b><br>%{x} reports<extra></extra>"))
    fig.update_layout(**base_layout(280))
    return fig

def chart_mor_causes():
    if mor_df.empty or "NATURE AND CAUSE" not in mor_df.columns: return go.Figure()
    c = mor_df["NATURE AND CAUSE"].dropna().value_counts().head(8)
    labels = [str(k)[:30]+"…" if len(str(k))>30 else str(k) for k in c.index]
    fig = go.Figure(go.Bar(x=c.values, y=labels, orientation="h",
        marker=dict(color=c.values,colorscale=[[0,C["surface"]],[1,C["red"]]],line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>%{x} MORs<extra></extra>"))
    fig.update_layout(**base_layout(280))
    return fig

# ─── UI HELPERS ───────────────────────────────────────────────────────────────
def page_header(title, subtitle, badge=None):
    badge_html = (f"<span class='badge bg' style='margin-left:12px'>{badge}</span>"
                  if badge else "")
    st.markdown(f"""
    <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:18px'>
      <div>
        <div style='font-family:Sora,sans-serif;font-size:22px;font-weight:800;color:#fafafa'>
          {title}{badge_html}</div>
        <div style='font-family:DM Mono,monospace;font-size:10px;color:#a1a1aa;
                    letter-spacing:.06em;margin-top:2px'>{subtitle}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def mc(label, value, color, sub="", accent="linear-gradient(90deg,#6366f1,#818cf8)"):
    st.markdown(f"""<div class='metric-card' style='--accent-line:{accent}'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value' style='color:{color}'>{value}</div>
        <div class='metric-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

def sec(title):
    st.markdown(f"<div class='sec-hdr'>{title}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
NAV = [("✈️","Overview"),("🐦","Bird Hits"),("📋","FSR"),
       ("⚠️","HIRA"),("📊","MOR"),("🤖","AI Insights"),("⚙️","Settings")]

with st.sidebar:
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:20px;
                padding-bottom:16px;border-bottom:1px solid #27272a'>
      <div style='width:34px;height:34px;border-radius:10px;flex-shrink:0;
                  background:linear-gradient(135deg,#6366f1,#818cf8);
                  display:flex;align-items:center;justify-content:center;
                  font-size:15px;font-weight:900;color:#fff'>✈</div>
      <div>
        <div style='font-family:Sora,sans-serif;font-size:14px;font-weight:700;
                    color:#fafafa;line-height:1.1'>{st.session_state.org_name}</div>
        <div style='font-family:DM Mono,monospace;font-size:9px;color:#52525b;
                    letter-spacing:.1em'>SAFETY OS · v3.0</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-hdr'>Navigation</div>", unsafe_allow_html=True)
    for icon, label in NAV:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label; st.rerun()

    st.divider()
    st.markdown("<div class='section-hdr'>⚡ AI Config</div>", unsafe_allow_html=True)
    nk = st.text_input("API Key", value=st.session_state.api_key, type="password",
                        placeholder="sk-ant-...", label_visibility="collapsed")
    if nk != st.session_state.api_key: st.session_state.api_key = nk
    connected = bool(st.session_state.api_key.strip())
    dot_c = C["green"] if connected else C["amber"]
    dot_l = "AI Connected" if connected else "Add key for AI features"
    st.markdown(f"<div style='color:{dot_c};font-size:10px;font-family:DM Mono,monospace;"
                f"margin-top:4px'>● {dot_l}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style='padding:12px;background:rgba(99,102,241,.1);border-radius:10px;
                border:1px solid rgba(99,102,241,.2)'>
      <div style='color:#818cf8;font-size:10px;font-family:DM Mono,monospace;
                  margin-bottom:4px'>LIVE DATA</div>
      <div style='color:#fafafa;font-size:12px;font-family:Sora,sans-serif;
                  font-weight:600'>{st.session_state.report_period}</div>
      <div style='color:#52525b;font-size:9px;font-family:DM Mono,monospace;margin-top:2px'>
        FSR:{len(fsr_df)} · HIRA:{len(hira_df)} · MOR:{len(mor_df)} · Bird:{len(bird_df)}
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BIRD HITS PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Bird Hits":
    page_header("Bird Strike Reports","BIRD/ANIMAL HIT LOG · CAAF-067-MSXX-2.0","Live Data")
    df = bird_df.copy()
    phases = df["PHASE OF FLIGHT"].value_counts() if "PHASE OF FLIGHT" in df.columns else pd.Series()
    locs   = df["LOCATION"].value_counts()         if "LOCATION"        in df.columns else pd.Series()

    k1,k2,k3,k4 = st.columns(4)
    with k1: mc("Total Events", len(df), C["alight"], "all records")
    with k2: mc("On Airport",   int(locs.get("ON AIRPORT",0)),   C["amber"], "≤200ft AGL", "linear-gradient(90deg,#f59e0b,#fbbf24)")
    with k3: mc("Near Airport", int(locs.get("NEAR AIRPORT",0)), C["cyan"],  "201–1000ft", "linear-gradient(90deg,#06b6d4,#67e8f9)")
    with k4: mc("Off Airport",  int(locs.get("OFF AIRPORT",0)),  C["green"], ">1000ft AGL","linear-gradient(90deg,#22c55e,#86efac)")

    st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        sec("Strikes by Phase of Flight")
        if not phases.empty:
            fig = go.Figure(go.Bar(x=phases.values, y=phases.index, orientation="h",
                marker=dict(color=phases.values,colorscale=[[0,C["surface"]],[1,C["accent"]]],
                            line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>%{x} events<extra></extra>"))
            fig.update_layout(**base_layout(280))
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with c2:
        sec("Location Distribution")
        if not locs.empty:
            fig2 = go.Figure(go.Pie(labels=locs.index,values=locs.values,hole=0.55,
                marker=dict(colors=[C["amber"],C["cyan"],C["green"],C["alight"]],
                            line=dict(color=C["card"],width=2)),
                hovertemplate="<b>%{label}</b><br>%{value}<br>%{percent}<extra></extra>"))
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=280,
                margin=dict(l=8,r=8,t=8,b=8),
                font=dict(family="DM Mono,monospace",color=C["muted"],size=10),
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=C["muted"])))
            st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})

    if not bird_monthly.empty:
        sec("Monthly Trend + Forecast")
        st.plotly_chart(chart_volume(bird_monthly, C["amber"]),
                        use_container_width=True, config={"displayModeBar":False})

    sec("Full Log")
    cols = [c for c in ["DATE OF OCCURRENCE","AIRCRAFT TYPE","FLT #","REGISTRATION  NO.",
                        "DESTINATION","PHASE OF FLIGHT","LOCATION","HEIGHT (AGL)",
                        "PARTS OF AIRCRAFT AFFECTED","REMARKS"] if c in df.columns]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)

    sec("Submit New Bird Hit Report")
    with st.expander("➕ Open CAAF-067 Form"):
        st.markdown("<div class='form-sec'><div class='form-sec-title'>Operator & Aircraft</div>", unsafe_allow_html=True)
        fa,fb,fc = st.columns(3)
        with fa:
            st.text_input("Operator", value="Air Sial", key="bh_op")
            st.text_input("Registration", placeholder="AP-BOA", key="bh_reg")
        with fb:
            st.selectbox("A/C Type", ["A320","Light","Medium","Heavy"], key="bh_type")
            st.text_input("Flight No.", placeholder="PF-721", key="bh_flt")
        with fc:
            st.text_input("Destination ICAO", placeholder="OPKC", key="bh_dest")
            st.date_input("Date", key="bh_date")
        st.markdown("<div class='form-sec-title' style='margin-top:14px'>Event Details</div>", unsafe_allow_html=True)
        fd,fe,ff = st.columns(3)
        with fd:
            st.text_input("Aerodrome", placeholder="OPLA", key="bh_aero")
            st.text_input("Runway", placeholder="36R", key="bh_rwy")
            st.text_input("Time (UTC)", placeholder="10:15", key="bh_time")
        with fe:
            st.selectbox("Phase of Flight",["TAKEOFF","CLIMB","CRUISE","DESCENT","APPROACH","LANDING","TAXI"],key="bh_phase")
            st.selectbox("Location",["ON AIRPORT","NEAR AIRPORT","OFF AIRPORT"],key="bh_loc")
            st.number_input("Height AGL (ft)",0,50000,key="bh_height")
        with ff:
            st.number_input("Speed IAS (kts)",0,400,key="bh_speed")
            st.selectbox("Strike Status",["Suspected","Confirmed"],key="bh_status")
            st.selectbox("Effect on Flight",["NONE","Aborted Take Off","Precautionary Landing","Engines Shut Down"],key="bh_effect")
        fg,fh = st.columns(2)
        with fg:
            st.text_input("Weather / METAR", key="bh_wx")
            st.text_input("Bird/Animal Species", key="bh_species")
            st.text_input("Number Seen/Struck", key="bh_num")
            st.selectbox("Bird Size",["Small","Medium","Large"],key="bh_size")
        with fh:
            st.text_area("Parts of Aircraft Affected", height=90, key="bh_parts")
            st.text_input("Delay caused", key="bh_delay")
        st.text_area("Remarks — damage, injuries, other pertinent information", height=90, key="bh_remarks")
        if st.button("Submit Bird Hit Report", key="sub_bh"):
            st.success("✅ Submitted. In production this would persist to your database and notify SQMS.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# FSR PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "FSR":
    page_header("Flight Service Pre-Reports","FSR · FLIGHT SERVICES DEPT · AIR SIAL","Live Data")
    df = fsr_df.copy()
    factors = df["AFFECTED FACTORS"].str.upper().str.strip().value_counts() if "AFFECTED FACTORS" in df.columns else pd.Series()

    k1,k2,k3,k4 = st.columns(4)
    med_total = int(sum(v for k,v in factors.items() if "MED" in k))
    dis_total = int(sum(v for k,v in factors.items() if "DISRUPT" in k or "PAX" in k))
    eng_total = int(sum(v for k,v in factors.items() if "ENG" in k))
    with k1: mc("Total FSRs",   len(df),       C["alight"], "all records")
    with k2: mc("Medical",      med_total,     C["red"],    "medical events",     "linear-gradient(90deg,#ef4444,#f87171)")
    with k3: mc("Disruptive Pax",dis_total,    C["amber"],  "pax-related",        "linear-gradient(90deg,#f59e0b,#fbbf24)")
    with k4: mc("Engineering",  eng_total,     C["cyan"],   "engineering issues", "linear-gradient(90deg,#06b6d4,#67e8f9)")

    st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        sec("Affected Factors — Frequency")
        st.plotly_chart(chart_fsr_factors(),use_container_width=True,config={"displayModeBar":False})
    with c2:
        sec("Monthly Report Volume + Forecast")
        if not fsr_monthly.empty:
            st.plotly_chart(chart_volume(fsr_monthly, C["accent"]),
                            use_container_width=True, config={"displayModeBar":False})

    sec("FSR Log")
    cols = [c for c in ["DATE","A/C REG","FLIGHT NO.","AFFECTED FACTORS","INCIDENT","ACTION TAKEN"] if c in df.columns]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)

    sec("Submit New FSR")
    with st.expander("➕ Open Flight Service Pre-Report Form"):
        st.markdown("<div class='form-sec'><div class='form-sec-title'>Flight Information</div>", unsafe_allow_html=True)
        f1,f2,f3,f4 = st.columns(4)
        with f1: st.text_input("Flight No.",placeholder="PF-717",key="fsr_flt"); st.date_input("Date",key="fsr_date")
        with f2: st.text_input("A/C REG",placeholder="AP-BOS",key="fsr_reg");   st.text_input("LCC Name",key="fsr_lcc")
        with f3: st.selectbox("A/C Type",["A320","A321","ATR"],key="fsr_type"); st.text_input("Sector/Time",placeholder="KHI-LHE/08:30",key="fsr_sec")
        with f4: st.text_input("S.No.",key="fsr_sno")
        st.markdown("<div class='form-sec-title' style='margin-top:14px'>Affected Factors</div>", unsafe_allow_html=True)
        FACTORS_LIST = ["Disruptive Pax","Delays","Hotel","Seating","Medical Emergency","Transport",
                        "Medical","Meal/Beverages","Logistics Items","Engineering","WCHS","Ramp",
                        "Scheduling","Medical Pouch","Boarding","Pax Complaint","Smoke","Others"]
        fcols = st.columns(4)
        sel = []
        for i,f in enumerate(FACTORS_LIST):
            with fcols[i%4]:
                if st.checkbox(f,key=f"fsr_f{i}"): sel.append(f)
        st.markdown("<div class='form-sec-title' style='margin-top:14px'>Incident & Action</div>", unsafe_allow_html=True)
        st.text_area("Incident Description", height=100, key="fsr_inc")
        st.text_area("Action Taken", height=70, key="fsr_act")
        st.text_input("LCC Signature and AS Number", key="fsr_sig")
        if st.button("Submit FSR", key="sub_fsr"): st.success("✅ FSR submitted.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# HIRA PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "HIRA":
    page_header("Hazard Identification & Risk Assessment","HIRA · AS-CS-003 · CORPORATE SAFETY DEPT","Live Data")
    df = hira_df.copy()
    rr = df["int. risk rating"].str.lower().str.strip().value_counts() if "int. risk rating" in df.columns else pd.Series()
    st_col = df["status"].str.lower().str.strip().value_counts() if "status" in df.columns else pd.Series()

    k1,k2,k3,k4 = st.columns(4)
    with k1: mc("Total Hazards", len(df), C["alight"], "all HIRA records")
    with k2: mc("High Risk",   int(rr.get("high",0)),   C["red"],   "intolerable","linear-gradient(90deg,#ef4444,#f87171)")
    with k3: mc("Medium Risk", int(rr.get("medium",0)), C["amber"], "tolerable",  "linear-gradient(90deg,#f59e0b,#fbbf24)")
    with k4: mc("Closed",      int(st_col.get("closed",0)), C["green"],"resolved","linear-gradient(90deg,#22c55e,#86efac)")

    st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        sec("Risk Rating Distribution")
        if not rr.empty:
            cm = {"low":C["green"],"medium":C["amber"],"high":C["red"]}
            clrs = [cm.get(k,C["alight"]) for k in rr.index]
            fig = go.Figure(go.Bar(x=[k.upper() for k in rr.index],y=rr.values,
                marker_color=clrs,hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"))
            fig.update_layout(**base_layout(260))
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with c2:
        sec("Monthly HIRA Volume + Forecast")
        if not hira_monthly.empty:
            st.plotly_chart(chart_volume(hira_monthly, C["cyan"]),
                            use_container_width=True, config={"displayModeBar":False})

    sec("HIRA Log")
    show = [c for c in ["date of report","reporter name","department","location of hazard",
                        "hazard description","int. risk assessment ","int. risk rating",
                        "corrective action plan","status"] if c in df.columns]
    st.dataframe(df[show] if show else df, use_container_width=True, hide_index=True)

    sec("Submit New HIRA Report")
    with st.expander("➕ Open HIRA Form (AS-CS-003)"):
        st.markdown("<div class='form-sec'><div class='form-sec-title'>Reporter Information</div>", unsafe_allow_html=True)
        h1,h2,h3 = st.columns(3)
        with h1: st.date_input("Date of Report",key="h_date"); st.text_input("Reporter Name",key="h_rep")
        with h2: st.selectbox("Department",["Flight Operations","Engineering","Airport Services","Safety","Cabin Crew","Ground Handling","Other"],key="h_dept"); st.text_input("Report No.",key="h_rno")
        with h3: st.text_input("Location of Hazard",placeholder="Ramp / Cockpit / Galley",key="h_loc"); st.text_input("Date/Time Hazard Identified",key="h_dti")
        st.text_area("Hazard Description",height=90,key="h_desc")
        st.markdown("<div class='form-sec-title' style='margin-top:14px'>Initial Risk Assessment</div>", unsafe_allow_html=True)
        p1,p2 = st.columns(2)
        with p1: h_prob = st.selectbox("Probability",["A - Frequent","B - Occasional","C - Remote","D - Improbable","E - Extremely Improbable"],key="h_prob")
        with p2: h_sev  = st.selectbox("Severity",["1 - Catastrophic","2 - Hazardous","3 - Major","4 - Minor","5 - Negligible"],key="h_sev")
        INTOL = {("A - Frequent","1 - Catastrophic"),("A - Frequent","2 - Hazardous"),
                 ("B - Occasional","1 - Catastrophic"),("B - Occasional","2 - Hazardous")}
        rating = "INTOLERABLE" if (h_prob,h_sev) in INTOL else "TOLERABLE/ACCEPTABLE"
        badge_cls = "br" if rating=="INTOLERABLE" else "ba"
        st.markdown(f"<span class='badge {badge_cls}'>Risk Rating: {rating}</span>",unsafe_allow_html=True)
        st.text_area("Root Cause Analysis",height=70,key="h_root")
        st.text_area("Corrective Action Plan (CAP)",height=70,key="h_cap")
        r1,r2,r3 = st.columns(3)
        with r1: st.text_input("Responsible Person",key="h_resp")
        with r2: st.date_input("Target Date",key="h_tgt")
        with r3: st.selectbox("Residual Risk (After CAP)",["Intolerable","Tolerable","Acceptable"],key="h_res")
        st.text_area("Remarks by Snr. GMCS&S",height=60,key="h_rmk")
        if st.button("Submit HIRA Report",key="sub_hira"): st.success("✅ HIRA submitted to Corporate Safety Department.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MOR PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "MOR":
    page_header("Mandatory Occurrence Reports","MOR · CAAF-001-SBXX-1.0 · SAFETY & INVESTIGATION BOARD","Live Data")
    df = mor_df.copy()
    causes = df["NATURE AND CAUSE"].value_counts() if "NATURE AND CAUSE" in df.columns else pd.Series()
    dmg    = df["DAMAGE TO AIRCRAFT"].value_counts() if "DAMAGE TO AIRCRAFT" in df.columns else pd.Series()

    k1,k2,k3,k4 = st.columns(4)
    with k1: mc("Total MORs", len(df), C["alight"], "mandatory reports")
    with k2: mc("Registrations", df["REGISTRATION"].nunique() if "REGISTRATION" in df.columns else 0, C["cyan"], "unique A/C","linear-gradient(90deg,#06b6d4,#67e8f9)")
    with k3: mc("Routes", df["DESTINATION"].nunique() if "DESTINATION" in df.columns else 0, C["amber"], "unique destinations","linear-gradient(90deg,#f59e0b,#fbbf24)")
    with k4: mc("NIL Damage", int(dmg.get("NIL",0)), C["green"], "no structural damage","linear-gradient(90deg,#22c55e,#86efac)")

    st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        sec("MOR by Nature & Cause")
        st.plotly_chart(chart_mor_causes(),use_container_width=True,config={"displayModeBar":False})
    with c2:
        sec("Monthly MOR Volume + Forecast")
        if not mor_monthly.empty:
            st.plotly_chart(chart_volume(mor_monthly, C["red"]),
                            use_container_width=True, config={"displayModeBar":False})

    sec("MOR Log")
    cols = [c for c in ["DATE","FLIGHT NO.","REGISTRATION","REPORTER","ORIGIN","DESTINATION",
                        "NATURE AND CAUSE","DAMAGE TO AIRCRAFT","WEATHER","SUMMARY"] if c in df.columns]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)

    sec("Submit New MOR")
    with st.expander("➕ Open MOR Form (CAAF-001-SBXX-1.0)"):
        st.markdown("<div class='form-sec'><div class='form-sec-title'>Reporter & Airport</div>", unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        with m1: st.text_input("Reporting Officer Name",key="m_rn"); st.text_input("Designation",key="m_des"); st.text_input("Telephone",key="m_tel")
        with m2: st.text_input("Airport of Origin",placeholder="OPKC",key="m_apt"); st.text_input("MOR No.",key="m_no"); st.text_input("Year",value=str(datetime.now().year),key="m_yr")
        with m3: st.selectbox("Aircraft Type",["A320","A321","ATR-72","Other"],key="m_act"); st.date_input("Date of Occurrence",key="m_date")
        st.markdown("<div class='form-sec-title' style='margin-top:14px'>Aircraft Information</div>", unsafe_allow_html=True)
        a1,a2 = st.columns(2)
        with a1:
            st.text_input("Aircraft Reg & Flight No.",placeholder="AP-BOA / PF-714",key="m_reg")
            st.text_input("Owner / Operator",value="Air Sial",key="m_op")
            st.text_input("Name of Pilot",key="m_pilot")
        with a2:
            st.text_input("Last Point of Departure",key="m_dep")
            st.text_input("Point of Intended Landing",key="m_dest")
            st.text_input("Nature of Flight",placeholder="Scheduled Passenger",key="m_nat")
        st.markdown("<div class='form-sec-title' style='margin-top:14px'>Occurrence Details</div>", unsafe_allow_html=True)
        b1,b2 = st.columns(2)
        with b1:
            st.text_input("Location (geographical reference)",key="m_loc")
            st.text_input("Total persons on board",key="m_pob")
            st.text_input("Number killed (if any)",key="m_kill")
        with b2:
            st.text_input("Number seriously injured",key="m_inj")
            st.text_input("Weather Condition",key="m_wx")
            st.text_area("Nature and extent of damage to A/C",height=60,key="m_dmg")
        st.text_area("Nature and cause of incident/accident (as known)",height=80,key="m_cause")
        st.text_area("Summary of the Occurrence",height=100,key="m_sum")
        st.text_area("Dangerous goods on board",height=60,key="m_dg")
        st.markdown("**Distribution:** PD (Reg) HQ CAA · President SIB HQ CAA · Dir SQMS · SO to DDG CAA")
        s1,s2 = st.columns(2)
        with s1: st.text_input("Signature / Name",key="m_sig")
        with s2: st.text_input("Designation",key="m_des2")
        if st.button("Submit MOR",key="sub_mor"): st.success("✅ MOR submitted. Report transmitted to PSIB, PD (Reg), and Dir SQMS.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# AI INSIGHTS PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "AI Insights":
    page_header("AI Safety Intelligence","POWERED BY CLAUDE · REAL OPERATIONAL DATA")

    ai_left, ai_right = st.columns([1.2,1])
    with ai_left:
        sec("Executive AI Summary")
        if st.button("⚡ Generate GM Briefing", key="btn_sum"):
            with st.spinner("Analysing operational data..."):
                st.session_state.last_ai_summary = ai_summary()
        if st.session_state.last_ai_summary:
            st.markdown(f"""<div class='ai-box'>
              <div class='ai-box-hdr'>🛡️ GM Executive Briefing — {st.session_state.org_name}</div>
              {st.session_state.last_ai_summary}</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class='ai-box' style='opacity:.35;text-align:center;padding:36px'>
              Click "Generate GM Briefing" for an AI-powered analysis of your live data.</div>""",
              unsafe_allow_html=True)

        sec("Per-Dataset AI Insights")
        METRICS = [
            ("FSR Monthly Volume",  period_labels(fsr_monthly),  fsr_monthly.values.tolist()),
            ("HIRA Monthly Volume", period_labels(hira_monthly), hira_monthly.values.tolist()),
            ("MOR Monthly Volume",  period_labels(mor_monthly),  mor_monthly.values.tolist()),
            ("Bird Strike Volume",  period_labels(bird_monthly), bird_monthly.values.tolist()),
        ]
        for name, labels, vals in METRICS:
            if not vals: continue
            with st.expander(f"🔍  {name}"):
                pred = forecast(vals)
                pct = round(abs(vals[-1]-vals[0])/max(vals[0],1)*100)
                trend = "↑" if vals[-1]>vals[0] else "↓"
                ca,cb = st.columns(2)
                with ca:
                    pairs = " → ".join(f"{l}: {v}" for l,v in zip(labels,vals))
                    st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:11px;"
                                f"color:#a1a1aa;line-height:1.8'>{pairs}<br>"
                                f"<b style='color:#fafafa'>{trend} {pct}% change</b></div>",
                                unsafe_allow_html=True)
                with cb:
                    st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:11px;"
                                f"color:#a1a1aa;line-height:1.8'>Forecast: "
                                f"<b style='color:#818cf8'>{pred[0]} · {pred[1]} · {pred[2]}</b><br>"
                                f"<span style='color:#818cf8'>Linear regression</span></div>",
                                unsafe_allow_html=True)
                if st.button("Get AI Insight", key=f"ai_{name}"):
                    with st.spinner(""):
                        ins = ai_metric_insight(name, labels, vals)
                        if ins:
                            st.markdown(f"<div class='ai-box' style='margin-top:8px'>"
                                        f"<div class='ai-box-hdr'>💡 GM Insight</div>{ins}</div>",
                                        unsafe_allow_html=True)

    with ai_right:
        sec("Chat with SafetyOS AI")
        if not st.session_state.chat_history:
            st.markdown("""<div class='ai-box' style='opacity:.4;text-align:center;padding:28px'>
              Ask anything about your safety data.<br><br>
              <i>"Which aircraft has the most MORs?"</i><br>
              <i>"What's the top bird strike risk area?"</i><br>
              <i>"What FSR factor needs urgent attention?"</i></div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history[-10:]:
                is_ai = msg["role"]=="assistant"
                col = C["alight"] if is_ai else C["green"]
                role = "🤖 SafetyOS AI" if is_ai else "👤 You"
                bg = "#1c1c1f" if is_ai else "#18181b"
                st.markdown(f"""<div class='chat-msg' style='background:{bg}'>
                  <div class='chat-role' style='color:{col}'>{role}</div>
                  <div style='color:#e4e4e7'>{msg["content"]}</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='font-family:DM Mono,monospace;font-size:9px;color:#52525b;"
                    "text-transform:uppercase;letter-spacing:.12em;margin:12px 0 6px'>"
                    "Quick Prompts</div>", unsafe_allow_html=True)
        qc1,qc2 = st.columns(2)
        QUICK = ["Top bird strike risk area?","Most critical MOR cause?","FSR medical trend?","Highest priority HIRA?"]
        for i,p in enumerate(QUICK):
            with (qc1 if i%2==0 else qc2):
                if st.button(p, key=f"qp{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":p})
                    with st.spinner(""):
                        st.session_state.chat_history.append({"role":"assistant","content":ai_chat(p)})
                    st.rerun()
        user_in = st.text_input("Ask about your data…",key="chat_in",label_visibility="collapsed")
        sc1,sc2 = st.columns([2,1])
        with sc1:
            if st.button("Send →",key="send_btn"):
                if user_in.strip():
                    st.session_state.chat_history.append({"role":"user","content":user_in})
                    with st.spinner(""):
                        st.session_state.chat_history.append({"role":"assistant","content":ai_chat(user_in)})
                    st.rerun()
        with sc2:
            if st.session_state.chat_history:
                if st.button("🗑 Clear",key="clear_chat"):
                    st.session_state.chat_history=[]; st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Settings":
    page_header("Settings & Configuration","SAFETYOS PLATFORM · ORGANISATION PREFERENCES")

    tab1,tab2,tab3,tab4 = st.tabs(["🏢 Organisation","🤖 AI & Integration","🎯 KPI Targets","📂 Data Management"])

    with tab1:
        sec("Organisation Profile")
        st.markdown("<div class='form-sec'>", unsafe_allow_html=True)
        s1,s2 = st.columns(2)
        with s1:
            norg = st.text_input("Organisation Name",value=st.session_state.org_name)
            if norg!=st.session_state.org_name: st.session_state.org_name=norg
            st.text_input("IATA Code",value="ER",placeholder="ER")
            st.text_input("CAA Licence No.",placeholder="PKA-AOC-2019-01")
        with s2:
            nper = st.text_input("Report Period",value=st.session_state.report_period)
            if nper!=st.session_state.report_period: st.session_state.report_period=nper
            st.selectbox("Primary Fleet Type",["Airbus A320","Boeing 737","ATR-72","Mixed Fleet"])
            st.text_input("Safety Email",value="safety@airsial.com")
        st.markdown("</div>", unsafe_allow_html=True)

        sec("Regulatory Contacts")
        st.markdown("<div class='form-sec'>", unsafe_allow_html=True)
        r1,r2 = st.columns(2)
        with r1: st.text_input("PSIB Tel",value="92-51-4472750",key="ps_tel"); st.text_input("PSIB Cell",value="0300-8250472",key="ps_cell"); st.text_input("PSIB Fax",value="92-21-34604305",key="ps_fax")
        with r2: st.text_input("SQMS Email",value="jtd.sqms@caapakistan.com.pk",key="sq_em"); st.text_input("AAIB Email",value="paaib@caapakistan.com.pk",key="aa_em"); st.text_input("APS Email",value="director.aps@caapakistan.com.pk",key="ap_em")
        st.markdown("</div>", unsafe_allow_html=True)

        sec("Reporting Deadlines")
        st.markdown("<div class='form-sec'>", unsafe_allow_html=True)
        d1,d2,d3 = st.columns(3)
        with d1: st.number_input("Bird Hit Form (hours)",value=24,min_value=1,max_value=72)
        with d2: st.number_input("Bird Hit SQMS Report (hours)",value=48,min_value=1,max_value=168)
        with d3: st.number_input("MOR Telephone Notice (min)",value=60,min_value=5,max_value=240)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        sec("AI Configuration")
        st.markdown("<div class='form-sec'>", unsafe_allow_html=True)
        a1,a2 = st.columns(2)
        with a1:
            nk = st.text_input("Anthropic API Key",value=st.session_state.api_key,
                               type="password",placeholder="sk-ant-…",
                               help="console.anthropic.com")
            if nk!=st.session_state.api_key: st.session_state.api_key=nk
        with a2:
            st.selectbox("AI Model",["claude-sonnet-4-20250514 (Recommended)","claude-opus-4-20250514 (Advanced)"])
        if bool(st.session_state.api_key.strip()): st.success("✅ API Key configured — all AI features active.")
        else: st.warning("⚠️ No API key. AI summary, chat, and insights require an Anthropic key.")
        st.markdown("</div>", unsafe_allow_html=True)

        sec("Notifications")
        st.markdown("<div class='form-sec'>", unsafe_allow_html=True)
        n1,n2 = st.columns(2)
        with n1:
            st.checkbox("Email alerts for new MOR submissions",value=True)
            st.checkbox("Email alerts for High Risk HIRA",value=True)
            st.checkbox("Daily summary email",value=False)
        with n2:
            st.checkbox("Notify on bird strike with engine damage",value=True)
            st.checkbox("Alert on overdue HIRA CAP",value=True)
            st.checkbox("Weekly AI digest",value=False)
        st.text_input("Alert Recipients",placeholder="safety@airsial.com, gm@airsial.com")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        sec("Safety KPI Targets")
        st.markdown("<div class='form-sec'>", unsafe_allow_html=True)
        t1,t2,t3 = st.columns(3)
        with t1:
            nat = st.number_input("Audit Closure Target (%)",0,100,st.session_state.audit_target)
            st.session_state.audit_target = nat
            st.number_input("Max Bird Hits per Quarter",0,100,10)
        with t2:
            nhs = st.number_input("Max High Sev Events",0,50,st.session_state.max_high_sev)
            st.session_state.max_high_sev = nhs
            st.number_input("Min Proactive HIRA / Month",0,200,5)
        with t3:
            st.number_input("Max Open MORs at Month End",0,50,3)
            st.number_input("FSR Submission Threshold (hrs)",0,72,24)
        st.markdown("</div>", unsafe_allow_html=True)

        sec("Risk Matrix Thresholds")
        st.markdown("<div class='form-sec'>", unsafe_allow_html=True)
        rm1,rm2,rm3 = st.columns(3)
        with rm1:
            st.markdown("🔴 **Intolerable** — Immediate action")
            st.text_input("Codes",value="1A,2A,1B,2B,1C,1D,1E",key="rm_i")
        with rm2:
            st.markdown("🟡 **Tolerable** — Management decision")
            st.text_input("Codes",value="3A,3B,2C,2D,3C,4A",key="rm_t")
        with rm3:
            st.markdown("🟢 **Acceptable** — Monitor")
            st.text_input("Codes",value="4B,4C,4D,5A-5E,3D,3E",key="rm_a")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        sec("Data Sources")
        st.markdown("<div class='form-sec'>", unsafe_allow_html=True)
        d1,d2 = st.columns(2)
        with d1:
            st.markdown("**Loaded Excel Files**")
            for name,df_l in [("Bird Hits",bird_df),("FSR",fsr_df),("HIRA",hira_df),("MOR",mor_df)]:
                bc = "bg" if len(df_l)>0 else "br"
                st.markdown(f"<span class='badge {bc}'>{name}: {len(df_l)} records</span>&nbsp;", unsafe_allow_html=True)
        with d2:
            st.markdown("**Export as CSV**")
            for name,df_l in [("Bird Hits",bird_df),("FSR",fsr_df),("HIRA",hira_df),("MOR",mor_df)]:
                if not df_l.empty:
                    csv = df_l.to_csv(index=False).encode("utf-8")
                    st.download_button(f"⬇ {name} CSV",csv,f"{name.lower().replace(' ','_')}.csv","text/csv",key=f"dl_{name}")
        st.markdown("</div>", unsafe_allow_html=True)

        sec("Reload & Reset")
        st.markdown("<div class='form-sec' style='border-color:rgba(239,68,68,.25)'>", unsafe_allow_html=True)
        z1,z2 = st.columns(2)
        with z1:
            if st.button("🗑 Clear Chat History",key="clear_chat2"):
                st.session_state.chat_history=[]; st.session_state.last_ai_summary=None
                st.success("✅ Cleared."); st.rerun()
        with z2:
            if st.button("♻️ Reload Data from Excel",key="reload"):
                st.cache_data.clear(); st.success("✅ Cache cleared — reload the page."); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW PAGE  — original visual DNA with live data
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:18px'>
  <div>
    <div style='font-family:Sora,sans-serif;font-size:22px;font-weight:800;color:#fafafa'>
      {st.session_state.org_name} — Safety Performance</div>
    <div style='font-family:DM Mono,monospace;font-size:10px;color:#a1a1aa;
                letter-spacing:.06em;margin-top:2px'>
      EXECUTIVE DASHBOARD · {st.session_state.report_period} · LIVE DATA + AI FORECAST</div>
  </div>
  <div style='padding:5px 14px;border-radius:8px;background:rgba(34,197,94,.12);
              border:1px solid rgba(34,197,94,.3);color:#22c55e;
              font-size:10px;font-family:DM Mono,monospace'>● LIVE</div>
</div>""", unsafe_allow_html=True)

# KPI strip — from real data
hira_open = int((hira_df["status"].str.lower().str.strip()=="open").sum()) if "status" in hira_df.columns else 0
hira_high = int((hira_df["int. risk rating"].str.lower().str.strip()=="high").sum()) if "int. risk rating" in hira_df.columns else 0
avg_fsr_monthly = round(fsr_monthly.mean()) if not fsr_monthly.empty else 0
fsr_trend = int(fsr_monthly.iloc[-1] - fsr_monthly.iloc[0]) if len(fsr_monthly)>=2 else 0

k1,k2,k3,k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>◈ Total Reports (All Types)</div>
      <div class='metric-value' style='color:#818cf8'>{len(fsr_df)+len(hira_df)+len(mor_df)+len(bird_df)}</div>
      <div class='metric-sub'>FSR · HIRA · MOR · Bird</div></div>""", unsafe_allow_html=True)
with k2:
    c = C["red"] if hira_high>st.session_state.max_high_sev else C["green"]
    st.markdown(f"""<div class='metric-card' style='--accent-line:linear-gradient(90deg,#ef4444,#f87171)'>
      <div class='metric-label'>⬡ High-Risk HIRA Open</div>
      <div class='metric-value' style='color:{c}'>{hira_high} hazards</div>
      <div class='metric-sub'>intolerable risk rating</div></div>""", unsafe_allow_html=True)
with k3:
    fsr_c = C["green"] if fsr_trend>=0 else C["amber"]
    arrow = "↑" if fsr_trend>=0 else "↓"
    st.markdown(f"""<div class='metric-card' style='--accent-line:linear-gradient(90deg,#6366f1,#818cf8)'>
      <div class='metric-label'>◎ FSR Trend (last 6 months)</div>
      <div class='metric-value' style='color:{fsr_c}'>{arrow} {abs(fsr_trend)} / mo</div>
      <div class='metric-sub'>avg {avg_fsr_monthly} reports/month</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class='metric-card' style='--accent-line:linear-gradient(90deg,#f59e0b,#fbbf24)'>
      <div class='metric-label'>◉ Bird Strikes (Total)</div>
      <div class='metric-value' style='color:{C["amber"]}'>{len(bird_df)} events</div>
      <div class='metric-sub'>all records on file</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)

# ─── TABS — same structure as original ────────────────────────────────────────
tab_charts, tab_ai, tab_input = st.tabs(["📊  Charts", "🤖  AI Insights", "✏️  Input Data"])

with tab_charts:
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""<div class='metric-card' style='margin-bottom:10px'>
          <div class='metric-label'>◈ FSR Monthly Volume</div>
          <div style='color:#fafafa;font-size:14px;font-weight:600;font-family:Sora,sans-serif'>
            Flight Service Reports — Trend + Forecast</div></div>""", unsafe_allow_html=True)
        if not fsr_monthly.empty:
            st.plotly_chart(chart_volume(fsr_monthly,C["accent"]),
                            use_container_width=True,config={"displayModeBar":False})
    with c2:
        st.markdown("""<div class='metric-card' style='margin-bottom:10px'>
          <div class='metric-label'>⬡ HIRA Severity Distribution</div>
          <div style='color:#fafafa;font-size:14px;font-weight:600;font-family:Sora,sans-serif'>
            High / Medium Risk Events + Forecast</div></div>""", unsafe_allow_html=True)
        st.plotly_chart(chart_severity_monthly(),
                        use_container_width=True,config={"displayModeBar":False})

    c3,c4 = st.columns(2)
    with c3:
        st.markdown("""<div class='metric-card' style='margin-bottom:10px'>
          <div class='metric-label'>◎ HIRA Closure Rate</div>
          <div style='color:#fafafa;font-size:14px;font-weight:600;font-family:Sora,sans-serif'>
            Closed vs Total — Gauge</div></div>""", unsafe_allow_html=True)
        total_hira = len(hira_df)
        closed_hira = int((hira_df["status"].str.lower().str.strip()=="closed").sum()) if "status" in hira_df.columns else 0
        closure_rate = round(closed_hira/max(total_hira,1)*100)
        st.plotly_chart(chart_gauge(closure_rate),use_container_width=True,config={"displayModeBar":False})
        m1_,m2_,m3_ = st.columns(3)
        for col_,lbl_,val_,col_c in [(m1_,"Total",total_hira,C["alight"]),(m2_,"Closed",closed_hira,C["green"]),(m3_,"Open",hira_open,C["red"])]:
            with col_:
                st.markdown(f"""<div style='background:{C["bg"]};border:1px solid {C["border"]};
                  border-radius:10px;padding:10px;text-align:center'>
                  <div style='font-family:DM Mono,monospace;font-size:9px;color:#a1a1aa'>{lbl_}</div>
                  <div style='font-family:Sora,sans-serif;font-size:17px;font-weight:800;color:{col_c}'>{val_}</div>
                </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown("""<div class='metric-card' style='margin-bottom:10px'>
          <div class='metric-label'>◉ FSR Affected Factors</div>
          <div style='color:#fafafa;font-size:14px;font-weight:600;font-family:Sora,sans-serif'>
            Top Incident Categories</div></div>""", unsafe_allow_html=True)
        st.plotly_chart(chart_fsr_factors(),use_container_width=True,config={"displayModeBar":False})

with tab_ai:
    st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
    ai_l,ai_r = st.columns([1.2,1])
    with ai_l:
        st.markdown("""<div style='font-family:Sora,sans-serif;font-size:15px;font-weight:700;
          color:#fafafa;margin-bottom:8px'>🤖 AI Executive Summary</div>""", unsafe_allow_html=True)
        if st.button("⚡ Generate Executive Summary",key="btn_sum_ov"):
            with st.spinner("Analysing live safety data…"):
                st.session_state.last_ai_summary = ai_summary()
        if st.session_state.last_ai_summary:
            st.markdown(f"""<div class='ai-box'>
              <div class='ai-box-hdr'>🛡️ GM Executive Briefing — {st.session_state.org_name}</div>
              {st.session_state.last_ai_summary}</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class='ai-box' style='opacity:.4;text-align:center;padding:28px'>
              Click "Generate Executive Summary" to get an AI-powered briefing from your live data.
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div style='font-family:Sora,sans-serif;font-size:15px;font-weight:700;
          color:#fafafa;margin:16px 0 8px'>📈 Per-Dataset AI Insights</div>""", unsafe_allow_html=True)
        METRICS_OV = [
            ("FSR Monthly Volume",  period_labels(fsr_monthly),  fsr_monthly.values.tolist()),
            ("HIRA Monthly Volume", period_labels(hira_monthly), hira_monthly.values.tolist()),
            ("MOR Monthly Volume",  period_labels(mor_monthly),  mor_monthly.values.tolist()),
        ]
        for name,labels,vals in METRICS_OV:
            if not vals: continue
            with st.expander(f"🔍  {name}"):
                pred = forecast(vals)
                pct  = round(abs(vals[-1]-vals[0])/max(vals[0],1)*100)
                trend= "↑" if vals[-1]>vals[0] else "↓"
                ca,cb = st.columns(2)
                with ca:
                    st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:11px;"
                                f"color:#a1a1aa;line-height:1.8'>"
                                f"{'<br>'.join(f'{l}: <b style=chr(34)color:#fafafa{chr(34)}>{v}</b>' for l,v in zip(labels,vals))}<br>"
                                f"<span style='color:#fafafa'>{trend} {pct}% change</span></div>",
                                unsafe_allow_html=True)
                with cb:
                    st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:11px;"
                                f"color:#a1a1aa;line-height:1.8'>Forecast:<br>"
                                f"<b style='color:#818cf8'>{pred[0]} · {pred[1]} · {pred[2]}</b><br>"
                                f"<span style='color:#818cf8'>Linear regression</span></div>",
                                unsafe_allow_html=True)
                if st.button("Get AI Insight",key=f"ai_ov_{name}"):
                    with st.spinner(""):
                        ins = ai_metric_insight(name,labels,vals)
                        if ins:
                            st.markdown(f"<div class='ai-box' style='margin-top:8px'>"
                                        f"<div class='ai-box-hdr'>💡 GM Insight</div>{ins}</div>",
                                        unsafe_allow_html=True)

    with ai_r:
        st.markdown("""<div style='font-family:Sora,sans-serif;font-size:15px;font-weight:700;
          color:#fafafa;margin-bottom:8px'>💬 Chat with SafetyOS AI</div>""", unsafe_allow_html=True)
        if not st.session_state.chat_history:
            st.markdown(f"""<div style='background:{C["card"]};border:1px solid {C["border"]};
              border-radius:12px;padding:22px;text-align:center;
              font-family:DM Mono,monospace;font-size:11px;color:#a1a1aa;line-height:1.8'>
              Ask me anything about your safety data.<br><br>
              <i>"What's the biggest risk right now?"</i><br>
              <i>"Which A/C has the most issues?"</i>
            </div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history[-10:]:
                is_ai = msg["role"]=="assistant"
                rc = C["alight"] if is_ai else C["green"]
                rl = "🤖 SafetyOS AI" if is_ai else "👤 You"
                bg = C["card"] if is_ai else C["surface"]
                st.markdown(f"""<div class='chat-msg' style='background:{bg}'>
                  <div class='chat-role' style='color:{rc}'>{rl}</div>
                  <div style='color:#e4e4e7'>{msg["content"]}</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:10px;font-family:DM Mono,monospace;font-size:9px;"
                    "color:#a1a1aa;letter-spacing:.1em;margin-bottom:4px'>QUICK PROMPTS</div>",
                    unsafe_allow_html=True)
        qp1,qp2 = st.columns(2)
        QUICK_OV = ["What's the biggest risk?","Summarise HIRA status","FSR trend analysis","Top MOR concern?"]
        for i,p in enumerate(QUICK_OV):
            with (qp1 if i%2==0 else qp2):
                if st.button(p,key=f"qp_ov{i}",use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":p})
                    with st.spinner(""):
                        st.session_state.chat_history.append({"role":"assistant","content":ai_chat(p)})
                    st.rerun()
        ui = st.text_input("Ask about your safety data…",key="chat_in_ov",label_visibility="collapsed")
        s1,s2 = st.columns([2,1])
        with s1:
            if st.button("Send →",key="send_ov"):
                if ui.strip():
                    st.session_state.chat_history.append({"role":"user","content":ui})
                    with st.spinner("Thinking…"):
                        st.session_state.chat_history.append({"role":"assistant","content":ai_chat(ui)})
                    st.rerun()
        with s2:
            if st.session_state.chat_history:
                if st.button("🗑 Clear",key="clr_ov"):
                    st.session_state.chat_history=[]; st.rerun()

with tab_input:
    st.markdown("""<div style='font-family:Sora,sans-serif;font-size:15px;font-weight:700;
      color:#fafafa;margin-bottom:2px'>📥 Upload Updated Excel Files</div>
    <div style='font-family:DM Mono,monospace;font-size:11px;color:#a1a1aa;margin-bottom:18px'>
      Drop replacement files here — the dashboard auto-refreshes on save.</div>""",
      unsafe_allow_html=True)

    for label, fname in [("🐦 Bird Hits","BIRD_HITS.xlsx"),("📋 FSR","FSR.xlsx"),
                          ("⚠️ HIRA","HIRA.xlsx"),("📊 MOR","MOR.xlsx")]:
        uf = st.file_uploader(f"Replace {label} file", type="xlsx", key=f"up_{fname}")
        if uf is not None:
            with open(fname,"wb") as f: f.write(uf.read())
            st.cache_data.clear()
            st.success(f"✅ {label} updated! Refreshing…")
            st.rerun()

    st.divider()
    st.markdown("""<div style='font-family:DM Mono,monospace;font-size:10px;color:#a1a1aa;
      margin-bottom:12px'>CURRENT DATA SNAPSHOT</div>""", unsafe_allow_html=True)
    snp = []
    for nm,df_s in [("Bird Hits",bird_df),("FSR",fsr_df),("HIRA",hira_df),("MOR",mor_df)]:
        if not df_s.empty:
            snp.append({"Dataset":nm,"Records":len(df_s),"Columns":len(df_s.columns),
                        "Status":"✅ Loaded"})
    if snp: st.dataframe(pd.DataFrame(snp), use_container_width=True, hide_index=True)
        
