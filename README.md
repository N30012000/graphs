# 🛡️ SafetyOS Executive Dashboard — Streamlit

A professional AI-powered Safety Performance Dashboard built with Streamlit, Plotly, and the Anthropic Claude API.

## Features
- 📊 4 interactive metric charts (Volume, Severity, Audit Gauge, Proactivity)
- 🔮 AI-powered 3-month predictive forecasts (linear regression)
- 🤖 Claude AI executive summary generation
- 💬 Chat interface to query your safety data
- 📈 Per-metric AI insights on demand
- ✏️ Live data input — charts update instantly on save
- 🧭 Full sidebar navigation

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Configure AI (optional but recommended)
- Open the app in your browser (http://localhost:8501)
- In the **sidebar**, paste your Anthropic API key (`sk-ant-...`)
- Get your key at: https://console.anthropic.com

## Usage

| Tab | What it does |
|-----|-------------|
| 📊 Charts | All 4 metric charts with hover tooltips and forecast lines |
| 🤖 AI Insights | Executive summary, per-metric AI analysis, chat assistant |
| ✏️ Input Data | Edit Jan/Feb/Mar values, preview data table |

## AI Features
- **Executive Summary** — One-click GM-level briefing of all Q1 metrics
- **Per-Metric Insights** — Deep-dive AI commentary on any individual metric
- **Chat Assistant** — Ask free-form questions about your safety data
- **Quick Prompts** — Pre-built questions for common GM queries

## Without an API Key
All charts, forecasts, and data input work fully without an API key.  
Only the AI text generation features require the Anthropic key.
