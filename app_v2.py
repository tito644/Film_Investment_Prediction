"""
==========================================================
FILM INVESTMENT INTELLIGENCE — STREAMLIT APP v2.0
Pre-Release Signals + TABB Sentiment + Portfolio Layer
==========================================================
Sprint 6 — Peter's Feedback Implementation
Author: Tarek ElNaggar | Lunim | 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Film Investment Intelligence v2",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLES ──
st.markdown("""
<style>
    .main { background-color: #0D1117; color: #FFFFFF; }
    .stApp { background-color: #0D1117; }
    [data-testid="stSidebar"] { background-color: #111927; }
    .metric-card {
        background: #161F2E; border-radius: 10px;
        padding: 16px 20px; border-left: 4px solid;
        margin: 6px 0;
    }
    .invest    { border-color: #2ECC71; }
    .caution   { border-color: #F39C12; }
    .review    { border-color: #E67E22; }
    .avoid     { border-color: #E74C3C; }
    .new-badge {
        background: #1A3A6B; color: #4A9EDB;
        padding: 2px 8px; border-radius: 12px;
        font-size: 11px; font-weight: bold;
    }
    h1, h2, h3 { color: #F5A623; }
    .stTabs [data-baseweb="tab"] { color: #B8C8D8; }
    .stTabs [aria-selected="true"] { color: #F5A623; border-bottom: 2px solid #F5A623; }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ──
@st.cache_resource
def load_model():
    base = Path("film_investment_model_v2")
    model    = joblib.load(base / "model_v2.pkl")
    scaler   = joblib.load(base / "scaler_v2.pkl")
    le_genre = joblib.load(base / "le_genre_v2.pkl")
    le_season= joblib.load(base / "le_season_v2.pkl")
    with open(base / "model_summary_v2.json") as f:
        summary = json.load(f)
    return model, scaler, le_genre, le_season, summary

try:
    model, scaler, le_genre, le_season, summary = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"⚠️  Model not found. Run film_investment_v2.py first.\n\n{e}")
    st.stop()


# ── HELPERS ──
GENRES  = sorted(le_genre.classes_.tolist())
SEASONS = sorted(le_season.classes_.tolist())

def predict(budget_m, popularity, vote_avg, vote_count, runtime,
            genre, season, trailer, social_buzz, festival,
            sentiment, creator):
    early_mom = (sentiment*0.40 + social_buzz*0.30 +
                 trailer*0.20 + (festival/10*100)*0.10)
    g = le_genre.transform([genre])[0]   if genre  in le_genre.classes_  else 0
    s = le_season.transform([season])[0] if season in le_season.classes_ else 0
    X = np.array([[budget_m, popularity, vote_avg, vote_count, runtime,
                   g, s, trailer, social_buzz, festival, sentiment, creator, early_mom]])
    X_s  = scaler.transform(X)
    prob = model.predict_proba(X_s)[0][1]
    return round(prob * 100, 1), round(early_mom, 1)

def tier(p):
    if p >= 70: return "INVEST WITH CONFIDENCE", "invest",  "#2ECC71"
    if p >= 50: return "INVEST WITH CAUTION",    "caution", "#F39C12"
    if p >= 35: return "REVIEW DETAILS",         "review",  "#E67E22"
    return             "DO NOT INVEST",           "avoid",   "#E74C3C"

def gauge(prob, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={'suffix': '%', 'font': {'size': 42, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#8899AA',
                     'tickfont': {'color': '#8899AA', 'size': 11}},
            'bar':  {'color': color, 'thickness': 0.28},
            'bgcolor': '#161F2E',
            'steps': [
                {'range': [0,  35], 'color': '#1A0808'},
                {'range': [35, 50], 'color': '#1A0F00'},
                {'range': [50, 70], 'color': '#1A1400'},
                {'range': [70,100], 'color': '#0A1F0A'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8, 'value': prob
            }
        }
    ))
    fig.update_layout(
        height=280, margin=dict(t=20,b=0,l=20,r=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#FFFFFF'}
    )
    return fig

def portfolio_chart(port_df):
    colors = port_df['Decision'].map({
        'INVEST':'#2ECC71','CAUTION':'#F39C12',
        'REVIEW':'#E67E22','AVOID':'#E74C3C'
    })
    fig = go.Figure(go.Bar(
        x=port_df['Film'], y=port_df['Prob %'],
        marker_color=colors,
        text=port_df['Prob %'].apply(lambda x: f"{x:.0f}%"),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>'
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="#2ECC71",
                  annotation_text="Invest threshold (70%)", annotation_font_color="#2ECC71")
    fig.add_hline(y=50, line_dash="dash", line_color="#F39C12",
                  annotation_text="Caution threshold (50%)", annotation_font_color="#F39C12")
    fig.update_layout(
        height=320, plot_bgcolor='#161F2E', paper_bgcolor='rgba(0,0,0,0)',
        font={'color':'#FFFFFF'},
        xaxis={'gridcolor':'#1A2B3C', 'tickfont':{'color':'#8899AA'}},
        yaxis={'gridcolor':'#1A2B3C', 'tickfont':{'color':'#8899AA'}, 'range':[0,110]},
        margin=dict(t=30,b=10,l=10,r=10), showlegend=False
    )
    return fig

def radar_chart(film_name, scores):
    categories = ['Trailer\nEngagement','Social\nBuzz','Festival\nScore',
                  'TABB\nSentiment','Creator\nTrack Record']
    vals = [scores['trailer'], scores['social'], scores['festival']*10,
            scores['sentiment'], scores['creator']*10]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=categories + [categories[0]],
        fill='toself', fillcolor='rgba(245,166,35,0.15)',
        line=dict(color='#F5A623', width=2),
        name=film_name
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100],
                   gridcolor='#1A2B3C', tickfont={'color':'#8899AA'}),
                   angularaxis=dict(tickfont={'color':'#AABBCC'}),
                   bgcolor='#161F2E'),
        paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
        height=280, margin=dict(t=20,b=10,l=30,r=30)
    )
    return fig


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.markdown("# 🎬  Film Investment Intelligence")
    st.markdown(
        "**Version 2.0** — Pre-Release Signals · TABB Sentiment · Portfolio Analysis  "
        '<span class="new-badge">SPRINT 6</span>', unsafe_allow_html=True
    )
with col_h2:
    st.markdown(f"""
    <div style='background:#161F2E;border-radius:8px;padding:12px;text-align:center;border:1px solid #F5A623'>
    <div style='color:#F5A623;font-size:22px;font-weight:bold'>{summary['accuracy_v2']*100:.1f}%</div>
    <div style='color:#8899AA;font-size:11px'>Model Accuracy</div>
    <div style='color:#4A9EDB;font-size:20px;font-weight:bold;margin-top:6px'>{summary['auc_v2']:.3f}</div>
    <div style='color:#8899AA;font-size:11px'>AUC Score</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Single Film Evaluation", "💼  Portfolio Analysis", "📊  Model Insights"])


# ════════════════════════════════════════════
# TAB 1 — SINGLE FILM
# ════════════════════════════════════════════
with tab1:
    with st.sidebar:
        st.markdown("### 🎬  Film Details")
        st.markdown("---")

        st.markdown("**📦  Original Signals**")
        budget_m   = st.slider("Budget ($M)",        0.1, 500.0,  5.0,  0.1)
        popularity = st.slider("Popularity Score",   0.0, 300.0,  80.0, 1.0)
        vote_avg   = st.slider("Vote Average",        1.0,  10.0,  7.2,  0.1)
        vote_count = st.slider("Vote Count",          0,   15000, 3500, 100)
        runtime    = st.slider("Runtime (mins)",      60,    240,   90,   1)
        genre      = st.selectbox("Genre",    GENRES,  index=GENRES.index("Horror")  if "Horror" in GENRES else 0)
        season     = st.selectbox("Season",   SEASONS, index=SEASONS.index("Holiday") if "Holiday" in SEASONS else 0)

        st.markdown("---")
        st.markdown('**🆕  Pre-Release Signals** <span class="new-badge">NEW</span>', unsafe_allow_html=True)
        st.caption("*Peter's recommendation — earlier-stage signals*")

        trailer   = st.slider("Trailer Engagement (0–100)",      0, 100,  75, 1)
        social    = st.slider("Social Buzz Index (0–100)",        0, 100,  70, 1)
        festival  = st.slider("Festival Score (0–10)",            0.0, 10.0, 3.0, 0.5)
        sentiment = st.slider("TABB Sentiment Score (0–100) ⭐",  0, 100,  78, 1,
                              help="Lunim/TABB community creator & audience sentiment — Peter's #1 requested signal")
        creator   = st.slider("Creator Track Record (0–10)",      0.0, 10.0, 5.0, 0.5)

        st.markdown("---")
        film_name_input = st.text_input("Film Name (optional)", "My Film")
        evaluate_btn = st.button("🎯  Evaluate Film", use_container_width=True, type="primary")
        add_portfolio = st.button("➕  Add to Portfolio", use_container_width=True)

    # ── AUTO PREDICT ──
    prob, early_mom = predict(budget_m, popularity, vote_avg, vote_count, runtime,
                              genre, season, trailer, social, festival, sentiment, creator)
    decision_label, decision_class, decision_color = tier(prob)

    # ── MAIN RESULTS ──
    col1, col2, col3 = st.columns([1.2, 1.6, 1.2])

    with col1:
        st.markdown("#### Investment Probability")
        st.plotly_chart(gauge(prob, decision_color), use_container_width=True)

    with col2:
        st.markdown("#### Decision")
        st.markdown(f"""
        <div class='metric-card {decision_class}'>
            <div style='font-size:22px;font-weight:bold;color:{decision_color}'>{decision_label}</div>
            <div style='color:#8899AA;font-size:13px;margin-top:6px'>
            {genre} · ${budget_m}M · {season} · {runtime} min
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### KPI Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Probability",   f"{prob}%")
        m2.metric("Early Momentum", f"{early_mom:.0f}/100")
        m3.metric("TABB Sentiment", f"{sentiment}/100")

        st.markdown("#### Pre-Release Signal Profile")
        st.plotly_chart(
            radar_chart(film_name_input, {'trailer':trailer,'social':social,
                                          'festival':festival,'sentiment':sentiment,'creator':creator}),
            use_container_width=True
        )

    with col3:
        st.markdown("#### Signal Breakdown")
        signals = {
            "vote_count":    ("Original",    round(vote_count/15000*100, 1)),
            "popularity":    ("Original",    round(popularity/300*100,   1)),
            "TABB Sentiment":("🆕 New",      sentiment),
            "Social Buzz":   ("🆕 New",      social),
            "Trailer Eng.":  ("🆕 New",      trailer),
            "Festival":      ("🆕 New",      festival*10),
            "Creator Track": ("🆕 New",      creator*10),
        }
        for sig, (stype, val) in signals.items():
            bar_color = "#F5A623" if "New" in stype else "#3A8FDB"
            st.markdown(f"""
            <div style='margin:4px 0'>
                <div style='display:flex;justify-content:space-between;font-size:12px'>
                    <span style='color:#AABBCC'>{sig}</span>
                    <span style='color:{bar_color};font-weight:bold'>{val:.0f}</span>
                </div>
                <div style='background:#161F2E;border-radius:4px;height:6px'>
                    <div style='background:{bar_color};width:{val}%;height:6px;border-radius:4px'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── ADD TO HISTORY ──
    if evaluate_btn or add_portfolio:
        entry = {
            'Film': film_name_input, 'Genre': genre, 'Budget $M': budget_m,
            'Season': season, 'TABB Sentiment': sentiment,
            'Early Momentum': early_mom, 'Prob %': prob, 'Decision': decision_label
        }
        st.session_state.history.append(entry)
        if add_portfolio:
            st.session_state.portfolio.append({
                'name': film_name_input, 'genre': genre, 'budget_m': budget_m,
                'season': season, 'prob': prob, 'decision': decision_label
            })
            st.toast(f"✅ {film_name_input} added to portfolio!", icon="✅")

    # ── HISTORY TABLE ──
    if st.session_state.history:
        st.markdown("---")
        st.markdown("#### 📋  Evaluation History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("🗑️  Clear History"):
            st.session_state.history = []
            st.rerun()


# ════════════════════════════════════════════
# TAB 2 — PORTFOLIO
# ════════════════════════════════════════════
with tab2:
    st.markdown("### 💼  Portfolio Analysis")
    st.caption("*Peter's second enhancement — compare multiple films, assess risk, suggest capital allocation*")

    # Use portfolio from session or load demo
    if st.session_state.portfolio:
        portfolio_data = st.session_state.portfolio
        st.info(f"📋  {len(portfolio_data)} film(s) in your portfolio from Tab 1.")
    else:
        st.info("💡  No films added yet. Use Tab 1 → 'Add to Portfolio', or use the demo below.")
        portfolio_data = None

    # Demo portfolio
    with st.expander("🎬  Load Demo Portfolio (5 Films)", expanded=portfolio_data is None):
        if st.button("▶️  Run Demo Portfolio Analysis"):
            demo = [
                dict(name="Horror Micro",   genre="Horror",           budget_m=5,   season="Holiday",
                     trailer=75, social_buzz=70, festival_score_val=3.0, sentiment=78, creator=5.0,
                     budget_m_=5,   popularity=80, vote_avg=7.2, vote_count=3500, runtime=87),
                dict(name="Action Summer",  genre="Action",           budget_m=50,  season="Summer",
                     trailer=62, social_buzz=65, festival_score_val=0.0, sentiment=64, creator=7.0,
                     popularity=65, vote_avg=6.8, vote_count=5200, runtime=118),
                dict(name="Comedy Spring",  genre="Comedy",           budget_m=15,  season="Spring",
                     trailer=42, social_buzz=38, festival_score_val=0.0, sentiment=49, creator=4.0,
                     popularity=40, vote_avg=6.1, vote_count=1200, runtime=95),
                dict(name="Drama Prestige", genre="Drama",            budget_m=200, season="Off-Season",
                     trailer=25, social_buzz=22, festival_score_val=7.5, sentiment=38, creator=8.5,
                     popularity=20, vote_avg=5.8, vote_count=180,  runtime=165),
                dict(name="Sci-Fi Holiday", genre="Science Fiction",  budget_m=80,  season="Holiday",
                     trailer=88, social_buzz=85, festival_score_val=2.0, sentiment=82, creator=7.5,
                     popularity=95, vote_avg=7.5, vote_count=7800, runtime=130),
            ]
            rows = []
            for d in demo:
                p, em = predict(
                    d.get('budget_m',d.get('budget_m_',10)),
                    d.get('popularity',50), d.get('vote_avg',6.5),
                    d.get('vote_count',1000), d.get('runtime',100),
                    d['genre'], d['season'],
                    d['trailer'], d['social_buzz'], d['festival_score_val'],
                    d['sentiment'], d['creator']
                )
                dl, dc, dcol = tier(p)
                rows.append({
                    'Film': d['name'], 'Genre': d['genre'],
                    'Budget $M': d['budget_m'], 'Season': d['season'],
                    'TABB Sentiment': d['sentiment'],
                    'Early Momentum': em, 'Prob %': p,
                    'Decision': dl, 'Color': dcol
                })
            port_df = pd.DataFrame(rows).sort_values('Prob %', ascending=False)

            # ── Portfolio bar chart ──
            st.plotly_chart(portfolio_chart(port_df), use_container_width=True)

            # ── Portfolio table ──
            st.markdown("#### Film Comparison Table")
            display_df = port_df.drop(columns=['Color'])

            # Capital allocation
            investable = port_df[port_df['Prob %'] >= 50]
            if len(investable):
                total_p = investable['Prob %'].sum()
                port_df.loc[investable.index, 'Allocation %'] = (
                    investable['Prob %'] / total_p * 100
                ).round(1)
                port_df.loc[~port_df.index.isin(investable.index), 'Allocation %'] = 0.0
                display_df = port_df.drop(columns=['Color'])

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # ── Portfolio metrics ──
            st.markdown("#### Portfolio-Level Metrics")
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Average Probability",   f"{port_df['Prob %'].mean():.1f}%")
            pm2.metric("Films to Invest",        len(port_df[port_df['Prob %']>=50]))
            pm3.metric("Films to Avoid",         len(port_df[port_df['Prob %']<35]))
            pm4.metric("Recommended Deploy",     f"${port_df[port_df['Prob %']>=50]['Budget $M'].sum()}M")

            # Risk summary
            st.markdown("#### Risk Assessment")
            genres_count = port_df['Genre'].value_counts()
            r1, r2 = st.columns(2)
            with r1:
                fig_risk = px.pie(
                    port_df, names='Genre', values='Prob %',
                    title='Portfolio by Genre',
                    color_discrete_sequence=['#F5A623','#3A8FDB','#2ECC71','#E74C3C','#9B59B6']
                )
                fig_risk.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color':'#FFFFFF'}, height=260,
                    margin=dict(t=40,b=0,l=0,r=0)
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            with r2:
                fig_scatter = px.scatter(
                    port_df, x='Budget $M', y='Prob %', color='Decision',
                    size='TABB Sentiment', hover_name='Film',
                    title='Budget vs Probability (bubble = TABB sentiment)',
                    color_discrete_map={
                        'INVEST WITH CONFIDENCE':'#2ECC71',
                        'INVEST WITH CAUTION':'#F39C12',
                        'REVIEW DETAILS':'#E67E22',
                        'DO NOT INVEST':'#E74C3C'
                    }
                )
                fig_scatter.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161F2E',
                    font={'color':'#FFFFFF'}, height=260,
                    xaxis={'gridcolor':'#1A2B3C'}, yaxis={'gridcolor':'#1A2B3C'},
                    margin=dict(t=40,b=0,l=0,r=0)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ════════════════════════════════════════════
with tab3:
    st.markdown("### 📊  Model Insights — V1 vs V2")

    # ── V1 vs V2 comparison ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### V1 — Original Model")
        st.markdown(f"""
        <div class='metric-card caution'>
        <div style='font-size:32px;font-weight:bold;color:#F39C12'>{summary['accuracy_v1']*100:.1f}%</div>
        <div style='color:#8899AA'>Accuracy</div>
        <div style='font-size:22px;font-weight:bold;color:#F39C12;margin-top:8px'>{summary['auc_v1']:.3f}</div>
        <div style='color:#8899AA'>AUC Score</div>
        <div style='color:#8899AA;margin-top:8px'>7 features — historical data only</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("#### V2 — Enhanced Model")
        st.markdown(f"""
        <div class='metric-card invest'>
        <div style='font-size:32px;font-weight:bold;color:#2ECC71'>{summary['accuracy_v2']*100:.1f}%</div>
        <div style='color:#8899AA'>Accuracy</div>
        <div style='font-size:22px;font-weight:bold;color:#2ECC71;margin-top:8px'>{summary['auc_v2']:.3f}</div>
        <div style='color:#8899AA'>AUC Score</div>
        <div style='color:#8899AA;margin-top:8px'>13 features — includes pre-release signals</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Feature Importance — V2")
    fi = summary['feature_importance']
    fi_df = pd.DataFrame({'Feature': list(fi.keys()), 'Importance': list(fi.values())})
    fi_df = fi_df.sort_values('Importance', ascending=True)
    fi_df['Type'] = fi_df['Feature'].apply(
        lambda f: '🆕 Pre-Release' if f in ['trailer_engagement_score','social_buzz_index',
            'festival_score','community_sentiment_score','creator_track_record','early_momentum']
        else '📦 Original'
    )
    fi_df['Color'] = fi_df['Type'].map({'🆕 Pre-Release':'#F5A623','📦 Original':'#3A8FDB'})

    fig_fi = go.Figure(go.Bar(
        x=fi_df['Importance']*100, y=fi_df['Feature'],
        orientation='h',
        marker_color=fi_df['Color'],
        text=[f"{v*100:.1f}%" for v in fi_df['Importance']],
        textposition='outside'
    ))
    fig_fi.update_layout(
        height=400, plot_bgcolor='#161F2E', paper_bgcolor='rgba(0,0,0,0)',
        font={'color':'#FFFFFF'},
        xaxis={'gridcolor':'#1A2B3C','title':'Importance %','tickfont':{'color':'#8899AA'}},
        yaxis={'gridcolor':'#1A2B3C','tickfont':{'color':'#AABBCC'}},
        margin=dict(t=20,b=10,l=10,r=60)
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    pre_total  = sum(v for k,v in fi.items() if k in
        ['trailer_engagement_score','social_buzz_index','festival_score',
         'community_sentiment_score','creator_track_record','early_momentum'])
    orig_total = 1 - pre_total

    col_a, col_b = st.columns(2)
    col_a.metric("Pre-Release Signals Contribution",  f"{pre_total*100:.1f}%",
                 help="Total importance of the 6 new signals added in V2")
    col_b.metric("Original Signals Contribution",     f"{orig_total*100:.1f}%",
                 help="Total importance of the 7 original V1 features")

    st.markdown("---")
    st.caption("🎬  Film Investment Intelligence v2.0 — Sprint 6  |  Pre-Release Signals · TABB Sentiment · Portfolio Layer  |  Tarek ElNaggar · Lunim · 2025")
