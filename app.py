import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load the form
model = joblib.load('film_investment_model/model.pkl')
scaler = joblib.load('film_investment_model/scaler.pkl')
le_genre = joblib.load('film_investment_model/le_genre.pkl')
le_season = joblib.load('film_investment_model/le_season.pkl')

# Page setup
st.set_page_config(
    page_title="Film Investment Intelligence",
    page_icon="🎬",
    layout="wide"
)

# Header
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>
        🎬 Film Investment Intelligence
    </h1>
    <p style='text-align: center; color: #7f8c8d; font-size: 18px;'>
        AI-Powered Investment Decision Tool — Powered by Lunim
    </p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar — Film Data
st.sidebar.title("📋 Film Data")
st.sidebar.markdown("---")

budget = st.sidebar.slider("💰 Budget (million dollars)", 1, 300, 30)
popularity = st.sidebar.slider("🔥 popularity", 1, 200, 50)
vote_avg = st.sidebar.slider("⭐ Popularity", 1.0, 10.0, 7.0, 0.1)
vote_count = st.sidebar.slider("👥 Expected Rating", 100, 10000, 2000)
runtime = st.sidebar.slider("⏱️ Film duration (minute)", 60, 240, 110)

genre = st.sidebar.selectbox("🎭 Film type", 
    sorted(le_genre.classes_.tolist()))

season = st.sidebar.selectbox("📅 Release season",
    sorted(le_season.classes_.tolist()))

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🚀 Investment analysis", use_container_width=True)

# Prediction
if predict_btn:
    genre_enc = le_genre.transform([genre])[0]
    season_enc = le_season.transform([season])[0]
    
    input_data = np.array([[budget * 1e6, popularity, vote_avg,
                            vote_count, runtime, genre_enc, season_enc]])
    input_scaled = scaler.transform(input_data)
    
    prob = model.predict_proba(input_scaled)[0][1]
    decision = model.predict(input_scaled)[0]

    # Main Outcome
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎯 Probability of Success", f"{prob*100:.1f}%")

    with col2:
        roi_est = (prob * 4)
        st.metric("💵 Expected ROI", f"{roi_est:.1f}x")

    with col3:
        risk = "Low" if prob > 0.7 else "Medium" if prob > 0.5 else "High"
        st.metric("⚠️ Risk Score", risk)

    st.markdown("---")

    # Investment Decision
    if prob >= 0.7:
        st.success("## ✅ Decision: Invest with Confidence")
    elif prob >= 0.5:
        st.warning("## ⚠️ Decision: Invest with Caution")
    elif prob >= 0.35:
        st.warning("## 🔶 Decision: Review Details Before Deciding")
    else:
        st.error("## ❌ Decision: Do Not Invest")

    st.markdown("---")

    # Gauge Chart
    col_a, col_b = st.columns(2)

    with col_a:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            title={'text': "Probability of Investment Success", 'font': {'size': 16}},
            delta={'reference': 56, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2ecc71" if prob > 0.6 else "#e74c3c"},
                'steps': [
                    {'range': [0, 35], 'color': '#fadbd8'},
                    {'range': [35, 60], 'color': '#fdebd0'},
                    {'range': [60, 100], 'color': '#d5f5e3'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': 56
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_b:
        # Summary Data
        st.markdown("### 📊 Movie Data Summary")
        summary_data = {
            'Factor': ['Budget', 'Popularity', 'Rating', 'Voters', 'Duration', 'Genre', 'Season'],
            'Value': [f'${budget}M', popularity, vote_avg, vote_count, f'{runtime} mins', genre, season]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

else:
    # Welcome Screen
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2 style='color: #2c3e50;'>👈 Enter movie data from side menu </h2>
        <p style='color: #7f8c8d; font-size: 16px;'>
            Then click "Investment Analysis" to get an instant decision
        </p>
    </div>
    """, unsafe_allow_html=True)

    # General Statistics
    st.markdown("### 📈 Model Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎬 Movies Analyzed", "3,229")
    c2.metric("🎯 Model Accuracy", "75.4%")
    c3.metric("📊 AUC Score", "0.826")
    c4.metric("🏆 Best Genre", "Horror")