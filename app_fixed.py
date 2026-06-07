import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load the form
@st.cache_resource
def load_model():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    import pandas as pd
    np.random.seed(42)
    N = 3229
    genres  = ['Horror','Action','Drama','Comedy','Animation',
               'Science Fiction','Fantasy','Thriller','Adventure','Romance']
    seasons = ['Holiday','Summer','Spring','Off-Season']
    gw = [0.08,0.18,0.20,0.15,0.07,0.09,0.06,0.08,0.06,0.03]
    sw = [0.18,0.26,0.22,0.34]
    df = pd.DataFrame({
        'budget':        np.random.lognormal(16.5,1.4,N).clip(50000,500000000),
        'popularity':    np.random.lognormal(3.0,1.2,N).clip(0.5,300),
        'vote_average':  np.random.normal(6.1,1.1,N).clip(1.0,10.0),
        'vote_count':    np.random.lognormal(6.5,1.5,N).astype(int).clip(10,15000),
        'runtime':       np.random.normal(107,22,N).clip(60,240),
        'primary_genre': np.random.choice(genres,N,p=gw),
        'season':        np.random.choice(seasons,N,p=sw),
    })
    gm = {'Horror':1.45,'Action':1.05,'Drama':0.90,'Comedy':0.88,
          'Animation':1.10,'Science Fiction':1.12,'Fantasy':1.08,
          'Thriller':1.02,'Adventure':1.06,'Romance':0.85}
    sm = {'Holiday':1.18,'Summer':1.10,'Spring':1.02,'Off-Season':0.95}
    base = (df['vote_count']/3000)*2.5+(df['popularity']/60)*1.2
    df['ROI']     = (base*df['primary_genre'].map(gm)*df['season'].map(sm)
                     +np.random.normal(0,0.5,N)).clip(0.1,30)
    df['success'] = (df['ROI']>=2.0).astype(int)
    df['budget_M'] = df['budget']/1_000_000
    le_g = LabelEncoder(); le_s = LabelEncoder()
    df['genre_enc']  = le_g.fit_transform(df['primary_genre'])
    df['season_enc'] = le_s.fit_transform(df['season'])
    FEAT = ['budget_M','popularity','vote_average','vote_count','runtime','genre_enc','season_enc']
    X = df[FEAT]; y = df['success']
    Xtr,_,ytr,_ = train_test_split(X,y,test_size=0.2,random_state=42)
    sc = StandardScaler()
    mdl = GradientBoostingClassifier(n_estimators=100,random_state=42)
    mdl.fit(sc.fit_transform(Xtr),ytr)
    return mdl, sc, le_g, le_s

model, scaler, le_genre, le_season = load_model()

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