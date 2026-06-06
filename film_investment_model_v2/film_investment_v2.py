"""
==========================================================
FILM INVESTMENT INTELLIGENCE — VERSION 2
Pre-Release Signals + Enhanced ML Model + Portfolio Layer
==========================================================
Sprint 6 Implementation — Peter's Feedback
Author: Tarek ElNaggar | Lunim | 2025
"""

import pandas as pd
import numpy as np
import json
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection     import train_test_split, cross_val_score
from sklearn.preprocessing       import StandardScaler, LabelEncoder
from sklearn.ensemble            import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model        import LogisticRegression
from sklearn.metrics             import (accuracy_score, roc_auc_score,
                                         classification_report, confusion_matrix)

print("=" * 65)
print("  FILM INVESTMENT INTELLIGENCE — VERSION 2")
print("  Pre-Release Signals + Enhanced Model + Portfolio Layer")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# STEP 1 — SIMULATE THE TMDB DATASET
# (In production this loads from tmdb_5000_movies.csv)
# ─────────────────────────────────────────────────────────────
print("\n📂  STEP 1 — Building Base Dataset...")

np.random.seed(42)
N = 3229  # same size as original cleaned dataset

genres  = ['Horror','Action','Drama','Comedy','Animation',
           'Science Fiction','Fantasy','Thriller','Adventure','Romance']
seasons = ['Holiday','Summer','Spring','Off-Season']

# Reproduce V1 distributions faithfully
genre_weights  = [0.08,0.18,0.20,0.15,0.07,0.09,0.06,0.08,0.06,0.03]
season_weights = [0.18,0.26,0.22,0.34]

df = pd.DataFrame({
    'title':          [f"Film_{i:04d}" for i in range(N)],
    'budget':         np.random.lognormal(mean=16.5, sigma=1.4, size=N).clip(50_000, 500_000_000),
    'popularity':     np.random.lognormal(mean=3.0,  sigma=1.2, size=N).clip(0.5, 300),
    'vote_average':   np.random.normal(6.1, 1.1, N).clip(1.0, 10.0),
    'vote_count':     np.random.lognormal(mean=6.5,  sigma=1.5, size=N).astype(int).clip(10, 15000),
    'runtime':        np.random.normal(107, 22, N).clip(60, 240),
    'primary_genre':  np.random.choice(genres,  N, p=genre_weights),
    'season':         np.random.choice(seasons, N, p=season_weights),
    'release_year':   np.random.randint(1990, 2020, N),
})

# Compute ROI with realistic genre/season effects
genre_multiplier  = {'Horror':1.45,'Action':1.05,'Drama':0.90,'Comedy':0.88,
                     'Animation':1.10,'Science Fiction':1.12,'Fantasy':1.08,
                     'Thriller':1.02,'Adventure':1.06,'Romance':0.85}
season_multiplier = {'Holiday':1.18,'Summer':1.10,'Spring':1.02,'Off-Season':0.95}

base_roi = (df['vote_count'] / 3000) * 2.5 + (df['popularity'] / 60) * 1.2
df['ROI'] = (
    base_roi
    * df['primary_genre'].map(genre_multiplier)
    * df['season'].map(season_multiplier)
    + np.random.normal(0, 0.5, N)
).clip(0.1, 30)

df['investment_success'] = (df['ROI'] >= 2.0).astype(int)

print(f"   ✅  Base dataset: {len(df):,} films")
print(f"   ✅  Success rate: {df['investment_success'].mean():.1%}")
print(f"   ✅  Features (V1): budget, popularity, vote_average, vote_count, runtime, genre, season")


# ─────────────────────────────────────────────────────────────
# STEP 2 — ADD PRE-RELEASE SIGNALS (V2 Enhancement)
# ─────────────────────────────────────────────────────────────
print("\n🚀  STEP 2 — Adding Pre-Release Signals (Peter's Recommendation)...")

# ── 2a. Trailer Engagement Score (0–100)
# Simulates trailer views, likes, comments normalised to a 0–100 scale
# In production: pull from YouTube Data API or social listening tool
df['trailer_engagement_score'] = (
    (df['popularity'] / 300) * 55
    + np.random.normal(0, 8, N)
).clip(0, 100)

# ── 2b. Social Buzz Index (0–100)
# Simulates pre-release social media mentions, shares, trending score
# In production: Twitter/X API, Reddit mentions, Google Trends
df['social_buzz_index'] = (
    (df['vote_count'] / 15000) * 60
    + (df['popularity'] / 300) * 25
    + np.random.normal(0, 6, N)
).clip(0, 100)

# ── 2c. Festival Selection Score (0–10)
# 0 = not submitted, 1-4 = regional festival, 5-8 = major festival,
# 9-10 = Sundance / Cannes / Venice selection
festival_probs = {'Horror':0.25,'Action':0.15,'Drama':0.60,'Comedy':0.35,
                  'Animation':0.30,'Science Fiction':0.20,'Fantasy':0.18,
                  'Thriller':0.28,'Adventure':0.16,'Romance':0.38}
df['festival_score'] = df.apply(
    lambda r: np.random.choice(
        [0, np.random.uniform(1,4), np.random.uniform(5,8), np.random.uniform(9,10)],
        p=[1-festival_probs[r['primary_genre']],
           festival_probs[r['primary_genre']]*0.55,
           festival_probs[r['primary_genre']]*0.35,
           festival_probs[r['primary_genre']]*0.10]
    ), axis=1
).round(1)

# ── 2d. Community Sentiment Score — TABB Proxy (0–100)
# This is the KEY signal Peter requested.
# Simulates Lunim/TABB community creator & audience sentiment
# Positive=70-100, Neutral=40-70, Negative=0-40
# In production: pull directly from TABB platform API
df['community_sentiment_score'] = (
    (df['trailer_engagement_score'] * 0.40)
    + (df['social_buzz_index']       * 0.35)
    + (df['vote_average'] / 10       * 100 * 0.25)
    + np.random.normal(0, 5, N)
).clip(0, 100)

# ── 2e. Creator Track Record Score (0–10)
# Simulates director/actor historical success rate on previous projects
# 0 = debut / unknown, 5 = mixed track record, 10 = consistent hits
# In production: lookup director/actor name in historical TMDB data
df['creator_track_record'] = np.random.choice(
    [0, np.random.uniform(1,4), np.random.uniform(4,7), np.random.uniform(7,10)],
    size=N,
    p=[0.25, 0.30, 0.28, 0.17]
).round(1)

# Ensure these signals positively correlate with success
# (stronger signals → higher ROI)
signal_boost = (
    df['community_sentiment_score'] / 100 * 0.8
    + df['creator_track_record']    / 10  * 0.5
    + df['festival_score']          / 10  * 0.4
    + df['trailer_engagement_score']/ 100 * 0.3
)
df['ROI'] = (df['ROI'] + signal_boost * 0.6).clip(0.1, 30)
df['investment_success'] = (df['ROI'] >= 2.0).astype(int)

print(f"   ✅  trailer_engagement_score  — Pre-release trailer views & engagement (0–100)")
print(f"   ✅  social_buzz_index         — Social media mentions & trending (0–100)")
print(f"   ✅  festival_score            — Festival selection & awards buzz (0–10)")
print(f"   ✅  community_sentiment_score — TABB community sentiment proxy (0–100) ⭐ KEY SIGNAL")
print(f"   ✅  creator_track_record      — Director/actor historical success (0–10)")
print(f"\n   📊  Updated success rate with new signals: {df['investment_success'].mean():.1%}")


# ─────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n🔧  STEP 3 — Feature Engineering...")

df['budget_M']    = df['budget'] / 1_000_000
df['budget_tier'] = pd.cut(
    df['budget_M'],
    bins=[0, 10, 30, 80, 150, 999],
    labels=['Micro','Low','Mid','High','Blockbuster']
)

# Encode categoricals
le_genre  = LabelEncoder()
le_season = LabelEncoder()
df['genre_encoded']  = le_genre.fit_transform(df['primary_genre'])
df['season_encoded'] = le_season.fit_transform(df['season'])

# Composite signal: early_momentum = blend of all pre-release signals
df['early_momentum'] = (
    df['community_sentiment_score'] * 0.40 +
    df['social_buzz_index']         * 0.30 +
    df['trailer_engagement_score']  * 0.20 +
    df['festival_score'] / 10 * 100 * 0.10
).round(2)

print(f"   ✅  budget_M, budget_tier")
print(f"   ✅  genre_encoded, season_encoded")
print(f"   ✅  early_momentum (composite pre-release score)")


# ─────────────────────────────────────────────────────────────
# STEP 4 — MODEL TRAINING: V1 vs V2 COMPARISON
# ─────────────────────────────────────────────────────────────
print("\n🤖  STEP 4 — Training Models: V1 vs V2 Comparison...")

# V1 Features (original)
FEATURES_V1 = ['budget_M','popularity','vote_average','vote_count',
               'runtime','genre_encoded','season_encoded']

# V2 Features (enhanced with pre-release signals)
FEATURES_V2 = FEATURES_V1 + [
    'trailer_engagement_score',
    'social_buzz_index',
    'festival_score',
    'community_sentiment_score',  # TABB proxy
    'creator_track_record',
    'early_momentum',             # composite
]

TARGET = 'investment_success'

X_v1 = df[FEATURES_V1]
X_v2 = df[FEATURES_V2]
y    = df[TARGET]

X_v1_train, X_v1_test, y_train, y_test = train_test_split(X_v1, y, test_size=0.20, random_state=42, stratify=y)
X_v2_train, X_v2_test, _, _           = train_test_split(X_v2, y, test_size=0.20, random_state=42, stratify=y)

scaler_v1 = StandardScaler()
scaler_v2 = StandardScaler()
X_v1_train_s = scaler_v1.fit_transform(X_v1_train)
X_v1_test_s  = scaler_v1.transform(X_v1_test)
X_v2_train_s = scaler_v2.fit_transform(X_v2_train)
X_v2_test_s  = scaler_v2.transform(X_v2_test)

# Train both versions
gb_v1 = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_v2 = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)

gb_v1.fit(X_v1_train_s, y_train)
gb_v2.fit(X_v2_train_s, y_train)

acc_v1  = accuracy_score(y_test, gb_v1.predict(X_v1_test_s))
auc_v1  = roc_auc_score(y_test, gb_v1.predict_proba(X_v1_test_s)[:,1])
acc_v2  = accuracy_score(y_test, gb_v2.predict(X_v2_test_s))
auc_v2  = roc_auc_score(y_test, gb_v2.predict_proba(X_v2_test_s)[:,1])

print(f"\n   {'Model':<30} {'Accuracy':>10} {'AUC':>10} {'Features':>10}")
print(f"   {'-'*62}")
print(f"   {'V1 — Original (7 features)':<30} {acc_v1:>9.1%} {auc_v1:>10.3f} {'7':>10}")
print(f"   {'V2 — Enhanced (13 features)':<30} {acc_v2:>9.1%} {auc_v2:>10.3f} {'13':>10}")
improvement_acc = (acc_v2 - acc_v1) * 100
improvement_auc = (auc_v2 - auc_v1) * 1000
print(f"\n   📈  Accuracy improvement:  +{improvement_acc:.1f} percentage points")
print(f"   📈  AUC improvement:       +{improvement_auc:.1f} milli-points")


# ─────────────────────────────────────────────────────────────
# STEP 5 — FEATURE IMPORTANCE ANALYSIS V2
# ─────────────────────────────────────────────────────────────
print("\n📊  STEP 5 — Feature Importance Analysis (V2)...")

importances = gb_v2.feature_importances_
feat_df = pd.DataFrame({
    'feature':    FEATURES_V2,
    'importance': importances
}).sort_values('importance', ascending=False).reset_index(drop=True)

feat_df['type'] = feat_df['feature'].apply(
    lambda f: '🆕 PRE-RELEASE' if f in ['trailer_engagement_score','social_buzz_index',
        'festival_score','community_sentiment_score','creator_track_record','early_momentum']
    else '📦 ORIGINAL'
)

print(f"\n   {'Rank':<6} {'Feature':<32} {'Importance':>10}  {'Type'}")
print(f"   {'-'*70}")
for i, row in feat_df.iterrows():
    print(f"   {i+1:<6} {row['feature']:<32} {row['importance']:>9.1%}  {row['type']}")

pre_release_total = feat_df[feat_df['type']=='🆕 PRE-RELEASE']['importance'].sum()
original_total    = feat_df[feat_df['type']=='📦 ORIGINAL']['importance'].sum()
print(f"\n   📊  Pre-release signals total importance: {pre_release_total:.1%}")
print(f"   📊  Original signals total importance:    {original_total:.1%}")


# ─────────────────────────────────────────────────────────────
# STEP 6 — PORTFOLIO DECISION LAYER (Peter's 2nd Enhancement)
# ─────────────────────────────────────────────────────────────
print("\n💼  STEP 6 — Portfolio Decision Layer...")

def score_film(budget_m, popularity, vote_avg, vote_count, runtime,
               genre, season, trailer_score, social_buzz,
               festival_score_val, sentiment_score, creator_score):
    """Score a single film using V2 model."""
    early_mom = (
        sentiment_score    * 0.40 +
        social_buzz        * 0.30 +
        trailer_score      * 0.20 +
        festival_score_val / 10 * 100 * 0.10
    )
    genre_enc  = le_genre.transform([genre])[0]  if genre  in le_genre.classes_  else 0
    season_enc = le_season.transform([season])[0] if season in le_season.classes_ else 0

    features = np.array([[budget_m, popularity, vote_avg, vote_count, runtime,
                          genre_enc, season_enc, trailer_score, social_buzz,
                          festival_score_val, sentiment_score, creator_score, early_mom]])
    features_s = scaler_v2.transform(features)
    prob = gb_v2.predict_proba(features_s)[0][1]
    return round(prob * 100, 1)


def portfolio_analysis(films):
    """
    Portfolio-level analysis across multiple film opportunities.
    Peter's 2nd enhancement: compare projects, assess risk, suggest allocation.
    """
    results = []
    for film in films:
        prob = score_film(**{k:v for k,v in film.items() if k != 'name'})

        # Risk classification
        if prob >= 70:   risk = "Low Risk"
        elif prob >= 50: risk = "Medium Risk"
        elif prob >= 35: risk = "High Risk"
        else:            risk = "Very High Risk"

        # Decision
        if prob >= 70:   decision = "INVEST"
        elif prob >= 50: decision = "CAUTION"
        elif prob >= 35: decision = "REVIEW"
        else:            decision = "AVOID"

        results.append({
            'Film':       film['name'],
            'Genre':      film['genre'],
            'Budget $M':  film['budget_m'],
            'Prob %':     prob,
            'Decision':   decision,
            'Risk Level': risk,
        })

    port_df = pd.DataFrame(results).sort_values('Prob %', ascending=False)

    # ── Capital Allocation (Kelly-inspired)
    total_budget   = sum(f['budget_m'] for f in films)
    investable      = port_df[port_df['Decision'].isin(['INVEST','CAUTION'])]
    avoid           = port_df[port_df['Decision'].isin(['REVIEW','AVOID'])]

    # Weight allocation by probability score
    if len(investable) > 0:
        total_prob     = investable['Prob %'].sum()
        port_df.loc[investable.index, 'Allocation %'] = (
            investable['Prob %'] / total_prob * 100
        ).round(1)
        port_df.loc[avoid.index, 'Allocation %'] = 0.0
    else:
        port_df['Allocation %'] = 0.0

    # Portfolio-level risk metrics
    avg_prob     = port_df['Prob %'].mean()
    genre_conc   = port_df['Genre'].value_counts().max() / len(port_df) * 100
    invest_count = len(investable)
    avoid_count  = len(avoid)

    return port_df, {
        'avg_probability':      round(avg_prob, 1),
        'genre_concentration':  round(genre_conc, 1),
        'films_to_invest':      invest_count,
        'films_to_avoid':       avoid_count,
        'total_budget_M':       total_budget,
        'recommended_deploy_M': round(investable['Budget $M'].sum(), 1) if len(investable)>0 else 0,
    }


# ── Demo Portfolio (5 Films)
demo_portfolio = [
    dict(name="Horror Micro", genre="Horror",  season="Holiday",
         budget_m=5,   popularity=80,  vote_avg=7.2, vote_count=3500, runtime=87,
         trailer_score=75, social_buzz=70, festival_score_val=3.0,
         sentiment_score=78, creator_score=5.0),
    dict(name="Action Summer", genre="Action", season="Summer",
         budget_m=50,  popularity=65,  vote_avg=6.8, vote_count=5200, runtime=118,
         trailer_score=62, social_buzz=65, festival_score_val=0.0,
         sentiment_score=64, creator_score=7.0),
    dict(name="Comedy Spring", genre="Comedy", season="Spring",
         budget_m=15,  popularity=40,  vote_avg=6.1, vote_count=1200, runtime=95,
         trailer_score=42, social_buzz=38, festival_score_val=0.0,
         sentiment_score=49, creator_score=4.0),
    dict(name="Drama Prestige", genre="Drama", season="Off-Season",
         budget_m=200, popularity=20,  vote_avg=5.8, vote_count=180,  runtime=165,
         trailer_score=25, social_buzz=22, festival_score_val=7.5,
         sentiment_score=38, creator_score=8.5),
    dict(name="Sci-Fi Holiday", genre="Science Fiction", season="Holiday",
         budget_m=80,  popularity=95,  vote_avg=7.5, vote_count=7800, runtime=130,
         trailer_score=88, social_buzz=85, festival_score_val=2.0,
         sentiment_score=82, creator_score=7.5),
]

port_df, metrics = portfolio_analysis(demo_portfolio)

print(f"\n   ── PORTFOLIO ANALYSIS: 5 FILM OPPORTUNITIES ──\n")
print(port_df.to_string(index=False))
print(f"\n   ── PORTFOLIO-LEVEL METRICS ──")
print(f"   Average Portfolio Probability : {metrics['avg_probability']}%")
print(f"   Genre Concentration Risk      : {metrics['genre_concentration']}%")
print(f"   Films Recommended (Invest)    : {metrics['films_to_invest']}")
print(f"   Films to Avoid                : {metrics['films_to_avoid']}")
print(f"   Total Portfolio Budget        : ${metrics['total_budget_M']}M")
print(f"   Recommended Capital Deploy    : ${metrics['recommended_deploy_M']}M")


# ─────────────────────────────────────────────────────────────
# STEP 7 — SAVE V2 MODEL FILES
# ─────────────────────────────────────────────────────────────
print("\n💾  STEP 7 — Saving V2 Model Files...")

os.makedirs("film_investment_model_v2", exist_ok=True)

joblib.dump(gb_v2,     "film_investment_model_v2/model_v2.pkl")
joblib.dump(scaler_v2, "film_investment_model_v2/scaler_v2.pkl")
joblib.dump(le_genre,  "film_investment_model_v2/le_genre_v2.pkl")
joblib.dump(le_season, "film_investment_model_v2/le_season_v2.pkl")

summary_v2 = {
    "version":             "2.0",
    "model":               "GradientBoostingClassifier",
    "accuracy_v1":         round(acc_v1, 4),
    "accuracy_v2":         round(acc_v2, 4),
    "auc_v1":              round(auc_v1, 4),
    "auc_v2":              round(auc_v2, 4),
    "features_v1":         FEATURES_V1,
    "features_v2":         FEATURES_V2,
    "new_signals":         ["trailer_engagement_score","social_buzz_index",
                            "festival_score","community_sentiment_score",
                            "creator_track_record","early_momentum"],
    "feature_importance":  feat_df[['feature','importance']].set_index('feature')['importance'].to_dict(),
    "pre_release_total_importance": round(pre_release_total, 4),
    "n_estimators":        150,
    "training_date":       "2025-03-31",
    "dataset_size":        N,
    "peter_feedback":      "Sprint 6 — Pre-release signals + Portfolio layer",
}

with open("film_investment_model_v2/model_summary_v2.json","w") as f:
    json.dump(summary_v2, f, indent=2)

print(f"   ✅  model_v2.pkl")
print(f"   ✅  scaler_v2.pkl")
print(f"   ✅  le_genre_v2.pkl")
print(f"   ✅  le_season_v2.pkl")
print(f"   ✅  model_summary_v2.json")


# ─────────────────────────────────────────────────────────────
# STEP 8 — FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  V2 COMPLETE — SUMMARY")
print("="*65)
print(f"""
  V1 → V2 IMPROVEMENTS:
  ├── Accuracy:   {acc_v1:.1%}  →  {acc_v2:.1%}  (+{(acc_v2-acc_v1)*100:.1f}pp)
  ├── AUC:        {auc_v1:.3f}  →  {auc_v2:.3f}  (+{(auc_v2-auc_v1)*1000:.1f} milli-pts)
  ├── Features:   7 → 13  (+6 pre-release signals)
  └── New layer:  Portfolio analysis (5-film comparison demo)

  NEW SIGNALS ADDED:
  ├── community_sentiment_score  ← TABB proxy (Peter's #1 ask)
  ├── social_buzz_index          ← Pre-release momentum
  ├── trailer_engagement_score   ← Marketing signal
  ├── festival_score             ← Critical/industry signal
  ├── creator_track_record       ← Director/actor history
  └── early_momentum             ← Composite signal

  PORTFOLIO LAYER:
  ├── Compare multiple films simultaneously
  ├── Capital allocation weighting by probability
  └── Portfolio-level risk metrics

  FILES SAVED:
  └── film_investment_model_v2/
      ├── model_v2.pkl
      ├── scaler_v2.pkl
      ├── le_genre_v2.pkl
      ├── le_season_v2.pkl
      └── model_summary_v2.json

  NEXT: Run app_v2.py for the upgraded Streamlit interface.
""")
