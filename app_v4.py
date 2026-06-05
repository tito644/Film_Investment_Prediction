import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Film Investment Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# SESSION STATE — Language & Theme
# ─────────────────────────────────────────────────────────────
if "lang"  not in st.session_state: st.session_state.lang  = "EN"
if "theme" not in st.session_state: st.session_state.theme = "dark"

lang  = st.session_state.lang
theme = st.session_state.theme
is_ar = lang == "AR"
is_dark = theme == "dark"

# ─────────────────────────────────────────────────────────────
# TRANSLATIONS
# ─────────────────────────────────────────────────────────────
T = {
    "EN": {
        "title":          "Film Investment Intelligence",
        "subtitle":       "Enter your film's data — get an instant investment decision + full brief",
        "badge":          "🎬 Powered by Lunim · Agent v4",
        "toggle_lang":    "🌐 العربية",
        "toggle_theme":   "☀️ Light",
        "sec_film":       "📋 Film Information",
        "film_name":      "Film Name",
        "film_name_ph":   "e.g. The Last Horizon",
        "genre":          "Genre",
        "season":         "Release Season",
        "budget":         "Budget ($ Million)",
        "runtime":        "Runtime (minutes)",
        "sec_market":     "📊 Market Signals",
        "market_note":    "Based on comparable films and initial market research",
        "popularity":     "Expected Popularity",
        "pop_help":       "Blockbuster: 100+ | Mid: 40–80 | Indie: 10–40",
        "rating":         "Expected Rating",
        "rating_help":    "Poor: <5 | Average: 5–7 | Good: 7+ | Excellent: 8.5+",
        "voters":         "Expected Voters",
        "voters_help":    "Niche: <500 | Mid: 1K–5K | Wide: 5K–15K | Blockbuster: 15K+",
        "sec_pre":        "⚡ Pre-Release Signals",
        "pre_note":       "These signals exist BEFORE the film releases — the key insight that drives the model",
        "trailer":        "Trailer Engagement",
        "trailer_help":   "Weak: <30 | Average: 30–60 | Strong: 60–80 | Viral: 80+",
        "social":         "Social Media Buzz",
        "social_help":    "Low: <30 | Moderate: 30–60 | High: 60–80 | Viral: 80+",
        "sentiment":      "Community Sentiment",
        "sentiment_help": "Negative: <30 | Mixed: 30–60 | Positive: 60–80 | Very Positive: 80+",
        "festival":       "Festival Score",
        "festival_help":  "None: 0 | Minor: 1–3 | Major: 4–7 | Award winner: 8+",
        "creator":        "Creator Track Record",
        "creator_help":   "Unknown: 0–3 | Some credits: 3–5 | Established: 5–7 | A-list: 8+",
        "analyze_btn":    "🚀 Analyze Investment",
        "spinner":        "🎬 Analyzing investment...",
        "results":        "📊 Results",
        "prob_label":     "Probability Score",
        "conf_label":     "Confidence Score",
        "momentum_label": "Early Momentum",
        "roi_label":      "Expected ROI",
        "risk_label":     "Risk Level",
        "conf_tag":       "Confidence",
        "human_req":      "🚨 HUMAN REVIEW REQUIRED",
        "auto_ok":        "✅ AUTONOMOUS DECISION — Agent confidence sufficient",
        "strengths":      "✓ Strengths",
        "risks":          "⚠ Risk Factors",
        "brief_title":    "📄 Full Investment Brief",
        "download":       "⬇️ Download Brief (.txt)",
        "no_name":        "⚠️ Please enter a film name",
        "welcome":        "Fill in the form above and click",
        "stats_title":    "📈 Model Stats",
        "films_analyzed": "Films Analyzed",
        "accuracy":       "Model Accuracy",
        "auc":            "AUC Score",
        "features":       "Features",
        "season_help":    "Summer=Jun–Aug | Holiday=Nov–Dec | Spring=Mar–May | Off-Season=rest",
        "brief_header":   "FILM INVESTMENT INTELLIGENCE BRIEF",
        "brief_powered":  "Powered by Lunim · Agent v4",
        "brief_date":     "Date",
        "brief_genre":    "Genre",
        "brief_season":   "Season",
        "brief_budget":   "Budget",
        "brief_runtime":  "Runtime",
        "brief_exec":     "EXECUTIVE SUMMARY",
        "brief_decision": "INVESTMENT DECISION",
        "brief_prob":     "Probability Score",
        "brief_risk":     "Risk Level",
        "brief_momentum": "Early Momentum",
        "brief_conf":     "Confidence",
        "brief_str":      "STRENGTHS",
        "brief_riskf":    "RISK FACTORS",
        "brief_weak":     "Weak Signals",
        "brief_contra":   "Signal Contradictions",
        "brief_escal":    "HUMAN ESCALATION",
        "brief_human":    "🚨 HUMAN REVIEW REQUIRED",
        "brief_auto":     "✅ AUTONOMOUS DECISION — No human review required",
        "brief_rec":      "RECOMMENDATION",
        "brief_gen":      "Generated",
        "brief_super":    "Supervisor",
    },
    "AR": {
        "title":          "ذكاء الاستثمار السينمائي",
        "subtitle":       "أدخل بيانات الفيلم — واحصل على قرار استثماري فوري + تقرير كامل",
        "badge":          "🎬 مدعوم من Lunim · الإصدار الرابع",
        "toggle_lang":    "🌐 English",
        "toggle_theme":   "☀️ فاتح",
        "sec_film":       "📋 معلومات الفيلم",
        "film_name":      "اسم الفيلم",
        "film_name_ph":   "مثال: الأفق الأخير",
        "genre":          "النوع",
        "season":         "موسم الإصدار",
        "budget":         "الميزانية (مليون دولار)",
        "runtime":        "المدة (دقيقة)",
        "sec_market":     "📊 إشارات السوق",
        "market_note":    "بناءً على أفلام مماثلة وبحث السوق الأولي",
        "popularity":     "الشعبية المتوقعة",
        "pop_help":       "ضخم: 100+ | متوسط: 40–80 | مستقل: 10–40",
        "rating":         "التقييم المتوقع",
        "rating_help":    "ضعيف: <5 | متوسط: 5–7 | جيد: 7+ | ممتاز: 8.5+",
        "voters":         "عدد المصوّتين المتوقع",
        "voters_help":    "محدود: <500 | متوسط: 1K–5K | واسع: 5K–15K | ضخم: 15K+",
        "sec_pre":        "⚡ إشارات ما قبل الإصدار",
        "pre_note":       "هذه الإشارات موجودة قبل إصدار الفيلم — الرؤية الأساسية التي تقود النموذج",
        "trailer":        "تفاعل المقطع الترويجي",
        "trailer_help":   "ضعيف: <30 | متوسط: 30–60 | قوي: 60–80 | فيروسي: 80+",
        "social":         "الضجة على السوشيال ميديا",
        "social_help":    "منخفض: <30 | معتدل: 30–60 | عالٍ: 60–80 | فيروسي: 80+",
        "sentiment":      "رأي المجتمع",
        "sentiment_help": "سلبي: <30 | مختلط: 30–60 | إيجابي: 60–80 | إيجابي جداً: 80+",
        "festival":       "تقييم المهرجانات",
        "festival_help":  "لا شيء: 0 | صغير: 1–3 | كبير: 4–7 | فائز بجائزة: 8+",
        "creator":        "سجل صانع الفيلم",
        "creator_help":   "مجهول: 0–3 | بعض الأعمال: 3–5 | راسخ: 5–7 | نجم: 8+",
        "analyze_btn":    "🚀 تحليل الاستثمار",
        "spinner":        "🎬 جاري التحليل...",
        "results":        "📊 النتائج",
        "prob_label":     "درجة الاحتمالية",
        "conf_label":     "درجة الثقة",
        "momentum_label": "الزخم المبكر",
        "roi_label":      "العائد المتوقع",
        "risk_label":     "مستوى المخاطرة",
        "conf_tag":       "الثقة",
        "human_req":      "🚨 مطلوب مراجعة بشرية",
        "auto_ok":        "✅ قرار آلي — ثقة الوكيل كافية",
        "strengths":      "✓ نقاط القوة",
        "risks":          "⚠ عوامل الخطر",
        "brief_title":    "📄 التقرير الاستثماري الكامل",
        "download":       "⬇️ تحميل التقرير (.txt)",
        "no_name":        "⚠️ من فضلك أدخل اسم الفيلم",
        "welcome":        "أدخل بيانات الفيلم واضغط",
        "stats_title":    "📈 إحصائيات النموذج",
        "films_analyzed": "فيلم تم تحليله",
        "accuracy":       "دقة النموذج",
        "auc":            "معدل AUC",
        "features":       "إشارة",
        "season_help":    "صيف=يونيو–أغسطس | عطلة=نوف–ديس | ربيع=مارس–مايو | خارج الموسم=باقي الشهور",
        "brief_header":   "تقرير ذكاء الاستثمار السينمائي",
        "brief_powered":  "مدعوم من Lunim · الإصدار الرابع",
        "brief_date":     "التاريخ",
        "brief_genre":    "النوع",
        "brief_season":   "الموسم",
        "brief_budget":   "الميزانية",
        "brief_runtime":  "المدة",
        "brief_exec":     "الملخص التنفيذي",
        "brief_decision": "قرار الاستثمار",
        "brief_prob":     "درجة الاحتمالية",
        "brief_risk":     "مستوى المخاطرة",
        "brief_momentum": "الزخم المبكر",
        "brief_conf":     "الثقة",
        "brief_str":      "نقاط القوة",
        "brief_riskf":    "عوامل الخطر",
        "brief_weak":     "إشارات ضعيفة",
        "brief_contra":   "تناقضات الإشارات",
        "brief_escal":    "التصعيد البشري",
        "brief_human":    "🚨 مطلوب مراجعة بشرية",
        "brief_auto":     "✅ قرار آلي — لا حاجة لمراجعة بشرية",
        "brief_rec":      "التوصية",
        "brief_gen":      "تاريخ الإنشاء",
        "brief_super":    "المشرف",
    }
}[lang]

# ─────────────────────────────────────────────────────────────
# THEME COLORS
# ─────────────────────────────────────────────────────────────
if is_dark:
    BG      = "#0D0D1A"
    CARD    = "#131D31"
    NAVY    = "#0F1F3D"
    TEXT    = "#FFFFFF"
    SUBTEXT = "#CBD5E0"
    MGRAY   = "#718096"
    BORDER  = "rgba(0,180,216,0.2)"
    INPUT_BG= "#131D31"
    HR      = "rgba(255,255,255,0.06)"
else:
    BG      = "#F7F9FC"
    CARD    = "#FFFFFF"
    NAVY    = "#E8F4FD"
    TEXT    = "#1A202C"
    SUBTEXT = "#4A5568"
    MGRAY   = "#718096"
    BORDER  = "rgba(0,150,180,0.25)"
    INPUT_BG= "#FFFFFF"
    HR      = "rgba(0,0,0,0.08)"

TEAL  = "#00B4D8"
GOLD  = "#FFD166"
GREEN = "#06D6A0"
RED   = "#E63946"
RTL   = 'direction:rtl; text-align:right;' if is_ar else ''

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&family=Cairo:wght@400;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: {'Cairo' if is_ar else 'DM Sans'}, sans-serif;
    background-color: {BG};
    color: {TEXT};
    {RTL}
}}
h1,h2,h3 {{ font-family: {'Cairo' if is_ar else 'Syne'}, sans-serif; }}
.main {{ background-color: {BG}; }}
.block-container {{ padding: 1.5rem 2.5rem !important; max-width: 1200px !important; }}

.hero {{
    background: {'linear-gradient(135deg, #0F1F3D 0%, #0D0D1A 60%, #001820 100%)' if is_dark else 'linear-gradient(135deg, #E8F4FD 0%, #F0F8FF 100%)'};
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    border: 1px solid {BORDER};
    position: relative;
    overflow: hidden;
    {RTL}
}}
.hero::before {{
    content: '';
    position: absolute;
    top: -40%;
    right: -5%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(0,180,216,0.07) 0%, transparent 70%);
    pointer-events: none;
}}
.hero-title {{
    font-family: {'Cairo' if is_ar else 'Syne'}, sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: {TEXT};
    margin: 0.3rem 0;
    line-height: 1.15;
}}
.hero-title span {{ color: {TEAL}; }}
.hero-sub {{ color: {SUBTEXT}; font-size: 1rem; margin-top: 0.4rem; }}
.badge {{
    display: inline-block;
    background: rgba(0,180,216,0.12);
    border: 1px solid {TEAL};
    color: {TEAL};
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}}
.topbar {{
    display: flex;
    justify-content: flex-end;
    gap: 0.6rem;
    margin-bottom: 1rem;
    {RTL}
}}
.section-header {{
    font-family: {'Cairo' if is_ar else 'Syne'}, sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: {TEAL};
    letter-spacing: {'0px' if is_ar else '2px'};
    text-transform: {'none' if is_ar else 'uppercase'};
    margin: 1.8rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {BORDER};
}}
.helper {{ color: {MGRAY}; font-size: 0.78rem; margin-top: -0.4rem; margin-bottom: 0.5rem; {RTL} }}
.metric-card {{
    background: {CARD};
    border-radius: 12px;
    padding: 1.3rem;
    border: 1px solid {'rgba(255,255,255,0.06)' if is_dark else 'rgba(0,0,0,0.08)'};
    text-align: center;
    box-shadow: {'0 4px 20px rgba(0,0,0,0.3)' if is_dark else '0 2px 12px rgba(0,0,0,0.08)'};
}}
.metric-value {{
    font-family: {'Cairo' if is_ar else 'Syne'}, sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
}}
.metric-label {{ color: {MGRAY}; font-size: 0.8rem; margin-top: 0.35rem; }}

.decision-invest {{
    background: {'rgba(6,214,160,0.12)' if is_dark else 'rgba(6,214,160,0.08)'};
    border: 2px solid {GREEN};
    border-radius: 12px; padding: 1.3rem 1.8rem;
    font-family: {'Cairo' if is_ar else 'Syne'}, sans-serif;
    font-size: 1.3rem; font-weight: 700; color: {GREEN}; {RTL}
}}
.decision-caution {{
    background: {'rgba(255,209,102,0.12)' if is_dark else 'rgba(255,209,102,0.08)'};
    border: 2px solid {GOLD};
    border-radius: 12px; padding: 1.3rem 1.8rem;
    font-family: {'Cairo' if is_ar else 'Syne'}, sans-serif;
    font-size: 1.3rem; font-weight: 700; color: {GOLD}; {RTL}
}}
.decision-review {{
    background: {'rgba(246,166,35,0.1)' if is_dark else 'rgba(246,166,35,0.07)'};
    border: 2px solid #f6a623;
    border-radius: 12px; padding: 1.3rem 1.8rem;
    font-family: {'Cairo' if is_ar else 'Syne'}, sans-serif;
    font-size: 1.3rem; font-weight: 700; color: #f6a623; {RTL}
}}
.decision-dont {{
    background: {'rgba(230,57,70,0.12)' if is_dark else 'rgba(230,57,70,0.08)'};
    border: 2px solid {RED};
    border-radius: 12px; padding: 1.3rem 1.8rem;
    font-family: {'Cairo' if is_ar else 'Syne'}, sans-serif;
    font-size: 1.3rem; font-weight: 700; color: {RED}; {RTL}
}}

.escalation-box {{
    background: {'rgba(230,57,70,0.1)' if is_dark else 'rgba(230,57,70,0.06)'};
    border: 2px solid {RED};
    border-radius: 12px; padding: 1rem 1.5rem;
    color: {RED}; font-weight: 500; {RTL}
}}
.autonomous-box {{
    background: {'rgba(6,214,160,0.07)' if is_dark else 'rgba(6,214,160,0.05)'};
    border: 1px solid {GREEN};
    border-radius: 12px; padding: 1rem 1.5rem; color: {GREEN}; {RTL}
}}
.brief-box {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px; padding: 1.8rem;
    font-family: 'Courier New', monospace;
    font-size: 0.8rem; line-height: 1.7;
    color: {SUBTEXT}; white-space: pre-wrap; overflow-x: auto;
}}
hr {{ border-color: {HR} !important; }}

/* Inputs */
.stTextInput > div > div > input {{
    background: {INPUT_BG} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 8px !important;
}}
.stNumberInput > div > div > input {{
    background: {INPUT_BG} !important;
    color: {TEXT} !important;
    border-radius: 8px !important;
}}
.stSelectbox > div > div {{
    background: {INPUT_BG} !important;
    border-color: {BORDER} !important;
    border-radius: 8px !important;
}}
label {{ color: {TEXT} !important; }}

.stButton > button {{
    background: linear-gradient(135deg, {TEAL}, #0077A8) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.7rem 2rem !important;
    font-family: {'Cairo' if is_ar else 'Syne'}, sans-serif !important;
    font-size: 1rem !important; font-weight: 700 !important;
    width: 100% !important; letter-spacing: {'0' if is_ar else '1px'} !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,180,216,0.3) !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    BASE     = "film_investment_model_v2"
    model    = joblib.load(f"{BASE}/model_v2.pkl")
    scaler   = joblib.load(f"{BASE}/scaler_v2.pkl")
    le_genre = joblib.load(f"{BASE}/le_genre_v2.pkl")
    le_season= joblib.load(f"{BASE}/le_season_v2.pkl")
    return model, scaler, le_genre, le_season

model, scaler, le_genre, le_season = load_model()
VALID_GENRES  = list(le_genre.classes_)
VALID_SEASONS = list(le_season.classes_)


# ─────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────
def score_film(d):
    ge = le_genre.transform([d["genre"]])[0]
    se = le_season.transform([d["season"]])[0]
    em = (d["sentiment_score"]*0.40 + d["social_buzz"]*0.30 +
          d["trailer_score"]*0.20 + (d["festival_score"]/10*100)*0.10)
    feat = np.array([[d["budget_m"], d["popularity"], d["vote_avg"],
                      d["vote_count"], d["runtime"], ge, se,
                      d["trailer_score"], d["social_buzz"], d["festival_score"],
                      d["sentiment_score"], d["creator_score"], em]])
    prob = model.predict_proba(scaler.transform(feat))[0][1]
    return round(prob*100,1), round(em,1)


def get_decision(prob, lang):
    if lang == "AR":
        if prob >= 70:   return "✅ استثمر بثقة",         "مخاطرة منخفضة",     "invest"
        elif prob >= 50: return "⚠️ استثمر بحذر",          "مخاطرة متوسطة",     "caution"
        elif prob >= 35: return "🔶 راجع التفاصيل",        "مخاطرة عالية",      "review"
        else:            return "❌ لا تستثمر",             "مخاطرة عالية جداً", "dont"
    else:
        if prob >= 70:   return "✅ INVEST WITH CONFIDENCE", "Low Risk",       "invest"
        elif prob >= 50: return "⚠️ INVEST WITH CAUTION",    "Medium Risk",    "caution"
        elif prob >= 35: return "🔶 REVIEW DETAILS",          "High Risk",      "review"
        else:            return "❌ DO NOT INVEST",            "Very High Risk", "dont"


def compute_confidence(prob, d):
    weak, contra, penalties = [], [], 0
    if 40 <= prob <= 65:
        penalties += 20; weak.append("Probability in grey zone" if lang=="EN" else "الاحتمالية في المنطقة الرمادية")
    if d["popularity"] < 30 and d["budget_m"] > 50:
        penalties += 15; contra.append(f"High budget (${d['budget_m']}M) but low popularity" if lang=="EN" else f"ميزانية عالية (${d['budget_m']}M) مع شعبية منخفضة")
    if d["social_buzz"] < 35 and d["sentiment_score"] > 70:
        penalties += 10; contra.append("High sentiment but low social buzz" if lang=="EN" else "رأي إيجابي لكن ضجة اجتماعية منخفضة")
    if d["festival_score"] >= 7 and d["popularity"] < 40:
        penalties += 10; weak.append("Strong festival but low market popularity" if lang=="EN" else "حضور مهرجانات قوي لكن شعبية سوقية منخفضة")
    if d["vote_count"] < 500:
        penalties += 15; weak.append(f"Very low expected voters ({int(d['vote_count'])})" if lang=="EN" else f"عدد مصوّتين منخفض جداً ({int(d['vote_count'])})")
    if d["creator_score"] < 4 and d["budget_m"] > 40:
        penalties += 10; contra.append("High budget but weak creator track record" if lang=="EN" else "ميزانية عالية مع سجل ضعيف لصانع الفيلم")
    pre_avg = np.mean([d["trailer_score"], d["social_buzz"], d["sentiment_score"], d["creator_score"]*10])
    if d["vote_count"] >= 3000 and pre_avg < 45:
        penalties += 35; contra.append(f"Model bias alert: pre-release signals weak (avg: {pre_avg:.0f}/100)" if lang=="EN" else f"تحذير: إشارات ما قبل الإصدار ضعيفة (متوسط: {pre_avg:.0f}/100)")
    if d["vote_avg"] < 5.0:
        penalties += 30; contra.append(f"Critical: Expected rating {d['vote_avg']}/10 is very low" if lang=="EN" else f"حرج: التقييم المتوقع {d['vote_avg']}/10 منخفض جداً")
    score = max(0, 100-penalties)
    label = "HIGH" if score >= 75 else "MEDIUM" if score >= 50 else "LOW"
    return score, label, weak, contra


def check_escalation(prob, conf_label, contradictions, d):
    reasons = []
    if conf_label == "LOW":
        reasons.append("Confidence is LOW — too many contradictory signals" if lang=="EN" else "الثقة منخفضة — إشارات متناقضة كثيرة")
    if 45 <= prob <= 60:
        reasons.append(f"Probability ({prob}%) in grey zone" if lang=="EN" else f"الاحتمالية ({prob}%) في المنطقة الرمادية")
    if len(contradictions) >= 2:
        reasons.append(f"{len(contradictions)} signal contradictions detected" if lang=="EN" else f"تم اكتشاف {len(contradictions)} تناقضات في الإشارات")
    if d["budget_m"] >= 100 and conf_label != "HIGH":
        reasons.append(f"High budget (${d['budget_m']}M) with non-HIGH confidence" if lang=="EN" else f"ميزانية عالية (${d['budget_m']}M) مع ثقة غير عالية")
    return len(reasons) > 0, reasons


def generate_brief(film_name, prob, decision, risk, em, d, cs, cl, weak, contra, esc, esc_r):
    now = datetime.now().strftime("%B %d, %Y")
    D   = "─" * 56
    t   = T

    if prob >= 70:   smry = f"{film_name} presents a strong investment opportunity ({prob}%). Confidence: {cl}."
    elif prob >= 50: smry = f"{film_name} shows moderate potential ({prob}%). Review recommended. Confidence: {cl}."
    elif prob >= 35: smry = f"{film_name} carries significant risk ({prob}%). Weak signals detected. Confidence: {cl}."
    else:            smry = f"{film_name} does not meet investment criteria ({prob}%). Unfavorable risk profile."

    strengths, risks = [], []
    if d["vote_avg"] >= 7.5:       strengths.append(f"Strong rating: {d['vote_avg']}/10")
    if d["trailer_score"] >= 70:   strengths.append(f"High trailer: {d['trailer_score']}/100")
    if d["festival_score"] >= 6:   strengths.append(f"Festival: {d['festival_score']}/10")
    if d["creator_score"] >= 7:    strengths.append(f"Creator: {d['creator_score']}/10")
    if d["sentiment_score"] >= 70: strengths.append(f"Sentiment: {d['sentiment_score']}/100")
    if d["season"] in ["Summer","Holiday"]: strengths.append(f"Season: {d['season']}")
    if not strengths:              strengths.append("Limited strengths in current signals")
    if d["budget_m"] > 100:    risks.append(f"High capital: ${d['budget_m']}M")
    if d["popularity"] < 30:   risks.append(f"Low popularity: {d['popularity']}")
    if d["creator_score"] < 5: risks.append(f"Weak creator: {d['creator_score']}/10")
    if d["season"] == "Off-Season": risks.append("Off-Season release")
    if d["social_buzz"] < 30:  risks.append(f"Low social buzz: {d['social_buzz']}/100")
    if not risks:              risks.append("No major risk flags")

    b  = f"{'═'*58}\n  {t['brief_header']}\n  {t['brief_powered']}\n{'═'*58}\n\n"
    b += f"  {t['brief_date']}    : {now}\n"
    b += f"  Film       : {film_name}\n"
    b += f"  {t['brief_genre']}    : {d['genre']}  |  {t['brief_season']}: {d['season']}\n"
    b += f"  {t['brief_budget']}  : ${d['budget_m']}M  |  {t['brief_runtime']}: {d['runtime']} min\n\n"
    b += f"  {D}\n  {t['brief_exec']}\n  {D}\n  {smry}\n\n"
    b += f"  {D}\n  {t['brief_decision']}\n  {D}\n"
    b += f"  {t['brief_prob']}  : {prob}%\n  Decision   : {decision}\n"
    b += f"  {t['brief_risk']}  : {risk}\n  {t['brief_momentum']}   : {em}/100\n"
    b += f"  {t['brief_conf']}       : {cl} ({cs}/100)\n\n"
    b += f"  {D}\n  {t['brief_str']}\n  {D}\n"
    for s in strengths: b += f"  ✓  {s}\n"
    b += f"\n  {D}\n  {t['brief_riskf']}\n  {D}\n"
    for r in risks: b += f"  ⚠  {r}\n"
    if weak:   b += f"\n  {t['brief_weak']}:\n" + "".join(f"  →  {w}\n" for w in weak)
    if contra: b += f"\n  {t['brief_contra']}:\n" + "".join(f"  ↯  {c}\n" for c in contra)
    b += f"\n  {D}\n  {t['brief_escal']}\n  {D}\n"
    b += (f"  {t['brief_human']}\n" + "".join(f"  •  {r}\n" for r in esc_r)) if esc else f"  {t['brief_auto']}\n"
    b += f"\n  {D}\n  {t['brief_rec']}\n  {D}\n  {decision}\n\n"
    b += f"  {t['brief_gen']}: {now}\n  {t['brief_super']}: Tarek ElNaggar · Lunim\n{'═'*58}"
    return b


# ─────────────────────────────────────────────────────────────
# TOP BAR — Language & Theme toggles
# ─────────────────────────────────────────────────────────────
tb_col1, tb_col2, tb_col3 = st.columns([6, 1, 1])
with tb_col2:
    if st.button(T["toggle_lang"], key="lang_btn"):
        st.session_state.lang = "AR" if lang == "EN" else "EN"
        st.rerun()
with tb_col3:
    theme_label = ("☀️ Light" if is_dark else "🌙 Dark") if lang == "EN" else ("☀️ فاتح" if is_dark else "🌙 داكن")
    if st.button(theme_label, key="theme_btn"):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()


# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
title_parts = T["title"].split(" ", 2)
st.markdown(f"""
<div class="hero">
    <div class="badge">{T['badge']}</div>
    <h1 class="hero-title">{T['title'].replace(title_parts[-1], f'<span>{title_parts[-1]}</span>')}</h1>
    <p class="hero-sub">{T['subtitle']}</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FORM
# ─────────────────────────────────────────────────────────────
with st.form("investment_form"):

    st.markdown(f'<div class="section-header">{T["sec_film"]}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        film_name = st.text_input(T["film_name"], placeholder=T["film_name_ph"])
    with col2:
        genre = st.selectbox(T["genre"], sorted(VALID_GENRES))

    col3, col4, col5 = st.columns(3)
    with col3:
        season = st.selectbox(T["season"], VALID_SEASONS, help=T["season_help"])
    with col4:
        budget_m = st.number_input(T["budget"], min_value=0.5, max_value=500.0, value=30.0, step=0.5)
    with col5:
        runtime = st.number_input(T["runtime"], min_value=60, max_value=240, value=110)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">{T["sec_market"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="helper">{T["market_note"]}</p>', unsafe_allow_html=True)

    col6, col7, col8 = st.columns(3)
    with col6:
        popularity = st.slider(T["popularity"], 1, 200, 60)
        st.markdown(f'<p class="helper">{T["pop_help"]}</p>', unsafe_allow_html=True)
    with col7:
        vote_avg = st.slider(T["rating"], 1.0, 10.0, 7.0, 0.1)
        st.markdown(f'<p class="helper">{T["rating_help"]}</p>', unsafe_allow_html=True)
    with col8:
        vote_count = st.number_input(T["voters"], min_value=100, max_value=50000, value=3000, step=100)
        st.markdown(f'<p class="helper">{T["voters_help"]}</p>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">{T["sec_pre"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="helper">{T["pre_note"]}</p>', unsafe_allow_html=True)

    col9, col10, col11 = st.columns(3)
    with col9:
        trailer_score = st.slider(T["trailer"], 0, 100, 60)
        st.markdown(f'<p class="helper">{T["trailer_help"]}</p>', unsafe_allow_html=True)
    with col10:
        social_buzz = st.slider(T["social"], 0, 100, 55)
        st.markdown(f'<p class="helper">{T["social_help"]}</p>', unsafe_allow_html=True)
    with col11:
        sentiment_score = st.slider(T["sentiment"], 0, 100, 65)
        st.markdown(f'<p class="helper">{T["sentiment_help"]}</p>', unsafe_allow_html=True)

    col12, col13 = st.columns(2)
    with col12:
        festival_score = st.slider(T["festival"], 0.0, 10.0, 2.0, 0.5)
        st.markdown(f'<p class="helper">{T["festival_help"]}</p>', unsafe_allow_html=True)
    with col13:
        creator_score = st.slider(T["creator"], 0.0, 10.0, 6.0, 0.5)
        st.markdown(f'<p class="helper">{T["creator_help"]}</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button(T["analyze_btn"])


# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
if submitted:
    if not film_name.strip():
        st.error(T["no_name"]); st.stop()

    d = dict(budget_m=float(budget_m), popularity=float(popularity),
             vote_avg=float(vote_avg), vote_count=float(vote_count),
             runtime=float(runtime), genre=genre, season=season,
             trailer_score=float(trailer_score), social_buzz=float(social_buzz),
             festival_score=float(festival_score), sentiment_score=float(sentiment_score),
             creator_score=float(creator_score))

    with st.spinner(T["spinner"]):
        prob, em                          = score_film(d)
        decision_label, risk, dtype       = get_decision(prob, lang)
        cs, cl, weak, contra              = compute_confidence(prob, d)
        esc, esc_r                        = check_escalation(prob, cl, contra, d)
        if cl == "LOW" and prob >= 70:
            decision_label = ("🔶 REVIEW — CONFIDENCE OVERRIDE" if lang=="EN" else "🔶 مراجعة — تجاوز الثقة")
            dtype = "review"; risk = ("High Risk (Override)" if lang=="EN" else "مخاطرة عالية (تجاوز)")
        brief = generate_brief(film_name, prob, decision_label, risk, em, d, cs, cl, weak, contra, esc, esc_r)

    st.markdown("---")
    st.markdown(f'<div class="section-header">{T["results"]} — {film_name}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    prob_color = GREEN if prob >= 70 else GOLD if prob >= 50 else RED
    conf_color = GREEN if cl == "HIGH" else GOLD if cl == "MEDIUM" else RED

    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{prob_color}">{prob}%</div><div class="metric-label">{T["prob_label"]}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{conf_color}">{cs}</div><div class="metric-label">{T["conf_label"]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{TEAL}">{em}</div><div class="metric-label">{T["momentum_label"]}</div></div>', unsafe_allow_html=True)
    with c4:
        roi = round(prob/100*4, 1)
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{GOLD}">{roi}x</div><div class="metric-label">{T["roi_label"]}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="decision-{dtype}">{decision_label}</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MGRAY}; margin-top:0.5rem;'>{T['risk_label']}: {risk} &nbsp;|&nbsp; {T['conf_tag']}: <strong style='color:{conf_color}'>{cl}</strong></p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if esc:
        st.markdown(f'<div class="escalation-box"><strong>{T["human_req"]}</strong><br>{"<br>".join("• "+r for r in esc_r)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="autonomous-box">{T["auto_ok"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cs1, cs2 = st.columns(2)
    with cs1:
        st.markdown(f'<div class="section-header">{T["strengths"]}</div>', unsafe_allow_html=True)
        strengths = []
        if d["vote_avg"] >= 7.5:       strengths.append(f"Rating: {d['vote_avg']}/10")
        if d["trailer_score"] >= 70:   strengths.append(f"Trailer: {d['trailer_score']}/100")
        if d["festival_score"] >= 6:   strengths.append(f"Festival: {d['festival_score']}/10")
        if d["creator_score"] >= 7:    strengths.append(f"Creator: {d['creator_score']}/10")
        if d["sentiment_score"] >= 70: strengths.append(f"Sentiment: {d['sentiment_score']}/100")
        if d["season"] in ["Summer","Holiday"]: strengths.append(f"Season: {d['season']}")
        if not strengths:              strengths.append("Limited strengths")
        for s in strengths: st.markdown(f"✓ {s}")
    with cs2:
        st.markdown(f'<div class="section-header">{T["risks"]}</div>', unsafe_allow_html=True)
        risks = []
        if d["budget_m"] > 100:    risks.append(f"High budget: ${d['budget_m']}M")
        if d["popularity"] < 30:   risks.append(f"Low popularity: {d['popularity']}")
        if d["creator_score"] < 5: risks.append(f"Weak creator: {d['creator_score']}/10")
        if d["season"] == "Off-Season": risks.append("Off-Season release")
        if d["social_buzz"] < 30:  risks.append(f"Low social buzz")
        for c in contra[:3]:       risks.append(c)
        if not risks:              risks.append("No major risk flags")
        for r in risks[:5]: st.markdown(f"⚠ {r}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">{T["brief_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="brief-box">{brief}</div>', unsafe_allow_html=True)
    st.download_button(T["download"], data=brief,
                       file_name=f"brief_{film_name.replace(' ','_')}.txt", mime="text/plain")
    st.markdown(f"<p style='color:{MGRAY}; font-size:0.75rem; text-align:center; margin-top:2rem;'>Film Investment Intelligence Agent v4 · Lunim · {datetime.now().strftime('%B %d, %Y')}</p>", unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style='text-align:center; padding:3rem 0; color:{MGRAY};'>
        <div style='font-size:3rem; margin-bottom:1rem;'>🎬</div>
        <p style='font-size:1.1rem; color:{SUBTEXT};'>{T['welcome']} <strong style='color:{TEAL}'>{T['analyze_btn']}</strong></p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="section-header">{T["stats_title"]}</div>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{TEAL}">3,229</div><div class="metric-label">{T["films_analyzed"]}</div></div>', unsafe_allow_html=True)
    with sc2: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{GREEN}">90.7%</div><div class="metric-label">{T["accuracy"]}</div></div>', unsafe_allow_html=True)
    with sc3: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{GOLD}">0.971</div><div class="metric-label">{T["auc"]}</div></div>', unsafe_allow_html=True)
    with sc4: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{TEAL}">13</div><div class="metric-label">{T["features"]}</div></div>', unsafe_allow_html=True)
