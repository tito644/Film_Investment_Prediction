import streamlit as st
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

        # Pre-Production Tab
        "tab_market":     "🎬 Market Viability",
        "tab_preprod":    "🏗️ Pre-Production Readiness",
        "preprod_subtitle": "Is this project ready to be made? — Before a single frame is shot",
        "preprod_sec_project":  "📋 Project Information",
        "project_name":   "Project Name",
        "project_name_ph":"e.g. Newark Street Documentary",
        "preprod_sec_script":   "📝 Script (35%)",
        "preprod_sec_team":     "👥 Team (30%)",
        "preprod_sec_financial":"💰 Financial & Logistics (35%)",
        "script_completion":    "Script Completion",
        "script_completion_help": "0% = not started | 100% = final draft ready",
        "script_quality":       "Script Quality Score",
        "script_quality_help":  "Coverage/reader score - like Slated's Script Score (75+ = strong)",
        "genre_feasibility":    "Genre Feasibility",
        "genre_feasibility_help": "Does this genre fit the production scale? 0 = mismatch | 10 = perfect fit",
        "director_attached":   "Director Attached",
        "director_attached_help": "Track record strength of attached director, 0-10",
        "cast_attached":        "Key Cast Attached",
        "cast_attached_help":   "Star power / confirmation of lead cast, 0-10",
        "key_crew_readiness":   "Key Crew Readiness",
        "key_crew_readiness_help": "Percent of key crew (DoP, Production Designer, etc.) confirmed",
        "financing_secured":    "Financing Secured",
        "financing_secured_help": "Percent of total budget confirmed/secured",
        "location_confirmed":   "Location Confirmed",
        "location_confirmed_help": "Is the location locked and permitted? 0 = not started | 10 = fully secured",
        "on_location_shoot":    "On-Location Shoot?",
        "on_location_yes":      "Yes - shooting on location",
        "on_location_no":       "No - studio / controlled environment",
        "distribution_interest":"Distribution Interest",
        "distribution_interest_help": "Sales agent / distributor interest level, 0-10",
        "production_timeline":  "Production Timeline",
        "production_timeline_help": "How realistic/confirmed is the schedule? 0-10",
        "analyze_preprod_btn":  "🏗️ Analyze Readiness",
        "preprod_results":      "📊 Pre-Production Readiness Results",
        "overall_score":        "Overall Readiness Score",
        "tier_label":           "Readiness Tier",
        "category_breakdown":   "Category Breakdown (Slated-style)",
        "script_score_label":   "Script Score",
        "team_score_label":     "Team Score",
        "financial_score_label":"Financial / Logistics Score",
        "strongest_area":       "Strongest Area",
        "weakest_area":         "Weakest Area",
        "flags_title":          "⚠ Flags for Review",
        "abelcine_flag_title":  "🎥 AbelCine Recommendation",
        "no_flags":             "No major gaps flagged",
        "preprod_brief_title":  "📄 Full Pre-Production Brief",
        "preprod_download":     "⬇️ Download Pre-Production Brief (.txt)",
        "no_project_name":      "⚠️ Please enter a project name",
        "preprod_welcome":      "Fill in the project details above and click",
        "what_this_means":      "What This Means",
        "what_this_means_text": "This score answers: \"Is this project ready to be made?\" It complements the Market Viability score (other tab), which answers: \"If made, will it succeed in the market?\" Together: full-lifecycle Decision Intelligence.",
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

        # Pre-Production Tab
        "tab_market":     "🎬 الجدوى السوقية",
        "tab_preprod":    "🏗️ جاهزية ما قبل الإنتاج",
        "preprod_subtitle": "هل هذا المشروع جاهز للتنفيذ؟ — قبل تصوير أي مشهد",
        "preprod_sec_project":  "📋 معلومات المشروع",
        "project_name":   "اسم المشروع",
        "project_name_ph":"مثال: وثائقي شارع نيوآرك",
        "preprod_sec_script":   "📝 السيناريو (35%)",
        "preprod_sec_team":     "👥 الفريق (30%)",
        "preprod_sec_financial":"💰 التمويل واللوجستيات (35%)",
        "script_completion":    "نسبة اكتمال السيناريو",
        "script_completion_help": "0% = لم يبدأ | 100% = النسخة النهائية جاهزة",
        "script_quality":       "تقييم جودة السيناريو",
        "script_quality_help":  "تقييم القراء — مثل Script Score في Slated (75+ = قوي)",
        "genre_feasibility":    "مناسبة النوع للإنتاج",
        "genre_feasibility_help": "هل النوع يناسب حجم الإنتاج؟ 0 = غير مناسب | 10 = مناسب تماماً",
        "director_attached":   "المخرج المؤكد",
        "director_attached_help": "قوة سجل المخرج المرتبط بالمشروع، 0-10",
        "cast_attached":        "الكاست الرئيسي المؤكد",
        "cast_attached_help":   "قوة نجومية وتأكيد الكاست الرئيسي، 0-10",
        "key_crew_readiness":   "جاهزية الفريق الأساسي",
        "key_crew_readiness_help": "نسبة تأكيد الفريق الأساسي (مدير تصوير، مصمم إنتاج، إلخ)",
        "financing_secured":    "التمويل المؤكد",
        "financing_secured_help": "نسبة الميزانية الإجمالية المؤكدة/المضمونة",
        "location_confirmed":   "تأكيد الموقع",
        "location_confirmed_help": "هل الموقع محجوز ومرخّص؟ 0 = لم يبدأ | 10 = مؤكد بالكامل",
        "on_location_shoot":    "تصوير في موقع خارجي؟",
        "on_location_yes":      "نعم - تصوير في موقع خارجي",
        "on_location_no":       "لا - استوديو / بيئة مغلقة",
        "distribution_interest":"اهتمام التوزيع",
        "distribution_interest_help": "مستوى اهتمام وكلاء المبيعات/التوزيع، 0-10",
        "production_timeline":  "الجدول الزمني للإنتاج",
        "production_timeline_help": "ما مدى واقعية/تأكيد الجدول الزمني؟ 0-10",
        "analyze_preprod_btn":  "🏗️ تحليل الجاهزية",
        "preprod_results":      "📊 نتائج جاهزية ما قبل الإنتاج",
        "overall_score":        "نتيجة الجاهزية الإجمالية",
        "tier_label":           "مستوى الجاهزية",
        "category_breakdown":   "تفاصيل الفئات (على نمط Slated)",
        "script_score_label":   "تقييم السيناريو",
        "team_score_label":     "تقييم الفريق",
        "financial_score_label":"تقييم التمويل / اللوجستيات",
        "strongest_area":       "أقوى فئة",
        "weakest_area":         "أضعف فئة",
        "flags_title":          "⚠ تحذيرات للمراجعة",
        "abelcine_flag_title":  "🎥 توصية AbelCine",
        "no_flags":             "لا توجد تحذيرات كبيرة",
        "preprod_brief_title":  "📄 التقرير الكامل لجاهزية ما قبل الإنتاج",
        "preprod_download":     "⬇️ تحميل تقرير ما قبل الإنتاج (.txt)",
        "no_project_name":      "⚠️ من فضلك أدخل اسم المشروع",
        "preprod_welcome":      "أدخل تفاصيل المشروع أعلاه واضغط",
        "what_this_means":      "ماذا يعني هذا",
        "what_this_means_text": "هذه النتيجة تجيب على: \"هل هذا المشروع جاهز للتنفيذ؟\" وهي تكمّل نتيجة الجدوى السوقية (التاب الآخر) التي تجيب على: \"إذا تم تنفيذه، هل سينجح في السوق؟\" معاً: ذكاء قرار شامل لكل مراحل المشروع.",
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
# Trained in-memory on app startup (cached) — avoids .pkl /
# scikit-learn version incompatibility across environments.
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    N = 3229
    genres  = ['Horror','Action','Drama','Comedy','Animation',
               'Science Fiction','Fantasy','Thriller','Adventure','Romance']
    seasons = ['Holiday','Summer','Spring','Off-Season']
    genre_weights  = [0.08,0.18,0.20,0.15,0.07,0.09,0.06,0.08,0.06,0.03]
    season_weights = [0.18,0.26,0.22,0.34]

    df = pd.DataFrame({
        'budget':       np.random.lognormal(mean=16.5, sigma=1.4, size=N).clip(50_000, 500_000_000),
        'popularity':   np.random.lognormal(mean=3.0,  sigma=1.2, size=N).clip(0.5, 300),
        'vote_average': np.random.normal(6.1, 1.1, N).clip(1.0, 10.0),
        'vote_count':   np.random.lognormal(mean=6.5, sigma=1.5, size=N).astype(int).clip(10, 15000),
        'runtime':      np.random.normal(107, 22, N).clip(60, 240),
        'primary_genre':np.random.choice(genres,  N, p=genre_weights),
        'season':       np.random.choice(seasons, N, p=season_weights),
    })

    genre_mult  = {'Horror':1.45,'Action':1.05,'Drama':0.90,'Comedy':0.88,
                   'Animation':1.10,'Science Fiction':1.12,'Fantasy':1.08,
                   'Thriller':1.02,'Adventure':1.06,'Romance':0.85}
    season_mult = {'Holiday':1.18,'Summer':1.10,'Spring':1.02,'Off-Season':0.95}

    base_roi = (df['vote_count']/3000)*2.5 + (df['popularity']/60)*1.2
    df['ROI'] = (base_roi * df['primary_genre'].map(genre_mult)
                 * df['season'].map(season_mult)
                 + np.random.normal(0, 0.5, N)).clip(0.1, 30)

    df['trailer_engagement_score'] = ((df['popularity']/300)*55 + np.random.normal(0,8,N)).clip(0,100)
    df['social_buzz_index']        = ((df['vote_count']/15000)*60 + (df['popularity']/300)*25 + np.random.normal(0,6,N)).clip(0,100)
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
        ), axis=1).round(1)
    df['community_sentiment_score'] = ((df['trailer_engagement_score']*0.40)
        + (df['social_buzz_index']*0.35)
        + (df['vote_average']/10*100*0.25)
        + np.random.normal(0,5,N)).clip(0,100)
    df['creator_track_record'] = np.random.choice(
        [0, np.random.uniform(1,4), np.random.uniform(4,7), np.random.uniform(7,10)],
        size=N, p=[0.25,0.30,0.28,0.17]).round(1)

    signal_boost = (df['community_sentiment_score']/100*0.8
                    + df['creator_track_record']/10*0.5
                    + df['festival_score']/10*0.4
                    + df['trailer_engagement_score']/100*0.3)
    df['ROI'] = (df['ROI'] + signal_boost*0.6).clip(0.1, 30)
    df['investment_success'] = (df['ROI'] >= 2.0).astype(int)

    df['budget_M'] = df['budget'] / 1_000_000
    le_genre_enc  = LabelEncoder()
    le_season_enc = LabelEncoder()
    df['genre_encoded']  = le_genre_enc.fit_transform(df['primary_genre'])
    df['season_encoded'] = le_season_enc.fit_transform(df['season'])
    df['early_momentum'] = (df['community_sentiment_score']*0.40
                            + df['social_buzz_index']*0.30
                            + df['trailer_engagement_score']*0.20
                            + df['festival_score']/10*100*0.10).round(2)

    FEATURES = ['budget_M','popularity','vote_average','vote_count','runtime',
                'genre_encoded','season_encoded','trailer_engagement_score',
                'social_buzz_index','festival_score','community_sentiment_score',
                'creator_track_record','early_momentum']

    X = df[FEATURES]
    y = df['investment_success']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_obj = StandardScaler()
    X_train_s  = scaler_obj.fit_transform(X_train)

    model_obj = GradientBoostingClassifier(n_estimators=150, random_state=42)
    model_obj.fit(X_train_s, y_train)

    return model_obj, scaler_obj, le_genre_enc, le_season_enc


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
# PRE-PRODUCTION SCORING (Slated-style: Script/Team/Financial)
# ─────────────────────────────────────────────────────────────
PREPROD_WEIGHTS = {"script": 0.35, "team": 0.30, "financial": 0.35}

def _pp_script(completion, quality, genre_fit):
    return round(completion*0.40 + quality*0.40 + (genre_fit/10*100)*0.20, 1)

def _pp_team(director, cast, crew_readiness):
    return round((director/10*100)*0.40 + (cast/10*100)*0.35 + crew_readiness*0.25, 1)

def _pp_financial(financing, location, distribution, timeline):
    return round(financing*0.40 + (location/10*100)*0.20 + (distribution/10*100)*0.20 + (timeline/10*100)*0.20, 1)

def compute_preproduction_score(p):
    script_score    = _pp_script(p["script_completion"], p["script_quality"], p["genre_feasibility"])
    team_score      = _pp_team(p["director_attached"], p["cast_attached"], p["key_crew_readiness"])
    financial_score = _pp_financial(p["financing_secured"], p["location_confirmed"],
                                     p["distribution_interest"], p["production_timeline"])
    overall = round(script_score*PREPROD_WEIGHTS["script"] +
                     team_score*PREPROD_WEIGHTS["team"] +
                     financial_score*PREPROD_WEIGHTS["financial"], 1)

    if overall >= 75:   tier_key = "ready_75"
    elif overall >= 55: tier_key = "developing_55"
    elif overall >= 35: tier_key = "early_35"
    else:               tier_key = "not_ready"

    categories = {"script": script_score, "team": team_score, "financial": financial_score}
    weakest   = max(categories, key=lambda k: -categories[k] if False else categories[k])
    weakest   = min(categories, key=categories.get)
    strongest = max(categories, key=categories.get)

    flags = []
    if script_score < 50:    flags.append("script_low")
    if team_score < 50:      flags.append("team_low")
    if financial_score < 50: flags.append("financial_low")

    location_need_flag = p["location_confirmed"] < 7 and p["on_location_shoot"]

    return {
        "script_score": script_score, "team_score": team_score,
        "financial_score": financial_score, "overall_score": overall,
        "tier_key": tier_key, "weakest": weakest, "strongest": strongest,
        "flags": flags, "location_need_flag": location_need_flag,
    }


def generate_preproduction_brief(project_name, p, r, lang):
    t   = T
    now = datetime.now().strftime("%B %d, %Y")
    D   = "─" * 56

    tier_label = t[r["tier_key"]]

    cat_names = {
        "EN": {"script": "Script", "team": "Team", "financial": "Financial/Logistics"},
        "AR": {"script": "السيناريو", "team": "الفريق", "financial": "التمويل/اللوجستيات"},
    }[lang]

    b  = f"{'═'*58}\n  {t['preprod_brief_title']}\n  Lunim · Decision Intelligence Layer · {now}\n{'═'*58}\n\n"
    b += f"  {t['project_name']}: {project_name}\n\n"
    b += f"  {D}\n  {t['overall_score']}\n  {D}\n"
    b += f"  {r['overall_score']}/100\n  {t['tier_label']}: {tier_label}\n\n"
    b += f"  {D}\n  {t['category_breakdown']}\n  {D}\n"
    b += f"  {t['script_score_label']}    : {r['script_score']}/100 (35%)\n"
    b += f"  {t['team_score_label']}      : {r['team_score']}/100 (30%)\n"
    b += f"  {t['financial_score_label']} : {r['financial_score']}/100 (35%)\n\n"
    b += f"  {D}\n  {t['strongest_area']} / {t['weakest_area']}\n  {D}\n"
    b += f"  {t['strongest_area']}: {cat_names[r['strongest']]}\n"
    b += f"  {t['weakest_area']}: {cat_names[r['weakest']]}\n\n"

    if r["abelcine_recommendation"]:
        b += f"  {D}\n  {t['abelcine_flag_title']}\n  {D}\n  {r['abelcine_recommendation']}\n\n"

    b += f"  {D}\n  {t['what_this_means']}\n  {D}\n  {t['what_this_means_text']}\n\n"
    b += f"{'═'*58}\n  Methodology inspired by Slated.com (Script/Team/Financial — 75+ = ready)\n"
    b += f"  {t['brief_super']}: Tarek ElNaggar · Lunim\n{'═'*58}"
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
# TABS — Market Viability  |  Pre-Production Readiness
# ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([T["tab_market"], T["tab_preprod"]])

with tab1:

  # ─────────────────────────────────────────────────────────
  # FORM
  # ─────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# TAB 2 — PRE-PRODUCTION READINESS
# ─────────────────────────────────────────────────────────────
with tab2:

  st.markdown(f'<p class="hero-sub" style="margin-bottom:1rem;">{T["preprod_subtitle"]}</p>', unsafe_allow_html=True)

  with st.form("preproduction_form"):

      st.markdown(f'<div class="section-header">{T["preprod_sec_project"]}</div>', unsafe_allow_html=True)
      pp_project_name = st.text_input(T["project_name"], placeholder=T["project_name_ph"], key="pp_name")

      st.markdown("<hr>", unsafe_allow_html=True)
      st.markdown(f'<div class="section-header">{T["preprod_sec_script"]}</div>', unsafe_allow_html=True)

      pcol1, pcol2, pcol3 = st.columns(3)
      with pcol1:
          script_completion = st.slider(T["script_completion"], 0, 100, 80, key="pp_completion")
          st.markdown(f'<p class="helper">{T["script_completion_help"]}</p>', unsafe_allow_html=True)
      with pcol2:
          script_quality = st.slider(T["script_quality"], 0, 100, 70, key="pp_quality")
          st.markdown(f'<p class="helper">{T["script_quality_help"]}</p>', unsafe_allow_html=True)
      with pcol3:
          genre_feasibility = st.slider(T["genre_feasibility"], 0.0, 10.0, 7.5, 0.5, key="pp_genre_fit")
          st.markdown(f'<p class="helper">{T["genre_feasibility_help"]}</p>', unsafe_allow_html=True)

      st.markdown("<hr>", unsafe_allow_html=True)
      st.markdown(f'<div class="section-header">{T["preprod_sec_team"]}</div>', unsafe_allow_html=True)

      pcol4, pcol5, pcol6 = st.columns(3)
      with pcol4:
          director_attached = st.slider(T["director_attached"], 0.0, 10.0, 6.5, 0.5, key="pp_director")
          st.markdown(f'<p class="helper">{T["director_attached_help"]}</p>', unsafe_allow_html=True)
      with pcol5:
          cast_attached = st.slider(T["cast_attached"], 0.0, 10.0, 5.0, 0.5, key="pp_cast")
          st.markdown(f'<p class="helper">{T["cast_attached_help"]}</p>', unsafe_allow_html=True)
      with pcol6:
          key_crew_readiness = st.slider(T["key_crew_readiness"], 0, 100, 60, key="pp_crew")
          st.markdown(f'<p class="helper">{T["key_crew_readiness_help"]}</p>', unsafe_allow_html=True)

      st.markdown("<hr>", unsafe_allow_html=True)
      st.markdown(f'<div class="section-header">{T["preprod_sec_financial"]}</div>', unsafe_allow_html=True)

      pcol7, pcol8 = st.columns(2)
      with pcol7:
          financing_secured = st.slider(T["financing_secured"], 0, 100, 70, key="pp_financing")
          st.markdown(f'<p class="helper">{T["financing_secured_help"]}</p>', unsafe_allow_html=True)
      with pcol8:
          location_confirmed = st.slider(T["location_confirmed"], 0.0, 10.0, 4.0, 0.5, key="pp_location")
          st.markdown(f'<p class="helper">{T["location_confirmed_help"]}</p>', unsafe_allow_html=True)

      pcol9, pcol10 = st.columns(2)
      with pcol9:
          distribution_interest = st.slider(T["distribution_interest"], 0.0, 10.0, 5.5, 0.5, key="pp_distribution")
          st.markdown(f'<p class="helper">{T["distribution_interest_help"]}</p>', unsafe_allow_html=True)
      with pcol10:
          production_timeline = st.slider(T["production_timeline"], 0.0, 10.0, 7.0, 0.5, key="pp_timeline")
          st.markdown(f'<p class="helper">{T["production_timeline_help"]}</p>', unsafe_allow_html=True)

      st.markdown("<hr>", unsafe_allow_html=True)
      on_location_label = st.radio(
          T["on_location_shoot"],
          options=[T["on_location_yes"], T["on_location_no"]],
          key="pp_on_location"
      )
      on_location_shoot = (on_location_label == T["on_location_yes"])

      st.markdown("<br>", unsafe_allow_html=True)
      pp_submitted = st.form_submit_button(T["analyze_preprod_btn"])

  # ─────────────────────────────────────────────────────────
  # PRE-PRODUCTION RESULTS
  # ─────────────────────────────────────────────────────────
  if pp_submitted:
      if not pp_project_name.strip():
          st.error(T["no_project_name"]); st.stop()

      pp_inputs = dict(
          script_completion=float(script_completion),
          script_quality=float(script_quality),
          genre_feasibility=float(genre_feasibility),
          director_attached=float(director_attached),
          cast_attached=float(cast_attached),
          key_crew_readiness=float(key_crew_readiness),
          financing_secured=float(financing_secured),
          location_confirmed=float(location_confirmed),
          distribution_interest=float(distribution_interest),
          production_timeline=float(production_timeline),
          on_location_shoot=on_location_shoot,
      )

      with st.spinner(T["spinner"]):
          pp_result = compute_preproduction_score(pp_inputs)

          # AbelCine recommendation text
          if pp_result["location_need_flag"]:
              if lang == "EN":
                  pp_result["abelcine_recommendation"] = (
                      f"AbelCine 3D LiDAR Scanning (PortalCam) recommended — on-location "
                      f"shoot confirmed, location not yet scanned/permitted "
                      f"(score: {location_confirmed}/10). Recommended BEFORE production begins."
                  )
              else:
                  pp_result["abelcine_recommendation"] = (
                      f"يُنصح بخدمة المسح الثلاثي الأبعاد LiDAR من AbelCine (PortalCam) — "
                      f"تم تأكيد التصوير في موقع خارجي، ولكن الموقع لم يتم مسحه/ترخيصه بعد "
                      f"(التقييم: {location_confirmed}/10). يُنصح بهذا قبل بدء الإنتاج."
                  )
          else:
              pp_result["abelcine_recommendation"] = None

          pp_brief = generate_preproduction_brief(pp_project_name, pp_inputs, pp_result, lang)

      st.markdown("---")
      st.markdown(f'<div class="section-header">{T["preprod_results"]} — {pp_project_name}</div>', unsafe_allow_html=True)

      # Overall score + tier
      overall = pp_result["overall_score"]
      tier_key = pp_result["tier_key"]
      tier_color = {"ready_75": GREEN, "developing_55": GOLD, "early_35": "#f6a623", "not_ready": RED}[tier_key]
      tier_class = {"ready_75": "invest", "developing_55": "caution", "early_35": "review", "not_ready": "dont"}[tier_key]

      pp_c1, pp_c2, pp_c3, pp_c4 = st.columns(4)
      with pp_c1:
          st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{tier_color}">{overall}</div><div class="metric-label">{T["overall_score"]}</div></div>', unsafe_allow_html=True)
      with pp_c2:
          st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{TEAL}">{pp_result["script_score"]}</div><div class="metric-label">{T["script_score_label"]}</div></div>', unsafe_allow_html=True)
      with pp_c3:
          st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{TEAL}">{pp_result["team_score"]}</div><div class="metric-label">{T["team_score_label"]}</div></div>', unsafe_allow_html=True)
      with pp_c4:
          st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{TEAL}">{pp_result["financial_score"]}</div><div class="metric-label">{T["financial_score_label"]}</div></div>', unsafe_allow_html=True)

      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown(f'<div class="decision-{tier_class}">{T[tier_key]}  —  {overall}/100</div>', unsafe_allow_html=True)

      st.markdown("<br>", unsafe_allow_html=True)
      cat_names = {
          "EN": {"script": "Script", "team": "Team", "financial": "Financial/Logistics"},
          "AR": {"script": "السيناريو", "team": "الفريق", "financial": "التمويل/اللوجستيات"},
      }[lang]
      pp_s1, pp_s2 = st.columns(2)
      with pp_s1:
          st.markdown(f'<div class="section-header">{T["strongest_area"]}</div>', unsafe_allow_html=True)
          st.markdown(f"✓ {cat_names[pp_result['strongest']]}")
      with pp_s2:
          st.markdown(f'<div class="section-header">{T["weakest_area"]}</div>', unsafe_allow_html=True)
          st.markdown(f"⚠ {cat_names[pp_result['weakest']]}")

      # AbelCine flag
      if pp_result["abelcine_recommendation"]:
          st.markdown("<br>", unsafe_allow_html=True)
          st.markdown(f'<div class="escalation-box"><strong>{T["abelcine_flag_title"]}</strong><br>{pp_result["abelcine_recommendation"]}</div>', unsafe_allow_html=True)

      # Full brief
      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown(f'<div class="section-header">{T["preprod_brief_title"]}</div>', unsafe_allow_html=True)
      st.markdown(f'<div class="brief-box">{pp_brief}</div>', unsafe_allow_html=True)
      st.download_button(T["preprod_download"], data=pp_brief,
                         file_name=f"preproduction_{pp_project_name.replace(' ','_')}.txt", mime="text/plain")
      st.markdown(f"<p style='color:{MGRAY}; font-size:0.75rem; text-align:center; margin-top:2rem;'>Film Investment Intelligence Agent v4 · Lunim · {datetime.now().strftime('%B %d, %Y')}</p>", unsafe_allow_html=True)

  else:
      st.markdown(f"""
      <div style='text-align:center; padding:3rem 0; color:{MGRAY};'>
          <div style='font-size:3rem; margin-bottom:1rem;'>🏗️</div>
          <p style='font-size:1.1rem; color:{SUBTEXT};'>{T['preprod_welcome']} <strong style='color:{TEAL}'>{T['analyze_preprod_btn']}</strong></p>
      </div>""", unsafe_allow_html=True)
