"""
ProdAdvisor — Application d'optimisation de production textile par Intelligence Artificielle.

Dashboard Streamlit complet avec :
  - Analyse exploratoire enrichie (EDA)
  - Predictions IA via Prophet et Machine Learning (Random Forest)
  - Simulateur d'élasticité prix
  - Recommandations de production
  - Assistant chatbot ProdBot
  - Analyse de rentabilite (ABC) et Fidelisation (RFM)
  - Export de rapport PDF

"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
from datetime import datetime


from src.data_loader import load_and_prepare_data, get_monthly_data, get_category_stats, calculate_rfm
from src.chatbot_engine import ProdBotEngine
from src.ml_models import train_satisfaction_model, predict_satisfaction
from src.pdf_generator import generate_executive_pdf
from src.forecasting import run_forecast, get_forecast_dataframe

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ProdAdvisor — Optimisation Textile IA",
    page_icon="PA",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<div id='top-anchor'></div>", unsafe_allow_html=True)

def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* Hide Streamlit Default UI Elements */
        #MainMenu {visibility: hidden;}
        [data-testid="stHeader"] {background-color: transparent;}
        .stDeployButton {display: none;}
        
        :root {
            --primary: #5b6df8; --primary-light: #717ff4; --secondary: #1b8755;
            --accent: #d97706; --danger: #e11d48; --dark: #0b0e14;
            --text-primary: #ffffff; --text-secondary: #a3aab5; --text-muted: #525964;
            --bg-card: rgba(21, 24, 34, 0.6); /* Semi-transparent background */
            --border-color: rgba(255, 255, 255, 0.08);
            --radius: 16px; --radius-sm: 10px;
        }
        
        /* Premium Background Gradient */
        .stApp { 
            background: radial-gradient(circle at 15% 50%, rgba(91, 109, 248, 0.08), transparent 25%),
                        radial-gradient(circle at 85% 30%, rgba(138, 43, 226, 0.08), transparent 25%),
                        #080a0f !important;
        }
        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; letter-spacing: -0.01em; }
        
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
        
        /* Header Title */
        .main-header { display: flex; align-items: center; gap: 16px; margin-bottom: 32px; margin-top: -30px; }
        .logo-icon { 
            background: linear-gradient(135deg, #4f8bf9 0%, #5b6df8 100%); 
            width: 48px; height: 48px; border-radius: 14px; 
            display: flex; align-items: center; justify-content: center; 
            font-weight: 900; font-size: 1.5rem; color: white; 
            box-shadow: 0 8px 24px rgba(91, 109, 248, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.3); 
            border: 1px solid rgba(255,255,255,0.1); letter-spacing: -1px;
            margin-top: 6px;
        }
        .main-title { color: #ffffff; font-weight: 800; font-size: 1.8rem; margin: 0 0 -8px 0 !important; line-height: 1.1; letter-spacing: -0.03em;}
        .main-title .highlight { background: linear-gradient(135deg, #a6b1ff, #5b6df8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .premium-badge { font-size: 0.65rem; background: linear-gradient(90deg, rgba(91, 109, 248, 0.2), rgba(138, 43, 226, 0.2)); color: #a6b1ff; padding: 3px 8px; border-radius: 6px; border: 1px solid rgba(91, 109, 248, 0.3); text-transform: uppercase; font-weight: 800; letter-spacing: 0.05em;}
        .subtitle { color: #8b949e; font-size: 0.85rem; margin: -14px 0 0 0 !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;}
        
        /* Glassmorphism Metric Cards & Animations */
        .metric-card { 
            background: var(--bg-card); 
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border-radius: var(--radius); padding: 24px; 
            border: 1px solid var(--border-color); 
            box-shadow: 0 8px 32px rgba(0,0,0,0.2); 
            height: 100%; min-height: 180px; position: relative; 
            display: flex; flex-direction: column; justify-content: flex-start;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(91, 109, 248, 0.15);
            border-color: rgba(91, 109, 248, 0.3);
        }
        .metric-label { color: var(--text-secondary); font-size: 0.85rem; font-weight: 600; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.03em;}
        .metric-value { font-size: 2.4rem; font-weight: 800; color: var(--text-primary); margin-bottom: 12px; line-height: 1; letter-spacing: -0.02em;}
        
        .badge-green { background: rgba(46, 204, 113, 0.15); color: #2ecc71; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; display: inline-flex; align-items: center; gap: 4px; border: 1px solid rgba(46, 204, 113, 0.2); }
        .badge-orange { background: rgba(243, 156, 18, 0.15); color: #f39c12; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; display: inline-flex; align-items: center; border: 1px solid rgba(243, 156, 18, 0.2); }
        .badge-purple { background: rgba(138, 43, 226, 0.15); color: #b388ff; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; display: inline-flex; align-items: center; border: 1px solid rgba(138, 43, 226, 0.2); }
        
        /* IA Card Glassmorphism */
        .ia-card { 
            background: linear-gradient(135deg, rgba(42, 32, 77, 0.8) 0%, rgba(23, 22, 59, 0.8) 100%); 
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(138, 43, 226, 0.3); 
            box-shadow: 0 10px 40px rgba(74, 56, 122, 0.3); 
            border-radius: var(--radius); padding: 24px; 
            height: 100%; min-height: 180px; display: flex; flex-direction: column; justify-content: flex-start;
            transition: transform 0.3s ease;
        }
        .ia-card:hover { transform: translateY(-4px); box-shadow: 0 15px 50px rgba(74, 56, 122, 0.4); }
        .ia-status { font-size: 0.75rem; color: #2ecc71; display: flex; align-items: center; gap: 6px; font-weight: 700; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.05em;}
        .ia-status::before { content: ''; width: 8px; height: 8px; background: #2ecc71; border-radius: 50%; box-shadow: 0 0 10px #2ecc71; }
        .ia-title { font-size: 1.5rem; font-weight: 800; color: white; margin-bottom: 22px; letter-spacing: -0.01em;}
        .ia-btn { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15); color: white; padding: 10px 20px; border-radius: 10px; font-size: 0.9rem; font-weight: 600; display: inline-flex; align-items: center; gap: 8px; cursor: pointer; transition: all 0.3s; }
        .ia-btn:hover { background: rgba(255,255,255,0.15); border-color: rgba(255,255,255,0.3); transform: scale(1.02);}
        
        /* Charts & Elements */
        .chart-container { 
            background: var(--bg-card); 
            backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
            border-radius: var(--radius); padding: 12px 20px; 
            border: 1px solid var(--border-color); 
            margin-top: 16px; margin-bottom: 8px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
        }
        .chart-container:hover { border-color: rgba(255,255,255,0.15); }
        .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0px; }
        .chart-title { color: white !important; font-weight: 700 !important; font-size: 1.05rem !important; margin: 0 !important; letter-spacing: 0.01em;}
        .chart-action { background: rgba(255,255,255,0.05); color: #a3aab5; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 5px 12px; font-size: 0.8rem; transition: background 0.2s;}
        .chart-action:hover { background: rgba(255,255,255,0.1); color: white; }
        
        h2, h3 { color: var(--text-primary) !important; font-weight: 800 !important; letter-spacing: -0.02em;}
        hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, var(--border-color), transparent); margin: 35px 0; }
        
        [data-testid="stSidebar"] { background: rgba(11, 14, 20, 0.8) !important; backdrop-filter: blur(20px); border-right: 1px solid var(--border-color); }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #a3aab5 !important; }
        
        /* Tabs styling Premium */
        .stTabs [data-baseweb="tab-list"] { gap: 12px; background: transparent; padding: 0; border: none; box-shadow: none; margin-bottom: 20px; }
        .stTabs [data-baseweb="tab"] { background: rgba(255,255,255,0.02); border-radius: 10px; padding: 10px 20px; font-weight: 600; font-size: 0.95rem; color: #8b949e; border: 1px solid rgba(255,255,255,0.05); transition: all 0.3s;}
        .stTabs [data-baseweb="tab"]:hover { color: white; background: rgba(255,255,255,0.06); transform: translateY(-2px);}
        .stTabs [aria-selected="true"] { background: linear-gradient(180deg, rgba(91, 109, 248, 0.15) 0%, rgba(91, 109, 248, 0.05) 100%) !important; color: white !important; border: 1px solid rgba(91, 109, 248, 0.4) !important; box-shadow: 0 4px 15px rgba(91,109,248,0.15); }
        
        .status-badge { display: inline-block; padding: 4px 14px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.5px; }
        .status-badge.active { background: rgba(46, 204, 113, 0.12); color: #2ecc71; border: 1px solid rgba(46, 204, 113, 0.3); }
        .stDownloadButton button { background: linear-gradient(135deg, var(--primary), #8a2be2) !important; color: white !important; border: none !important; border-radius: var(--radius-sm) !important; padding: 12px 28px !important; font-weight: 700 !important; font-size: 0.95rem !important; transition: all 0.3s ease !important; box-shadow: 0 4px 15px rgba(91, 109, 248, 0.4) !important;}
        .stDownloadButton button:hover { transform: translateY(-3px) !important; box-shadow: 0 8px 25px rgba(91, 109, 248, 0.6) !important; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

st.html("""
<script>
    const parentDoc = window.parent ? window.parent.document : document;
    
    // FAB Button (Bottom Right)
    setInterval(() => {
        if (!parentDoc.getElementById("chatbot-fab")) {
            const btn = parentDoc.createElement("div");
            btn.id = "chatbot-fab";
            btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>`;
            btn.title = "Ouvrir l'Assistant ProdBot";
            btn.style.cssText = `
                position: fixed; bottom: 28px; right: 28px;
                background: #0f4c81; color: white; width: 52px; height: 52px; border-radius: 12px;
                display: flex; justify-content: center; align-items: center;
                cursor: pointer; box-shadow: 0 4px 16px rgba(15,76,129,0.3); z-index: 999999;
            `;
            btn.onclick = () => {
                const tabs = Array.from(parentDoc.querySelectorAll('button[role="tab"]'));
                const iaTab = tabs.find(t => t.textContent && t.textContent.includes('Conseiller IA'));
                if (iaTab) iaTab.click();
            };
            parentDoc.body.appendChild(btn);
        }
    }, 1000);
</script>
""")

st.markdown("""
<div class='main-header'>
    <div class='logo-icon'>PA</div>
    <div style='display: flex; flex-direction: column; justify-content: center;'>
        <h1 class='main-title'>Prod<span class='highlight'>Advisor</span></h1>
        <p class='subtitle'>Planification textile & supply chain IA</p>
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def cached_load_data():
    return load_and_prepare_data(os.path.dirname(os.path.abspath(__file__)))

df = cached_load_data()

if df is None:
    st.error("Aucun jeu de donnees trouve. Veuillez placer un fichier dataset.csv dans le dossier data/.")
    st.stop()



# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### ProdAdvisor v3.0")
st.sidebar.markdown("<span class='status-badge active'>Systeme Actif</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("#### Parametres de prediction")
marge_securite = st.sidebar.slider("Marge de securite (%)", 0, 50, 10, 5)
horizon_prediction = st.sidebar.slider("Horizon de prediction (mois)", 1, 6, 3, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Filtres de donnees")
categories_dispos = sorted(df['Categorie'].unique().tolist())
categories_selectionnees = st.sidebar.multiselect("Filtrer par categorie", categories_dispos, default=categories_dispos)
date_min, date_max = df['Date'].min().date(), df['Date'].max().date()
date_range = st.sidebar.date_input("Periode d'analyse", value=(date_min, date_max), min_value=date_min, max_value=date_max)

df_filtered = df.copy()
if categories_selectionnees:
    df_filtered = df_filtered[df_filtered['Categorie'].isin(categories_selectionnees)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    df_filtered = df_filtered[(df_filtered['Date'].dt.date >= date_range[0]) & (df_filtered['Date'].dt.date <= date_range[1])]

st.sidebar.markdown("---")
if len(df_filtered) > 0:
    st.sidebar.markdown(f"**{len(df_filtered):,}** transactions")
    
    # Bouton d'export PDF
    cat_stats = get_category_stats(df_filtered)
    rfm_stats = calculate_rfm(df_filtered)
    pdf_bytes = generate_executive_pdf(df_filtered, cat_stats, rfm_stats)
    st.sidebar.download_button(
        label="Telecharger Rapport Executif PDF",
        data=bytes(pdf_bytes),
        file_name=f"Rapport_ProdAdvisor_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime='application/pdf'
    )

if len(df_filtered) == 0:
    st.warning("Aucune donnee ne correspond aux filtres.")
    st.stop()

df_monthly = get_monthly_data(df_filtered)
cat_stats = get_category_stats(df_filtered)
rfm_stats = calculate_rfm(df_filtered)

PALETTE = ['#0f4c81', '#1a6fb5', '#2ecc71', '#e67e22', '#9b59b6', '#1abc9c', '#e74c3c', '#34495e', '#f39c12', '#2c3e50']

@alt.theme.register('prodadvisor', enable=True)
def prodadvisor_theme():
    return alt.theme.ThemeConfig(
        **{
            'config': {
                'view': {'strokeWidth': 0},
                'axis': {
                    'labelFont': 'Inter', 'titleFont': 'Inter', 
                    'labelFontSize': 11, 'titleFontSize': 12, 
                    'titleColor': '#8b949e', 'labelColor': '#8b949e', 
                    'gridColor': '#262b3d', 'domainColor': '#262b3d',
                    'tickColor': '#262b3d'
                },
                'title': {'font': 'Inter', 'fontSize': 14, 'color': '#ffffff'},
                'background': 'transparent',
                'legend': {'labelColor': '#8b949e', 'titleColor': '#8b949e'}
            }
        }
    )

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Tableau de bord", "Ventes & Analyse", "Lancements", "Stocks", "Historique", "Conseiller IA"
])

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 : VUE D'ENSEMBLE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab1:
    # --- Metrics Computation with Empty State Checks ---
    if not df_filtered.empty:
        total_sales = df_filtered['Quantite'].sum()
        total_revenue = (df_filtered['Quantite'] * df_filtered['Prix_Unitaire']).sum()
        panier_moyen = total_revenue / len(df_filtered) if len(df_filtered) > 0 else 0
        
        try:
            top_cat = df_filtered.groupby('Categorie')['Quantite'].sum().idxmax()
            top_cat_data = df_filtered[df_filtered['Categorie'] == top_cat]
            top_size = top_cat_data['Taille'].value_counts().idxmax() if ('Taille' in top_cat_data.columns and not top_cat_data['Taille'].empty and len(top_cat_data['Taille'].value_counts())>0) else 'N/A'
        except Exception:
            top_cat = "N/A"
            top_size = "N/A"
            
        nb_clients = df_filtered['Client_ID'].nunique() if 'Client_ID' in df_filtered.columns else 0
        if rfm_stats is not None and not rfm_stats.empty:
            vip_pct = (len(rfm_stats[rfm_stats['Segment'].isin(['VIP', 'Fideles'])]) / len(rfm_stats)) * 100
        else:
            vip_pct = 0
    else:
        total_sales = 0
        total_revenue = 0
        panier_moyen = 0
        top_cat = "N/A"
        top_size = "N/A"
        nb_clients = 0
        vip_pct = 0
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Chiffre d'Affaires</div>
            <div class='metric-value'>{total_revenue:,.0f} €</div>
            <div class='badge-green'>Panier moyen : {panier_moyen:,.0f} €</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Top Produit Vendu</div>
            <div class='metric-value'>{top_cat}</div>
            <div class='badge-orange'>Taille préférée : {top_size}</div>
            <div style='color: #8b949e; font-size: 0.75rem; margin-top: 8px;'>Total : <b style='color: white;'>{total_sales:,}</b> unités</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Fidélité Client</div>
            <div class='metric-value'>{nb_clients:,} <span style='font-size: 1rem; color: #8b949e; font-weight: 500;'>clients</span></div>
            <div class='badge-purple'>{vip_pct:.1f}% de clients VIP</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='ia-card'>
            <div class='ia-status'>Statut du Système</div>
            <div class='ia-title' style='margin-bottom: 0;'>Assistant Actif</div>
            <div style='color: #8b949e; font-size: 0.8rem; margin-top: 10px;'>Prêt à analyser vos données. Rendez-vous dans l'onglet "Conseiller IA".</div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("<div class='chart-container'><div class='chart-header'><h3 class='chart-title'>Évolution mensuelle</h3><div class='chart-action'>Volume</div></div>", unsafe_allow_html=True)
        if not df_monthly.empty:
            with st.spinner("Génération des prévisions..."):
                df_combined, _, _ = get_forecast_dataframe(df_monthly, horizon_prediction)
            
            # Zone pour l'historique uniquement
            area = alt.Chart(df_combined[df_combined['Type'] == 'Historique']).mark_area(
                opacity=0.1, 
                color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='#717ff4', offset=0), alt.GradientStop(color='rgba(113, 127, 244, 0)', offset=1)], x1=1, x2=1, y1=1, y2=0)
            ).encode(
                x=alt.X('Date:T', title=None, axis=alt.Axis(grid=False, domainColor='transparent', tickColor='transparent', labelColor='#8b949e', format='%b %Y', labelAngle=-45, values=df_combined['Date'].unique().tolist())),
                y=alt.Y('Quantite:Q', title=None, axis=alt.Axis(grid=True, gridColor='#262b3d', gridDash=[4,4], domainColor='transparent', tickColor='transparent', labelColor='#8b949e'))
            ).properties(height=320)
            
            # Duplication du dernier point historique pour assurer la continuité de la courbe
            last_hist = df_combined[df_combined['Type'] == 'Historique'].iloc[-1:].copy()
            last_hist['Type'] = 'Prévision'
            df_plot = pd.concat([df_combined, last_hist], ignore_index=True)
            
            # Lignes (Continue pour historique, pointillée pour prévision)
            line = alt.Chart(df_plot).mark_line(
                strokeWidth=3, interpolate='monotone'
            ).encode(
                x='Date:T', 
                y='Quantite:Q',
                color=alt.Color('Type:N', scale=alt.Scale(domain=['Historique', 'Prévision'], range=['#717ff4', '#f39c12']), legend=alt.Legend(title=None, orient='top', labelColor='#8b949e')),
                strokeDash=alt.condition(alt.datum.Type == 'Prévision', alt.value([5, 5]), alt.value([0]))
            )
            
            # Points historiques avec CA
            points = alt.Chart(df_plot[df_plot['Type'] == 'Historique']).mark_circle(
                size=60, color='#151822', opacity=1, stroke='#717ff4', strokeWidth=2
            ).encode(
                x='Date:T', 
                y='Quantite:Q',
                tooltip=['Date:T', 'Quantite:Q', 'CA:Q']
            )
            
            # Points de prévision sans CA
            pred_points = alt.Chart(df_plot[(df_plot['Type'] == 'Prévision') & (df_plot['Date'] > df_combined[df_combined['Type']=='Historique']['Date'].max())]).mark_circle(
                size=60, color='#151822', opacity=1, stroke='#f39c12', strokeWidth=2
            ).encode(
                x='Date:T', 
                y='Quantite:Q',
                tooltip=['Date:T', 'Quantite:Q', 'Type:N']
            )
            
            st.altair_chart(area + line + points + pred_points, width='stretch')
        else:
            st.info("Aucune donnée temporelle disponible.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='chart-container'><div class='chart-header'><h3 class='chart-title'>Répartition des Ventes</h3><div class='chart-action'>Top 5</div></div>", unsafe_allow_html=True)
        if not cat_stats.empty:
            donut = alt.Chart(cat_stats.head(5)).mark_arc(
                innerRadius=80, stroke='#151822', strokeWidth=4
            ).encode(
                theta='Volume:Q',
                color=alt.Color('Categorie:N', scale=alt.Scale(range=['#b388ff', '#2ecc71', '#4f8bf9', '#f368e0', '#f39c12']), legend=None),
                tooltip=['Categorie', 'Volume', 'CA']
            ).properties(height=320)
            
            legend_html = "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; padding: 0 10px;'>"
            colors = ['#b388ff', '#2ecc71', '#4f8bf9', '#f368e0', '#f39c12']
            for i, row in cat_stats.head(5).iterrows():
                legend_html += f"<div style='display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #8b949e;'><div style='width: 10px; height: 10px; border-radius: 50%; background: {colors[i%len(colors)]};'></div><span style='color: white; font-weight: 500;'>{row['Categorie']}</span> ({row['Volume']})</div>"
            legend_html += "</div>"
            
            st.altair_chart(donut, width='stretch')
            st.markdown(legend_html, unsafe_allow_html=True)
        else:
            st.info("Aucune catégorie disponible.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # NOUVEAU: Top 5 Dataframe
    st.markdown("<div class='chart-container'><div class='chart-header'><h3 class='chart-title'>Détail des ventes</h3></div>", unsafe_allow_html=True)
    if not cat_stats.empty:
        top_details = cat_stats[['Categorie', 'Volume', 'CA', 'Prix_Moyen']].head(5).copy()
        top_details['Prix_Moyen'] = top_details['Prix_Moyen'].apply(lambda x: f"{x:.2f} €")
        top_details['CA'] = top_details['CA'].apply(lambda x: f"{x:,.0f} €")
        st.dataframe(top_details, width='stretch', hide_index=True)
    else:
        st.info("Aucune donnée disponible.")
    st.markdown("</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 : IA & SIMULATIONS                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.header("Intelligence Artificielle & Simulations Avancees")
    
    st.subheader("1. Simulateur d'Elasticite Prix")
    st.markdown("Evaluez l'impact d'une variation de prix sur vos volumes de vente (Modele d'elasticite simplifie).")
    
    cat_sim = st.selectbox("Choisir une categorie a simuler :", categories_dispos)
    variation_prix = st.slider("Variation du prix de vente (%) :", min_value=-50, max_value=50, value=0, step=1)
    
    cat_data = cat_stats[cat_stats['Categorie'] == cat_sim].iloc[0]
    prix_actuel = cat_data['Prix_Moyen']
    vol_actuel = cat_data['Volume']
    ca_actuel = cat_data['CA']
    
    elasticite = st.slider("Coefficient d'élasticité", min_value=-5.0, max_value=5.0, value=-1.5, step=0.1)
    variation_vol = elasticite * (variation_prix / 100)
    
    nouveau_prix = prix_actuel * (1 + (variation_prix / 100))
    nouveau_vol = max(0, vol_actuel * (1 + variation_vol))
    nouveau_ca = nouveau_prix * nouveau_vol
    
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Prix Simule", f"{nouveau_prix:,.1f} $", f"{variation_prix}%")
    sc2.metric("Volume Estime", f"{int(nouveau_vol):,} pcs", f"{variation_vol*100:+.1f}%")
    sc3.metric("Chiffre d'Affaires Projete", f"{nouveau_ca:,.0f} $", f"{(nouveau_ca - ca_actuel)/ca_actuel*100:+.1f}%")
    
    st.markdown("---")
    
    st.subheader("2. Machine Learning : Prediction de Satisfaction (XGBoost Fine-Tuned)")
    st.markdown("Le modele ML (XGBoost) apprend de l'historique pour predire la note qu'obtiendra une nouvelle combinaison produit.")
    
    model, metrics, importances = train_satisfaction_model(df_filtered)
    
    if model is not None:
        p1, p2, p3, p4 = st.columns(4)
        pred_cat = p1.selectbox("Categorie", categories_dispos, key="ml_cat")
        pred_taille = p2.selectbox("Taille", ['XS', 'S', 'M', 'L', 'XL', 'XXL'])
        pred_couleur = p3.selectbox("Couleur", ['Noir', 'Blanc', 'Bleu', 'Rouge', 'Vert', 'Beige', 'Rose'])
        pred_prix = p4.number_input("Prix cible ($)", value=float(cat_data['Prix_Moyen']))
        
        predicted_score = predict_satisfaction(model, pred_cat, pred_taille, pred_couleur, pred_prix)
        
        st.markdown(f"""
        <div style='background: var(--bg-card); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); padding: 20px; border-radius: var(--radius); text-align: center; margin-top: 10px; border: 1px solid var(--border-color); box-shadow: 0 8px 32px rgba(0,0,0,0.15); transition: transform 0.3s ease;' onmouseover='this.style.transform="translateY(-4px)"' onmouseout='this.style.transform="translateY(0)"'>
            <h4 style='margin:0; color: #ffffff;'>Note Cliente Predite :</h4>
            <div style='font-size: 2.2rem; font-weight: 800; color: {'#2ecc71' if predicted_score>=3.8 else '#e67e22'}; margin: 10px 0;'>{predicted_score:.2f} / 5</div>
            <p style='margin:0; font-size: 0.8rem; color: #a3aab5;'>Prevu avec une precision (R2) de {metrics['R2']:.2f} | MAE: {metrics['MAE']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Donnees insuffisantes pour entrainer le modele de Machine Learning (Satisfaction).")
        
    st.markdown("---")
    
    st.subheader("3. Previsions Temporelles (Time Series)")
    if len(df_monthly) >= 3:
        with st.spinner("Analyse des tendances en cours..."):
            forecast_result = run_forecast(df_monthly, horizon=1)
        
        next_month_pred = forecast_result['prediction']
        model_name = forecast_result['model_used']
        
        color = "#b388ff" if model_name == "TimeGPT" else "#4f8bf9"
        
        st.markdown(f"""
        <div style='border-left: 4px solid {color}; padding-left: 15px; margin-bottom: 20px;'>
            <div style='font-size: 0.9rem; color: #8b949e; text-transform: uppercase; font-weight: 600;'>Modele Actif</div>
            <div style='font-size: 1.2rem; font-weight: 700; color: white;'>Propulse par {model_name}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Prevision Volume (Mois M+1)", f"{next_month_pred:,} pieces", "Demande projetee")
    else:
        next_month_pred = 0
        st.info("Donnees insuffisantes pour une prevision temporelle.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 : RECOMMANDATIONS                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.header("Recommandations de production")
    n_months_data = max(1, df_filtered['Date'].dt.to_period('M').nunique())
    avg_monthly_demand = df_filtered['Quantite'].sum() / n_months_data
    stock_estime = int(avg_monthly_demand * 0.5)
    
    base_prediction = next_month_pred if next_month_pred > 0 else int(avg_monthly_demand)
    objectif_stock = int(base_prediction * (1 + (marge_securite / 100)))
    qte_a_produire = max(0, objectif_stock - stock_estime)

    st.markdown(f"""
    <div class='production-card'>
        <h3 style='margin-bottom: 10px;'>Objectif de production Net</h3>
        <div style='display: flex; align-items: center; gap: 12px; flex-wrap: wrap; font-size: 1.05rem;'>
            <span>Objectif Stock [Demande (<b>{base_prediction:,}</b>) + Marge (<b>{marge_securite}%</b>) = <b>{objectif_stock:,}</b>] - Stock Actuel (<b>{stock_estime:,}</b>) =</span>
            <span class='production-highlight' style='margin: 0; padding: 4px 12px; font-size: 1.2rem; border-radius: 8px;'>{qte_a_produire:,} pieces</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Plan de repartition par categorie")
    cat_pct = df_filtered['Categorie'].value_counts(normalize=True)
    reco_data = [{"Produit": cat, "Qte a produire": int(qte_a_produire * cat_pct[cat]), "Part": f"{cat_pct[cat]*100:.1f}%"} for cat in cat_pct.index[:10]]
    st.dataframe(pd.DataFrame(reco_data), width='stretch', hide_index=True)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 : FIDELISATION (RFM)                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.header("Fidelisation et Analyse de Cohorte (RFM)")
    if rfm_stats is not None and not rfm_stats.empty:
        st.markdown("L'analyse **RFM** (Recence, Frequence, Montant) segmente vos clients selon leur comportement d'achat historique.")
        
        col_rfm1, col_rfm2 = st.columns(2)
        with col_rfm1:
            segment_counts = rfm_stats['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Clients']
            bar_rfm = alt.Chart(segment_counts).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('Clients:Q', title='Nombre de clients'),
                y=alt.Y('Segment:N', sort='-x', title=''),
                color=alt.Color('Segment:N', scale=alt.Scale(range=['#2ecc71', '#0f4c81', '#f39c12', '#c0392b', '#95a5b6']), legend=None)
            ).properties(height=350, title="Repartition des segments clients")
            st.altair_chart(bar_rfm, width='stretch')
            
        with col_rfm2:
            if 'R' in rfm_stats.columns and 'F' in rfm_stats.columns:
                # Création de la matrice RFM
                rfm_matrix = rfm_stats.groupby(['R', 'F']).size().reset_index(name='Clients')
                
                heatmap = alt.Chart(rfm_matrix).mark_rect(cornerRadius=4, stroke='#151822', strokeWidth=2).encode(
                    x=alt.X('R:O', title='Récence (4 = Très Récent)', sort='descending'),
                    y=alt.Y('F:O', title='Fréquence (4 = Très Fréquent)', sort='descending'),
                    color=alt.Color('Clients:Q', scale=alt.Scale(scheme='purples'), legend=None),
                    tooltip=['R', 'F', 'Clients']
                ).properties(height=350, title="Matrice RFM (Récence vs Fréquence)")
                
                text = alt.Chart(rfm_matrix).mark_text(baseline='middle', fontSize=16, fontWeight='bold').encode(
                    x=alt.X('R:O', sort='descending'),
                    y=alt.Y('F:O', sort='descending'),
                    text='Clients:Q',
                    color=alt.condition(
                        alt.datum.Clients > rfm_matrix['Clients'].max() / 2,
                        alt.value('white'),
                        alt.value('#a6b1ff')
                    )
                )
                
                st.altair_chart(heatmap + text, width='stretch')
            else:
                scatter_rfm = alt.Chart(rfm_stats).mark_circle(opacity=0.7).encode(
                    x=alt.X('Recence:Q', title='Recence (Jours)'),
                    y=alt.Y('Montant:Q', title='Montant Total ($)'),
                    color=alt.Color('Segment:N'),
                    tooltip=['Client_ID', 'Segment', 'Recence', 'Frequence', 'Montant']
                ).properties(height=350, title="Cartographie RFM").interactive()
                st.altair_chart(scatter_rfm, width='stretch')
            
        st.markdown("---")
        st.dataframe(rfm_stats[['Client_ID', 'Recence', 'Frequence', 'Montant', 'RFM_Score', 'Segment']].head(15), width='stretch', hide_index=True)
    else:
        st.warning("Impossible de calculer le RFM : La colonne 'Client_ID' est manquante ou invalide.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 5 : RENTABILITE                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab5:
    st.header("Analyse de rentabilite (Loi de Pareto ABC)")
    st.dataframe(cat_stats[['Categorie', 'CA', 'Volume', 'Classe_ABC']], width='stretch', hide_index=True)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 6 : ASSISTANT                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab6:
    st.header("Assistant Decisionnel ProdBot")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour, je suis ProdBot. Comment puis-je vous aider ?"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Tapez votre question ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        engine = ProdBotEngine(df_filtered, prediction=next_month_pred, cat_stats=cat_stats, rfm_stats=rfm_stats)
        with st.chat_message("assistant"):
            response_stream = engine.generate_response_stream(prompt, chat_history=st.session_state.messages)
            response = st.write_stream(response_stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
