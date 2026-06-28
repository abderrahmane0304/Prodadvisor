import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os
import streamlit as st

class PDFReport(FPDF):
    def header(self):
        # Logo placeholder or Title
        self.set_font("helvetica", "B", 18)
        self.set_text_color(15, 76, 129) # Primary color
        self.cell(0, 10, "ProdAdvisor - Rapport Executif", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("helvetica", "I", 10)
        self.set_text_color(107, 124, 147)
        self.cell(0, 10, f"Genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')}", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(150, 165, 182)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("helvetica", "B", 14)
        self.set_text_color(28, 35, 49)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        # Underline
        self.set_draw_color(15, 76, 129)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(5)

    def metric_row(self, label, value):
        self.set_font("helvetica", "B", 11)
        self.set_text_color(107, 124, 147)
        self.cell(80, 8, label)
        self.set_font("helvetica", "B", 11)
        self.set_text_color(28, 35, 49)
        self.cell(0, 8, str(value), new_x="LMARGIN", new_y="NEXT")

@st.cache_data(show_spinner=False)
def generate_executive_pdf(df: pd.DataFrame, cat_stats: pd.DataFrame, rfm_stats: pd.DataFrame = None) -> bytes:
    """Génère un rapport PDF et retourne son contenu en bytes."""
    pdf = PDFReport()
    pdf.add_page()
    
    # 1. Vue d'ensemble
    pdf.section_title("1. Synthese de l'activite")
    total_sales = df['Quantite'].sum()
    total_revenue = (df['Quantite'] * df['Prix_Unitaire']).sum()
    nb_transactions = len(df)
    
    pdf.metric_row("Chiffre d'Affaires Total :", f"{total_revenue:,.0f} $")
    pdf.metric_row("Volume Total Vendu :", f"{total_sales:,} pieces")
    pdf.metric_row("Nombre de Transactions :", f"{nb_transactions:,}")
    if 'Avis_Client' in df.columns:
        pdf.metric_row("Note Client Moyenne :", f"{df['Avis_Client'].mean():.2f} / 5")
    pdf.ln(10)
    
    # 2. Top Produits (ABC)
    pdf.section_title("2. Top Categories (Classe A)")
    pdf.set_font("helvetica", "", 10)
    
    # Table header
    pdf.set_fill_color(240, 242, 246)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(50, 8, "Categorie", border=1, fill=True)
    pdf.cell(40, 8, "CA ($)", border=1, align="R", fill=True)
    pdf.cell(30, 8, "Volume", border=1, align="R", fill=True)
    pdf.cell(30, 8, "Part CA (%)", border=1, align="R", fill=True, new_x="LMARGIN", new_y="NEXT")
    
    # Table content
    pdf.set_font("helvetica", "", 10)
    top_cats = cat_stats[cat_stats['Classe_ABC'] == 'A'].head(10)
    for _, row in top_cats.iterrows():
        pdf.cell(50, 8, str(row['Categorie']), border=1)
        pdf.cell(40, 8, f"{row['CA']:,.0f}", border=1, align="R")
        pdf.cell(30, 8, f"{row['Volume']:,}", border=1, align="R")
        pdf.cell(30, 8, f"{row['Part_CA']:.1f}%", border=1, align="R", new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(10)
    
    # 3. RFM (Si disponible)
    if rfm_stats is not None and not rfm_stats.empty:
        pdf.section_title("3. Segmentation Clients (RFM)")
        segment_counts = rfm_stats['Segment'].value_counts()
        for segment, count in segment_counts.items():
            pct = count / len(rfm_stats) * 100
            pdf.metric_row(f"Segment {segment} :", f"{count} clients ({pct:.1f}%)")
        pdf.ln(10)
        
    # 4. Recommandations
    pdf.section_title("4. Recommandations Expert & IA")
    pdf.set_font("helvetica", "", 10)
    
    top_cat_name = cat_stats['Categorie'].iloc[0] if not cat_stats.empty else 'vos produits phares'
    
    reco_text = (
        f"- Production : Concentrez 80% de votre capacite de production sur la Classe A, en priorite sur la categorie '{top_cat_name}'.\n"
        "- Supply Chain : Le stock de securite doit etre calcule sur la base de la demande prevue et non sur le besoin net.\n"
        "- Pricing : Appliquez des tests d'elasticite-prix (+5%) sur vos produits 'Champions' ayant les meilleures notes clients.\n"
        "- Fidelisation : Lancez des campagnes de reactivation (promotions ou emails cibles) vers vos clients 'A Risque'.\n"
        "- Qualite : Realisez un audit approfondi des composants pour les categories obtenant une note moyenne < 3.5/5."
    )
    pdf.multi_cell(0, 6, reco_text)
    
    return pdf.output()
