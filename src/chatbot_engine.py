"""
chatbot_engine.py — Moteur de chatbot contextuel ProdBot.

Ce module gère les réponses intelligentes du chatbot en utilisant l'API Mistral
et en fournissant le contexte du dataset en temps réel.
"""

import pandas as pd
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv

try:
    from mistralai.client import Mistral
except ImportError:
    try:
        from mistralai import Mistral
    except ImportError:
        Mistral = None

class ProdBotEngine:
    """
    Moteur de réponses contextuelles pour ProdBot utilisant Mistral AI.
    """
    
    def __init__(self, df: pd.DataFrame, prediction: Optional[int] = None, cat_stats: pd.DataFrame = None, rfm_stats: pd.DataFrame = None):
        self.df = df
        self.prediction = prediction
        self.cat_stats = cat_stats
        self.rfm_stats = rfm_stats
        load_dotenv()
        self.api_key = os.getenv("MISTRAL_API_KEY", "")
        self.client = None
        
        if Mistral:
            try:
                self.client = Mistral(api_key=self.api_key)
            except Exception as e:
                print(f"Erreur d'initialisation Mistral: {e}")
                
        self._compute_stats()
        self.system_context = self._generate_system_context()
        
    def _compute_stats(self):
        """Pré-calcule les statistiques pour donner du contexte au modèle."""
        self.total_ca = (self.df['Quantite'] * self.df['Prix_Unitaire']).sum() if len(self.df) > 0 else 0
        self.total_ventes = self.df['Quantite'].sum() if len(self.df) > 0 else 0
        
        cat_sales = self.df.groupby('Categorie')['Quantite'].sum()
        self.top_cat = cat_sales.idxmax() if not cat_sales.empty else 'N/A'
        self.top_cat_qty = cat_sales.max() if not cat_sales.empty else 0
        
        self.avg_rating = self.df['Avis_Client'].mean() if 'Avis_Client' in self.df.columns else 0
        if len(self.df) > 0:
            if 'Client_ID' in self.df.columns:
                nb_commandes = self.df.groupby(['Client_ID', self.df['Date'].dt.date]).ngroups
                self.panier_moyen = self.total_ca / nb_commandes if nb_commandes > 0 else 0
            else:
                self.panier_moyen = self.total_ca / len(self.df)
        else:
            self.panier_moyen = 0
        
        self.nb_mois = self.df['Date'].dt.to_period('M').nunique() if 'Date' in self.df.columns else 1
        self.ventes_mensuelles_moy = self.total_ventes / self.nb_mois if self.nb_mois > 0 else 0
        
        # Classement ABC
        if len(self.df) > 0:
            ca_by_cat = (self.df.groupby('Categorie')
                         .apply(lambda x: (x['Quantite'] * x['Prix_Unitaire']).sum())
                         .sort_values(ascending=False))
            self.top3_ca = list(ca_by_cat.head(3).index)
        else:
            self.top3_ca = []

    def _generate_system_context(self) -> str:
        """Génère le prompt système contenant le résumé financier de l'entreprise."""
        cat_details = ""
        if self.cat_stats is not None and not self.cat_stats.empty:
            cat_details = "\n- Détail des Top 5 Catégories :"
            for _, row in self.cat_stats.head(5).iterrows():
                part_ca = row['Part_CA'] if 'Part_CA' in row else 0
                cat_details += f"\n  * {row['Categorie']} : {row['CA']:,.0f} $ (Part: {part_ca}%)"

        rfm_details = ""
        if self.rfm_stats is not None and not self.rfm_stats.empty:
            segments = self.rfm_stats['Segment'].value_counts()
            rfm_details = "\n- Segmentation Clients (RFM) :"
            for seg, count in segments.items():
                rfm_details += f"\n  * {seg} : {count} clients"

        return f"""Tu es ProdBot, un assistant virtuel expert en supply chain, production textile, et analyse de ventes. Tu es intégré à l'ERP ProdAdvisor. 
Tu dois aider le directeur de l'entreprise en répondant à ses questions de façon professionnelle, concise et directement applicable.
Utilise des tableaux Markdown, du texte en gras, et des listes à puces pour rendre tes réponses très visuelles.
NE METS PAS D'EMOJIS dans tes réponses. Garde un ton exécutif et formel.

CONTEXTE FINANCIER ACTUEL DE L'ENTREPRISE (basé sur le tableau de bord) :
- Chiffre d'Affaires Total : {self.total_ca:,.0f} $
- Volume Total Vendu : {self.total_ventes:,} pièces
- Panier moyen : {self.panier_moyen:,.0f} $
- Ventes mensuelles moyennes : {self.ventes_mensuelles_moy:,.0f} pièces
- Période analysée : {self.nb_mois} mois
- Produit Leader en volume : {self.top_cat} ({self.top_cat_qty:,} pièces)
- Top 3 catégories rentables (Classe A) : {", ".join(self.top3_ca)}
- Note client moyenne globale : {self.avg_rating:.2f}/5
- Prédiction de production pour le mois prochain : {self.prediction if self.prediction else 'Non disponible'} pièces.{cat_details}{rfm_details}

Règle stricte: N'invente pas de chiffres hors de ce contexte si on te demande un total ou une moyenne globale. Si on te demande quelque chose de spécifique que tu n'as pas dans ce contexte, utilise ton expertise métier pour conseiller la méthode d'analyse.
Réponds toujours en Français."""

    def generate_response(self, prompt: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Génère une réponse via l'API Mistral en tenant compte de l'historique.
        """
        if not self.client:
            return "L'intégration Mistral AI n'est pas disponible (bibliothèque non installée ou erreur d'initialisation). Lancez `pip install mistralai`."
            
        messages = [{"role": "system", "content": self.system_context}]
        
        if chat_history:
            # On garde les 5 derniers échanges pour éviter d'exploser le token count
            for msg in chat_history[-6:]: 
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                    
        # On s'assure que le dernier message est bien le prompt de l'utilisateur
        if not messages or messages[-1]["content"] != prompt:
            messages.append({"role": "user", "content": prompt})

        try:
            chat_response = self.client.chat.complete(
                model="mistral-large-latest",
                messages=messages,
                temperature=0.2,
                max_tokens=800
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            return f"**Erreur de communication avec l'API Mistral :**\n```\n{e}\n```\nVeuillez vérifier que vous disposez d'une connexion internet."

    def generate_response_stream(self, prompt: str, chat_history: List[Dict[str, str]] = None):
        """
        Génère une réponse via l'API Mistral en mode streaming.
        """
        if not self.client:
            yield "L'intégration Mistral AI n'est pas disponible (bibliothèque non installée ou erreur d'initialisation). Lancez `pip install mistralai`."
            return
            
        messages = [{"role": "system", "content": self.system_context}]
        
        if chat_history:
            for msg in chat_history[-6:]: 
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                    
        if not messages or messages[-1]["content"] != prompt:
            messages.append({"role": "user", "content": prompt})

        try:
            response_stream = self.client.chat.stream(
                model="mistral-large-latest",
                messages=messages,
                temperature=0.2,
                max_tokens=800
            )
            for chunk in response_stream:
                if chunk.data.choices[0].delta.content is not None:
                    yield chunk.data.choices[0].delta.content
        except Exception as e:
            yield f"\n**Erreur de communication avec l'API Mistral :**\n```\n{e}\n```\nVeuillez vérifier que vous disposez d'une connexion internet."
