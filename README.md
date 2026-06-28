# ProdAdvisor — Optimisation Textile IA

ProdAdvisor est une application d'aide à la décision propulsée par l'Intelligence Artificielle, conçue spécifiquement pour la planification de la production textile et la supply chain. Grâce à un tableau de bord interactif (Dashboard), la solution combine l'analyse exploratoire de données, la simulation, la prévision temporelle et l'assistance conversationnelle pour optimiser les marges et réduire le sur-stockage.

## ✨ Fonctionnalités Principales

### 1. Tableau de bord & Analyse Exploratoire (EDA)
- Suivi en temps réel des KPIs : Chiffre d'Affaires, Panier moyen, Fidélité client (clients VIP).
- Graphique dynamique interactif pour l'**Évolution mensuelle** des ventes (historique et prévisions fusionnées).
- Répartition des ventes et détails sur le "Top 5" des produits.

### 2. Intelligence Artificielle & Simulations Avancées
- **Simulateur d'élasticité prix** : Évalue l'impact direct d'une variation de prix de vente sur les volumes estimés et le chiffre d'affaires projeté.
- **Machine Learning (Prédiction de Satisfaction)** : Modèle ML (XGBoost/Random Forest fine-tuné) qui apprend de l'historique pour prédire la note (sur 5) qu'obtiendra une nouvelle combinaison produit (Catégorie, Taille, Couleur, Prix cible).
- **Prévisions Temporelles (Time Series)** : Modèle de prévision (TimeGPT ou Prophet) qui analyse l'historique pour projeter la demande future sur les prochains mois.

### 3. Recommandations de Production
- Système d'objectif de stock (Demande estimée + Marge de sécurité ajustable) face au stock actuel.
- Calcul de la quantité nette à produire avec suggestion d'un plan de répartition par catégorie.

### 4. Fidélisation & Rentabilité
- **Analyse RFM (Récence, Fréquence, Montant)** : Segmentation de la clientèle pour identifier les clients "VIP", "Fidèles", etc. Cartographie visuelle des segments.
- **Loi de Pareto (Classement ABC)** : Analyse de la rentabilité pour repérer les produits moteurs (A) contre les produits à rotation lente (C).

### 5. ProdBot (Conseiller IA)
- Assistant conversationnel directement intégré sous forme d'onglet ou via un bouton flottant rapide (FAB).
- Capable de répondre aux questions sur vos données, les tendances et de proposer des recommandations basées sur l'état du stock et les modèles prédictifs.

### 6. Exportation Exécutive
- Génération d'un rapport exécutif synthétique en **PDF**, téléchargeable d'un simple clic depuis la barre latérale pour la communication interne et la direction.

## 🛠️ Technologies Utilisées

- **Frontend / Interface** : [Streamlit](https://streamlit.io/) et design personnalisé en CSS (Glassmorphism).
- **Visualisations** : [Altair](https://altair-viz.github.io/).
- **Data Science & ML** : `pandas`, `numpy`, `scikit-learn` / XGBoost.
- **Forecasting** : `Prophet`, `Nixtla (TimeGPT)`.
- **Génération de documents** : Outils de génération de rapports (PDF).

## 🚀 Installation & Exécution

1. **Cloner le projet** ou s'assurer d'être dans le dossier principal.
2. **Créer un environnement virtuel** (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sous Windows : venv\Scripts\activate
   ```
3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
4. **Configurer l'environnement** :
   Créez un fichier `.env` à la racine (ou mettez-y à jour les clés existantes) pour l'API TimeGPT si vous l'utilisez :
   ```env
   TIMEGPT_TOKEN=votre_clé_api_ici
   ```
5. **Lancer l'application** :
   ```bash
   python -m streamlit run app.py
   ```
6. **L'application sera accessible sur** `http://localhost:8501`

## 📂 Structure du projet (Aperçu)

- `app.py` : Point d'entrée principal de l'application Streamlit.
- `src/` : Modules fonctionnels de l'application.
  - `data_loader.py` : Préparation des données, calcul du RFM et catégorisations.
  - `forecasting.py` : Algorithmes prédictifs pour l'évolution mensuelle.
  - `chatbot_engine.py` : Cœur de l'assistant ProdBot.
  - `ml_models.py` : Modèles de prédiction de satisfaction.
  - `pdf_generator.py` : Exportateur de rapports PDF.
- `data/` : Dossier prévu pour le jeu de données local (ex: `dataset.csv`).
- `.env` : Variables d'environnement.

---
*Auteurs : BENHAYOUN Abderrahmane, ELKARI Wafae, SIRAJI Souha, MOUNCIR Hiba*
