"""
data_loader.py — Module de chargement et pré-traitement des données ProdAdvisor.

Ce module centralise toute la logique de chargement, nettoyage et enrichissement
des données pour l'application ProdAdvisor.
"""

import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def load_and_prepare_data(base_dir: str = None) -> pd.DataFrame | None:
    """
    Charge et prépare le jeu de données pour l'application.
    
    Priorité : dataset_clean.csv > dataset.csv
    Gère les deux formats de colonnes (Kaggle et généré).
    
    Args:
        base_dir: Répertoire racine du projet. Si None, utilise le répertoire du fichier.
    
    Returns:
        DataFrame prêt à l'emploi ou None si aucun fichier trouvé.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    clean_path = os.path.join(base_dir, 'data', 'dataset_clean.csv')
    raw_path = os.path.join(base_dir, 'data', 'dataset.csv')
    
    data_path = clean_path if os.path.exists(clean_path) else raw_path
    
    if not os.path.exists(data_path):
        logger.error(f"Aucun fichier de données trouvé dans {os.path.join(base_dir, 'data')}")
        return None
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Données chargées depuis {data_path} : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {data_path}: {e}")
        return None
    
    # --- Adaptation des colonnes Kaggle vers le format interne ---
    column_mapping = {
        'Date Purchase': 'Date',
        'Item Purchased': 'Categorie',
        'Purchase Amount (USD)': 'Prix_Unitaire',
        'Review Rating': 'Avis_Client',
        'Payment Method': 'Methode_Paiement',
        'Customer Reference ID': 'Client_ID'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # --- Traitement des valeurs manquantes ---
    if 'Prix_Unitaire' in df.columns:
        median_price = df['Prix_Unitaire'].median()
        df['Prix_Unitaire'] = df['Prix_Unitaire'].fillna(median_price)
        
    if 'Avis_Client' in df.columns:
        mean_rating = df['Avis_Client'].mean()
        df['Avis_Client'] = df['Avis_Client'].fillna(round(mean_rating, 1))
    
    # --- Ajout des colonnes manquantes ---
    if 'Quantite' not in df.columns:
        df['Quantite'] = 1
    
    np.random.seed(42)  # Reproductibilité
    
    if 'Taille' not in df.columns:
        df['Taille'] = np.random.choice(
            ['XS', 'S', 'M', 'L', 'XL'],
            len(df),
            p=[0.05, 0.2, 0.4, 0.25, 0.1]
        )
    
    if 'Couleur' not in df.columns:
        df['Couleur'] = np.random.choice(
            ['Noir', 'Blanc', 'Bleu', 'Rouge', 'Vert', 'Beige'],
            len(df),
            p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1]
        )
    
    # --- Conversion de la date ---
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
    except Exception:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        except Exception as e:
            logger.error(f"Impossible de convertir les dates: {e}")
            return None
    
    # --- Colonnes dérivées ---
    df['CA'] = df['Quantite'] * df['Prix_Unitaire']
    df['Mois'] = df['Date'].dt.month
    df['Nom_Mois'] = df['Date'].dt.strftime('%B')
    df['Annee'] = df['Date'].dt.year
    df['Trimestre'] = df['Date'].dt.quarter
    df['Jour_Semaine'] = df['Date'].dt.day_name()
    
    # Tri chronologique
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Données prêtes : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def get_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les données par mois pour les tendances et prédictions."""
    df_monthly = df.set_index('Date').resample('ME').agg({
        'Quantite': 'sum',
        'CA': 'sum',
        'Prix_Unitaire': 'mean'
    }).reset_index()
    
    # Exclure le dernier mois s'il est incomplet (ex: arrêté au 1er du mois)
    # pour éviter une chute brutale (à 0) sur les graphiques
    if not df.empty:
        max_date = df['Date'].max()
        if max_date.day < 25:
            df_monthly = df_monthly.iloc[:-1]
            
    return df_monthly


def get_category_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les statistiques clés par catégorie."""
    stats = df.groupby('Categorie').agg(
        Volume=('Quantite', 'sum'),
        CA=('CA', 'sum'),
        Prix_Moyen=('Prix_Unitaire', 'mean'),
        Note_Moyenne=('Avis_Client', 'mean'),
        Nb_Transactions=('Quantite', 'count')
    ).reset_index()
    
    stats['Part_CA'] = (stats['CA'] / stats['CA'].sum() * 100).round(1)
    stats = stats.sort_values('CA', ascending=False).reset_index(drop=True)
    
    # Classification ABC
    stats['CA_Cumul_Pct'] = stats['CA'].cumsum() / stats['CA'].sum() * 100
    stats['Classe_ABC'] = stats['CA_Cumul_Pct'].apply(
        lambda x: 'A' if x <= 80 else ('B' if x <= 95 else 'C')
    )
    
    return stats


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame | None:
    """Calcule les scores RFM (Récence, Fréquence, Montant) par client si Client_ID est présent."""
    if 'Client_ID' not in df.columns or 'Date' not in df.columns or 'CA' not in df.columns:
        return None
        
    # Date de référence = jour de la dernière transaction + 1 jour
    ref_date = df['Date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('Client_ID').agg(
        Recence=('Date', lambda x: (ref_date - x.max()).days),
        Frequence=('Date', 'count'),
        Montant=('CA', 'sum')
    )
    
    # Si le jeu de données a peu de clients uniques par rapport au total (ex: dataset généré sans client_id fixe),
    # les quantiles peuvent échouer s'il y a trop d'ex-aequo. On utilise des bins fixes ou drop_duplicates
    try:
        r_quartiles = pd.qcut(rfm['Recence'].rank(method='first'), q=4, labels=range(4, 0, -1))
        f_quartiles = pd.qcut(rfm['Frequence'].rank(method='first'), q=4, labels=range(1, 5))
        m_quartiles = pd.qcut(rfm['Montant'].rank(method='first'), q=4, labels=range(1, 5))
        
        rfm = rfm.assign(R=r_quartiles.values, F=f_quartiles.values, M=m_quartiles.values)
        
        # Concaténer pour faire un score RFM
        rfm['RFM_Score'] = rfm[['R', 'F', 'M']].apply(lambda x: ''.join(x.dropna().astype(int).astype(str)), axis=1)
        
        # Segmentation experte (Matrice RFM standard)
        def segment_client(score_str):
            if len(score_str) < 3: return 'Inconnu'
            r, f, m = int(score_str[0]), int(score_str[1]), int(score_str[2])
            if r >= 3 and f >= 3 and m >= 3:
                return 'Champions'
            elif r >= 2 and f >= 3:
                return 'Fidèles'
            elif r >= 3 and f <= 2:
                return 'Prometteurs'
            elif r <= 2 and f >= 2:
                return 'À Risque'
            else:
                return 'Perdus (Hibernating)'
                
        rfm['Segment'] = rfm['RFM_Score'].apply(segment_client)
        
    except Exception as e:
        logger.warning(f"Impossible de calculer les quartiles RFM complets : {e}")
        rfm['Segment'] = 'Standard'
        
    return rfm.reset_index()
