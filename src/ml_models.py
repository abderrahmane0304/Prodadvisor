import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st

try:
    from xgboost import XGBRegressor
except ImportError:
    from sklearn.ensemble import RandomForestRegressor as XGBRegressor

@st.cache_resource(show_spinner=False)
def train_satisfaction_model(df: pd.DataFrame):
    """
    Entraîne un modèle XGBoost pour prédire la note client (Avis_Client) avec fine-tuning.
    Retourne le modèle entraîné, les métriques (MAE, R2) et les importances des features.
    """
    if 'Avis_Client' not in df.columns or df['Avis_Client'].isnull().all():
        return None, None, None
        
    # Variables prédictives et cible
    target = 'Avis_Client'
    features = ['Categorie', 'Taille', 'Couleur', 'Prix_Unitaire']
    
    # Nettoyage
    model_df = df[features + [target]].dropna()
    if len(model_df) < 50:
        return None, None, None
        
    X = model_df[features]
    y = model_df[target]
    
    # Séparation Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Préparation des pipelines
    numeric_features = ['Prix_Unitaire']
    categorical_features = ['Categorie', 'Taille', 'Couleur']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
    # Vérification si on utilise bien XGBoost ou le repli Random Forest
    is_xgb = XGBRegressor.__name__ == 'XGBRegressor'
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(random_state=42))
    ])
    
    # Fine-Tuning Hyperparamètres avec RandomizedSearchCV
    if is_xgb:
        param_distributions = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 7, 10],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0]
        }
    else:
        param_distributions = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, None]
        }
        
    # Fine-Tuning étendu (n_iter=10) pour une meilleure performance
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )
    
    # Entraînement
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Évaluation
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Récupération des importances de features
    try:
        cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat']
        cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
        all_features = numeric_features + cat_features
        importances = best_model.named_steps['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(10)
    except Exception:
        feature_importance_df = pd.DataFrame()
        
    metrics = {'MAE': mae, 'R2': r2, 'best_params': search.best_params_}
    
    return best_model, metrics, feature_importance_df

def predict_satisfaction(model, categorie: str, taille: str, couleur: str, prix: float) -> float:
    """
    Prédit la satisfaction pour un produit spécifique en utilisant le modèle entraîné.
    """
    if model is None:
        return 0.0
        
    input_data = pd.DataFrame({
        'Categorie': [categorie],
        'Taille': [taille],
        'Couleur': [couleur],
        'Prix_Unitaire': [prix]
    })
    
    return model.predict(input_data)[0]
