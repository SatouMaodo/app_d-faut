import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from PIL import Image
import plotly.graph_objects as go
import mlflow

def impute_and_standardize(X):
    """Impute les valeurs manquantes et standardise les données.

    Args:
        X: DataFrame contenant les données à imputer et standardiser.

    Returns:
        DataFrame avec les valeurs manquantes imputées et les données standardisées.
    """

    # Créer une copie de X pour éviter de modifier les données d'origine
    X_imputed = X.copy()

    # Replace infinite values with NaNs
    X_imputed = X_imputed.replace([np.inf, -np.inf], np.nan)

    # Imputer les valeurs manquantes avec la médiane pour les variables numériques
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_imputed)

    # Standardiser les données numériques
    scaler = StandardScaler()
    X_scaled= scaler.fit_transform(X_imputed)

    return X_scaled
# Chargement du modèle et des données
best_model = joblib.load('best_model.joblib')

test_df= pd.read_csv('test_df.csv')
y = test_df['TARGET']
X = test_df.drop(columns=['TARGET'])


from sklearn.pipeline import Pipeline
# Créer la pipeline
pipeline = Pipeline([
    ('imputation_standardisation', FunctionTransformer(impute_and_standardize, validate=False)),  # Encapsuler dans FunctionTransformer
    ('model', best_model)
])

# --- Variables les plus influentes ---
influential_features = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'PAYMENT_RATE']
# Nombre d'acceptés et de refusés
# Prédictions du modèle
#predict = best_model.predict(X)
#predict = np.array(predict)
#num_accepted = sum(predict < 0.76)
#num_refused = sum(predict >=0.76)

# Charger le logo
logo_image = Image.open("logo.png")
# Afficher le logo en haut à gauche (une seule fois)
st.sidebar.image(logo_image, use_container_width=True)
#st.sidebar.markdown(f"**Nombre d'acceptés :** {num_accepted}")
#st.sidebar.markdown(f"**Nombre de refusés :** {num_refused}")

# Afficher le texte sous le logo (horizontalement)
st.sidebar.markdown(
    """
    <style>
    .wordart-text {
        font-family: 'Arial Black', sans-serif;
        font-size: 12px;
        font-weight: bold;
        color: #800080; /* Violet */

        margin-top: 200px; /* Ajouter une marge supérieure */
    }
    .smaller-text {
        font-size: 12px;
        color: black;
        margin-top:10 px; /* Ajouter une marge supérieure */
    }
    </style>
    <div class="wordart-text">Etudiante: Amsatou NDIAYE - Parcours Data Science </div>
    <div class="smaller-text">Titre du Projet: Implémenter un modèle de scoring-Openclassrooms 2025 </div>
    """,
    unsafe_allow_html=True,
)
# --- Mise en forme de l'application ---
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background-image: linear-gradient(to bottom right, #e0ffff, #cce0ff); 
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Liste des identifiants uniques des demandeurs
sk_id_curr_list = test_df['SK_ID_CURR'].unique()

# --- Affichage ---
st.markdown("## Prédiction de la probabilité de défaut")
decision_threshold = st.slider("Seuil de décision", min_value=0.0, max_value=1.0, value=0.76, step=0.01)

# --- Sélection de l'identité du demandeur ---
selected_sk_id_curr = st.selectbox("Sélectionnez l'identité du demandeur (SK_ID_CURR)", sk_id_curr_list)

# --- Prédiction et importance des variables ---
if selected_sk_id_curr :
    # Filtrer les données pour l'identité sélectionnée
    selected_data = test_df[test_df['SK_ID_CURR'] == selected_sk_id_curr ]

    # --- Informations sur les variables influentes ---
    st.markdown("### Informations sur les variables influentes")

    # --- Informations personnelles ---
    st.markdown("#### Informations personnelles")
    personal_info = selected_data[['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED']].T.rename(
        columns={selected_data.index[0]: 'Valeur'})
    personal_info.index = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED']  # Renommer les index
    st.table(personal_info)

    # --- Informations financières ---
    st.markdown("#### Informations financières")
    financial_info = selected_data[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'PAYMENT_RATE']].T.rename(
        columns={selected_data.index[0]: 'Valeur'})
    financial_info.index = ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'PAYMENT_RATE']  # Renommer les index
    st.table(financial_info)

    # Supprimer la colonne 'TARGET' si elle est présente
    if 'TARGET' in selected_data.columns:
        selected_data = selected_data.drop(columns=['TARGET'])

    # Prédiction
    prediction = best_model.predict_proba(selected_data)[0, 1]
    st.markdown(f"**Probabilité de défaut:** {prediction:.2f}")

    # Score de crédit et décision
    credit_score = int((1 - prediction) * 100)
    decision = "Accepté" if prediction < decision_threshold else "Refusé"

    # Afficher la décision avec la couleur appropriée
    if decision == "Accepté":
        st.success(f"Décision: {decision} (Score: {credit_score})")
    else:
        st.error(f"Décision: {decision} (Score: {credit_score})")

    # Afficher la décision avec la couleur appropriée
    #if decision == "Accepté":
       # st.markdown(f"<span style='color:green'>**Décision : {decision}**</span>", unsafe_allow_html=True)
    #else:
     #   st.markdown(f"<span style='color:red'>**Décision : {decision}**</span>", unsafe_allow_html=True)

    st.markdown(f"**Score de crédit :** {credit_score}")
    # Jauge pour le score de crédit

    # Jauge pour le score de crédit (avec st.progress)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=credit_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score de crédit"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 76], 'color': "orange"},
                {'range': [76, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': credit_score  # Remplacer 700 par credit_score
            }
        }
    ))

    st.plotly_chart(fig)

    # --- Décision Accepté/Refusé ---

    # Feature importance locale avec shap.plots.waterfall
    feature_names = X.columns
    explainer = shap.TreeExplainer(best_model)
    # Get SHAP values as an Explanation object
    shap_values_explanation = explainer(selected_data)
    # Waterfall plot
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_values_explanation[0], show=False)
    plt.title("Importance des variables locales (Waterfall Plot)", fontsize=16)
    plt.xlabel("Impact sur la prédiction", fontsize=12)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    # --- Comparaison avec les autres clients ---
    st.markdown("### Comparaison avec les autres clients")

    # Sélectionner les features à comparer
    features_to_compare = st.multiselect("Sélectionnez les features à comparer", selected_data.columns)

    if features_to_compare:
        # Calculer les statistiques descriptives pour les features sélectionnées
        client_data = selected_data[features_to_compare].iloc[0]  # Données du client sélectionné
        other_clients_data = test_df[features_to_compare]  # Données des autres clients

        # Créer un DataFrame pour la comparaison
        comparison_df = pd.DataFrame({
            "Client sélectionné": client_data,
            "Moyenne des autres clients": other_clients_data.mean(),
            "Écart-type des autres clients": other_clients_data.std()
        })

        # Afficher le DataFrame de comparaison
        st.table(comparison_df)

        # Créer des graphiques pour la comparaison
        for feature in features_to_compare:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=other_clients_data[feature], name="Autres clients"))
            fig.add_trace(go.Scatter(x=[client_data[feature]], y=[0], mode="markers",
                                     marker=dict(size=10, color="red"), name="Client sélectionné"))
            fig.update_layout(title=f"Distribution de {feature}")
            st.plotly_chart(fig)

