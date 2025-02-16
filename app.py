import streamlit as st
import requests
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from PIL import Image

# === Définition de l'URL de l'API ===
API_URL = "https://fastapi-appli-88ed9063cc28.herokuapp.com"

# === Chargement des données test ===
test_df = pd.read_csv('test.csv', sep=';')

# Chargement du logo
logo_image = Image.open("logo.png")
st.sidebar.image(logo_image, use_container_width=True)

# Informations sur l'étudiante et le projet
st.sidebar.markdown(
    """
    <style>
    .wordart-text {
        font-family: 'Arial Black', sans-serif;
        font-size: 12px;
        font-weight: bold;
        color: #800080;
        margin-top: 200px;
    }
    .smaller-text {
        font-size: 12px;
        color: black;
        margin-top:10px;
    }
    </style>
    <div class="wordart-text">Étudiante: Amsatou NDIAYE - Parcours Data Science </div>
    <div class="smaller-text">Titre du Projet: Implémenter un modèle de scoring - Openclassrooms 2025 </div>
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

# === Interface principale ===
st.markdown("## Prédiction de la probabilité de défaut")

# Seuil de décision
decision_threshold = 0.76

# Sélection de l'identité du demandeur
sk_id_curr_list = test_df['SK_ID_CURR'].unique()
selected_sk_id_curr = st.selectbox("Sélectionnez un demandeur", sk_id_curr_list)

if selected_sk_id_curr:
    if st.button("Prédire"):
        try:
            response = requests.post(f"{API_URL}/predict", json={"client_id": int(selected_sk_id_curr)}, verify=False)

            if response.status_code == 200:
                prediction = response.json()["prediction"]
                credit_score = int((1 - prediction) * 100)

                st.markdown(f"**Probabilité de défaut:** {prediction:.2f}")
                st.markdown(f"**Score de crédit:** {credit_score:.2f}")

                # Interprétation du score
                st.markdown("### Interprétation du score")
                st.write(
                    """
                    - **Le score est calculé comme suit :** (1 - Probabilité de défaut) * 100.
                    - **Un score proche de 100** indique un **faible risque** de défaut.
                    - **Un score proche de 0** indique un **risque élevé** de défaut.
                    - **Seuil de probabilité critique :** Une probabilité de défaut supérieure à **0.76** est considérée comme un défaut.
                    - **Score critique=**(1-0.76)*100= **24%**. Un score inférieur ou égal à 24% indique un défaut.
                    """
                )

                # Affichage de la jauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=credit_score,
                    title={'text': "Score de crédit"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "grey"},
                        'steps': [
                            {'range': [0, 24], 'color': "red"},
                            {'range': [25, 50], 'color': "orange"},
                            {'range': [51, 76], 'color': "yellow"},
                            {'range': [77, 100], 'color': "green"}
                        ]
                    }
                ))
                st.plotly_chart(fig)

                # Affichage du résultat avec couleur
                if credit_score <= 24:
                    st.markdown("<h4 style='color:red;'>Risque de défaut du client élevé</h4>", unsafe_allow_html=True)
                elif 24 < credit_score <= 50:
                    st.markdown("<h4 style='color:balck;'>Risque modéré : Analyse complémentaire nécessaire.Cela signifie que des informations supplémentaires doivent être collectées et analysées, comme des documents financiers ou des justificatifs, avant de prendre une décision</h4>",
                                unsafe_allow_html=True)
                elif 51 <= credit_score <= 76:
                    st.markdown("<h4 style='color:black;'>Risque intermédiaire : Revue attentive ( La situation semble globalement acceptable, mais une vérification plus approfondie du dossier est recommandée avant d'approuver le crédit)</h4>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("<h4 style='color:green;'>Faible risque de défaut</h4>", unsafe_allow_html=True)

                # Décision finale
                decision = "Accepté" if prediction < decision_threshold else "Refusé"
                st.markdown(f"### Décision : {decision}")
            else:
                st.error(f"Erreur de l'API: {response.status_code}")
        except Exception as e:
            st.error(f"Erreur lors de la récupération des données : {str(e)}")

    # === Requête à l'API pour obtenir les valeurs SHAP ===
    if st.button("Interprétatbilité locale"):
        try:
            # Ajout de 'verify=False' pour ignorer l'erreur SSL
            response = requests.post(f"{API_URL}/interpretabilite_locale", json={"client_id": int(selected_sk_id_curr)}, verify=False)

            if response.status_code == 200:
                shap_data = response.json()
                shap_values = shap_data["shap_values"]
                base_values = shap_data["base_values"]

                # Conversion des listes en tableaux NumPy si nécessaire
                if isinstance(shap_values, list):
                    shap_values = np.array(shap_values)

                if isinstance(base_values, list):
                    base_values = np.array(base_values)

                st.markdown("### Explication de la décision")
                plt.figure(figsize=(12, 6))
                shap.waterfall_plot(
                    shap.Explanation(values=shap_values, base_values=base_values, feature_names=test_df.columns)
                )
                st.pyplot(plt.gcf())
            else:
                st.error("Erreur lors de la récupération des valeurs SHAP")
        except requests.exceptions.SSLError as e:
            st.error(f"Erreur SSL lors de la récupération des valeurs SHAP: survient lorsqu'il y a un problème avec la connexion sécurisée entre ton application et un serveur via le protocole HTTPS: {e}")

    # === Comparaison avec les autres clients ===
    st.markdown("### Comparaison avec les autres clients")
    features_to_compare = st.multiselect("Sélectionnez les features à comparer", test_df.columns)

    if features_to_compare:
        client_data = test_df[test_df['SK_ID_CURR'] == selected_sk_id_curr][features_to_compare].iloc[0]
        other_clients_data = test_df[features_to_compare]

        comparison_df = pd.DataFrame({
            "Client sélectionné": client_data,
            "Moyenne des autres clients": other_clients_data.mean(),
            "Écart-type des autres clients": other_clients_data.std()
        })
        st.table(comparison_df)

        for feature in features_to_compare:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=other_clients_data[feature], name="Autres clients"))
            fig.add_trace(go.Scatter(x=[client_data[feature]], y=[0], mode="markers",
                                     marker=dict(size=10, color="red"), name="Client sélectionné"))
            fig.update_layout(title=f"Distribution de {feature}")
            st.plotly_chart(fig)
