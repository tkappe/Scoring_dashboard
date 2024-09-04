import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

# Chargement du modèle et des données
model = pickle.load(open('lgbmt_best_model.pkl', 'rb'))
data_test_sampled = pd.read_csv('data/data_test_sampled.csv', sep=',', index_col=[0], encoding='utf-8')

# Configuration du layout
st.set_page_config(layout="wide")

# Barre latérale fixe avec logo, sélection de l'ID du client, et navigation

def sidebar():
    st.sidebar.image('img/logo.png', use_column_width=True)  # Remplacez "logo.png" par le chemin de votre logo
    st.sidebar.title("Menu de navigation")

    pages = {
        "Accueil": "home",
        "Résultats": "results",
        "Compréhension": "comprehension",
        "Analyses Univariées": "univariate",
        "Analyses Bivariées": "bivariate",
        "Projections": "projections"
    }

    selected_page = st.sidebar.radio("Aller à", list(pages.keys()))
    st.session_state.page = pages[selected_page]

    client_id = st.sidebar.selectbox("Sélectionnez l'ID du client", data_test_sampled.index.tolist(), key="selected_client_id")

    if st.sidebar.button("Soumettre"):
        st.session_state.client_id = client_id
        st.session_state.submitted = True        

# Fonction pour afficher la page d'accueil
def show_home():
    st.title("Dashboard de Prédiction de Scoring")
    st.header("Bienvenue")
    st.write("Bienvenue sur le dashboard de prédiction de scoring.")
    st.write("Ce tableau de bord vous permet de comprendre et d'analyser les risques de défaut de crédit pour les clients. Vous pouvez explorer les résultats de la prédiction, comprendre les facteurs qui influencent les décisions, et effectuer des analyses univariées et bivariées.")
    st.write("L'objectif est de fournir une explication transparente des prédictions et d'explorer les impacts possibles de la modification de certaines caractéristiques.")

# Fonction pour afficher les résultats de la prédiction
def show_results():
    client_id = st.session_state.client_id
    
    st.title(f"Prédiction du risque de défaut pour le client {client_id}")
    
    if client_id not in data_test_sampled.index:
        st.error(f"Client ID {client_id} non trouvé dans la base de données.")
        return

    client_data = data_test_sampled.loc[[client_id]]
    probability = model.predict_proba(client_data)[:, 1][0]
    threshold = 0.133
    predicted_class = int(probability > threshold)

    st.subheader(f"Résultat de la demande")
    st.write(f"Probabilité de défaut : {probability:.2%}")
    st.write(f"Classe : {'Défaillant' if predicted_class == 1 else 'Non défaillant'}")
    st.write(f"Décision : {'Prêt rejeté' if predicted_class == 1 else 'Prêt accepté'}")

    st.subheader("Positionnement du client par rapport au seuil")
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh([0], [threshold], color='green', height=0.3)  # Zone avant le seuil (verte)
    ax.barh([0], [1-threshold], left=threshold, color='red', height=0.3)  # Zone après le seuil (rouge)
    ax.plot(probability, 0, marker='v', color='black', markersize=15)  # Flèche pour la probabilité du client
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xticklabels([f'{int(x*100)}%' for x in np.linspace(0, 1, 11)])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axvline(threshold, color='blue', linestyle='--', label='Seuil')
    ax.legend(loc='upper left')
    st.pyplot(fig)

    st.subheader("Importance globale des features")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_test_sampled)
    
    mean_shap_values = np.abs(shap_values[1]).mean(axis=0)
    top_10_features_idx = np.argsort(mean_shap_values)[-10:]
    top_10_feature_names = data_test_sampled.columns[top_10_features_idx]
    top_10_shap_values = shap_values[1][:, top_10_features_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(top_10_shap_values, features=data_test_sampled[top_10_feature_names], plot_type="bar", show=False)
    st.pyplot(fig)

# Fonction pour afficher la compréhension de la prédiction
def show_comprehension():
    client_id = st.session_state.client_id
    
    st.title(f"Compréhension de la prédiction du client {client_id}")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_test_sampled)
    
    st.subheader("Importance locale des features")
    client_shap_values = shap_values[1][data_test_sampled.index.get_loc(client_id)]
    
    top_positive_impact = np.argsort(-client_shap_values)[:5]
    top_negative_impact = np.argsort(client_shap_values)[:5]
    
    st.write("**5 features les plus impactantes positivement :**")
    st.write(data_test_sampled.columns[top_positive_impact])
    
    st.write("**5 features les plus impactantes négativement :**")
    st.write(data_test_sampled.columns[top_negative_impact])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.force_plot(explainer.expected_value[1], client_shap_values, data_test_sampled.loc[[client_id]], matplotlib=True)
    st.pyplot(fig)

# Fonction pour afficher les analyses univariées
def show_univariate_analysis():
    client_id = st.session_state.client_id
    
    st.title(f"Analyses univariées et positionnement du client {client_id}")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_test_sampled)
    
    client_shap_values = shap_values[1][data_test_sampled.index.get_loc(client_id)]
    abs_shap_values = np.abs(client_shap_values)
    top_6_features_idx = np.argsort(abs_shap_values)[-6:]
    top_6_feature_names = data_test_sampled.columns[top_6_features_idx]
    
    for feature in top_6_feature_names:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(data_test_sampled[feature], bins=30, alpha=0.5, color='gray')
        ax.axvline(data_test_sampled.loc[client_id, feature], color='red', linewidth=2, label=f'Client {client_id}')
        ax.set_title(f'{feature}')
        ax.legend()
        st.pyplot(fig)

# Fonction pour afficher les analyses bivariées
def show_bivariate_analysis():
    client_id = st.session_state.client_id
    
    st.title(f"Analyses bivariées et positionnement du client {client_id}")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_test_sampled)
    
    client_shap_values = shap_values[1][data_test_sampled.index.get_loc(client_id)]
    abs_shap_values = np.abs(client_shap_values)
    top_10_features_idx = np.argsort(abs_shap_values)[-10:]
    top_10_feature_names = data_test_sampled.columns[top_10_features_idx]
    
    feature1 = st.selectbox("Sélectionnez la première feature", top_10_feature_names, key="feature1")
    feature2 = st.selectbox("Sélectionnez la seconde feature", top_10_feature_names, key="feature2")

    if feature1 and feature2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data_test_sampled[feature1], data_test_sampled[feature2], alpha=0.5)
        ax.scatter(data_test_sampled.loc[[client_id]][feature1], data_test_sampled.loc[[client_id]][feature2], color='red', label="Client")
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.legend()
        st.pyplot(fig)

# Fonction pour afficher les projections
def show_projections():
    client_id = st.session_state.client_id
    
    st.title(f"Modifications projections et impacts pour le client {client_id}")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_test_sampled)
    
    client_shap_values = shap_values[1][data_test_sampled.index.get_loc(client_id)]
    top_negative_impact = np.argsort(client_shap_values)[:5]
    top_negative_features = data_test_sampled.columns[top_negative_impact]
    
    new_client_data = data_test_sampled.loc[[client_id]].copy()
    
    st.subheader("Modifier les valeurs des features les plus impactantes négativement :")
    for feature in top_negative_features:
        new_value = st.number_input(f"{feature}", value=new_client_data[feature].values[0])
        new_client_data[feature] = new_value
    
    new_probability = model.predict_proba(new_client_data)[:, 1][0]
    new_class = int(new_probability > 0.133)
    
    st.write(f"**Nouvelle probabilité de défaut :** {new_probability:.2%}")
    st.write(f"**Nouvelle décision :** {'Prêt rejeté' if new_class == 1 else 'Prêt accepté'}")

# Fonction principale pour gérer la navigation entre les pages
def main():
    sidebar()

    if 'submitted' in st.session_state and st.session_state.submitted:
        st.session_state.submitted = False  # Réinitialiser l'indicateur après la soumission

    if st.session_state.page == "home":
        show_home()
    elif st.session_state.page == "results":
        show_results()
    elif st.session_state.page == "comprehension":
        show_comprehension()
    elif st.session_state.page == "univariate":
        show_univariate_analysis()
    elif st.session_state.page == "bivariate":
        show_bivariate_analysis()
    elif st.session_state.page == "projections":
        show_projections()        

if __name__ == "__main__":
    if 'client_id' not in st.session_state:
        st.session_state.client_id = data_test_sampled.index[0]
    
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    
    main()

    