import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier

# Chargement du modèle et des données
model = pickle.load(open('lgbmt_best_model.pkl', 'rb'))
data_test_sampled = pd.read_csv('data/data_test_sampled.csv', sep=',', index_col=[0], encoding='utf-8')

# Configuration du layout
st.set_page_config(layout="wide")

# Barre latérale fixe avec logo, sélection de l'ID du client, et navigation

def sidebar():
    st.sidebar.image('img/logo.png', use_column_width=True) 
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
    st.header("Bienvenue sur le dashboard de prédiction de scoring.")
    st.write("Ce tableau de bord vous permet de comprendre et d'analyser les risques de défaut de crédit pour les clients. Vous pouvez explorer les résultats de la prédiction, comprendre les facteurs qui influencent les décisions, et effectuer des analyses univariées et bivariées.")
    st.write("L'objectif est de fournir une explication transparente des prédictions et d'explorer les impacts possibles de la modification de certaines caractéristiques. Améliorer sans cesse notre service est une priorité absolue!")

# Fonction pour afficher les résultats de la prédiction
def show_results():
    client_id = st.session_state.client_id

    st.title(f"Prédiction du risque de défaut pour le client {client_id}")

    # Appeler l'API pour obtenir la prédiction
    api_url = "https://scoringappp7-8f781cb475fc.herokuapp.com/predict2"  # Remplacez par l'URL réelle de votre API
    response = requests.get(api_url, params={"client_id": client_id})

    # Vérifiez si la réponse est réussie (code 200)
    if response.status_code != 200:
        st.error(f"Erreur lors de la récupération des prédictions. Statut: {response.status_code}")
        st.write("Détails de l'erreur :", response.text)  # Affichez le texte de la réponse pour le diagnostic
        return
    
    try:
        # Vérifier si la réponse est en JSON
        if response.headers.get('Content-Type') == 'application/json':
            prediction_result = response.json()
        else:
            st.error("La réponse de l'API n'est pas au format JSON.")
            st.write("Contenu brut de la réponse :", response.text)
            return
    except ValueError as e:
        st.error("Erreur lors du décodage de la réponse JSON.")
        st.write("Contenu brut de la réponse :", response.text)
        return
    
    # Si tout va bien, continuez avec l'affichage des résultats
    st.subheader(f"Résultat de la demande")
    st.write(f"Probabilité de défaut : {prediction_result['probability']:.2%}")
    st.write(f"Décision : {'Prêt rejeté' if prediction_result['score'] == 1 else 'Prêt accepté'}")

    st.subheader("Positionnement du client par rapport au seuil")
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh([0], [0.133], color='green', height=0.3)  # Zone avant le seuil (verte)
    ax.barh([0], [1-0.133], left=0.133, color='red', height=0.3)  # Zone après le seuil (rouge)
    ax.plot(prediction_result['probability'], 0, marker='v', color='black', markersize=15)  # Flèche pour la probabilité du client
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xticklabels([f'{int(x*100)}%' for x in np.linspace(0, 1, 11)])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axvline(0.133, color='blue', linestyle='--', label='Seuil')
    ax.legend(loc='upper left')
    st.pyplot(fig)

    st.subheader("Importance Globale des Features")

    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[-10:]  # Les 10 plus importantes
    sorted_importances = feature_importance[sorted_idx]
    sorted_features = data_test_sampled.columns[sorted_idx]

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("coolwarm", len(sorted_features))
    plt.barh(sorted_features, sorted_importances, color=colors)
    plt.title("Feature Importance Globale")
    st.pyplot(plt)

# Fonction pour afficher la compréhension de la prédiction
def show_comprehension():
    st.title(f"Compréhension de la Prédiction pour le Client {st.session_state.client_id}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_test_sampled)

    client_index = data_test_sampled.index.get_loc(st.session_state.client_id)
    client_shap_values = shap_values[1][client_index]
    
    # Séparer les valeurs positives et négatives
    positive_idx = np.argsort(client_shap_values)[-5:]  # Top 5 positives
    negative_idx = np.argsort(client_shap_values)[:5]   # Top 5 negatives

    plt.figure(figsize=(10, 6))
    # Dégradé pour les positives
    colors_positive = sns.color_palette("Blues", len(positive_idx))
    plt.barh(data_test_sampled.columns[positive_idx], client_shap_values[positive_idx], color=colors_positive)
    
    # Dégradé pour les négatives
    colors_negative = sns.color_palette("Reds", len(negative_idx))
    plt.barh(data_test_sampled.columns[negative_idx], client_shap_values[negative_idx], color=colors_negative)
    
    plt.title("Feature Importance Locale du Client")
    plt.xlabel("SHAP Value")
    st.pyplot(plt)
    
# Fonction pour afficher les analyses univariées
def show_univariate_analysis():
    st.title(f"Analyses Univariées et Positionnement du Client {st.session_state.client_id}")

    selected_feature = st.selectbox("Sélectionnez une feature", data_test_sampled.columns)

    X = data_test_sampled[selected_feature]
    Y = model.predict(data_test_sampled)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(X[Y == 0], fill=True, cmap="Greens", levels=20)
    sns.kdeplot(X[Y == 1], fill=True, cmap="Reds", levels=20)
    plt.axvline(data_test_sampled.loc[st.session_state.client_id, selected_feature], 
                color='blue', linestyle='--', label='Client position')
    plt.title(f"Distribution de {selected_feature}")
    plt.legend()
    st.pyplot(plt)
    
# Fonction pour afficher les analyses bivariées
def show_bivariate_analysis():
    st.title(f"Analyses Bivariées et Positionnement du Client {st.session_state.client_id}")

    feature_x = st.selectbox("Sélectionnez la première feature", data_test_sampled.columns)
    feature_y = st.selectbox("Sélectionnez la deuxième feature", data_test_sampled.columns)
    
    X = data_test_sampled[feature_x]
    Y = data_test_sampled[feature_y]
    Z = model.predict(data_test_sampled)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=X[Z == 0], y=Y[Z == 0], cmap="Greens", fill=True, levels=20)
    sns.kdeplot(x=X[Z == 1], y=Y[Z == 1], cmap="Reds", fill=True, levels=20)
    plt.scatter(data_test_sampled.loc[st.session_state.client_id, feature_x], 
                data_test_sampled.loc[st.session_state.client_id, feature_y], 
                color='blue', label='Client position', s=100, edgecolor='k')
    plt.title(f"Contourf Plot de {feature_x} vs {feature_y}")
    plt.legend()
    st.pyplot(plt)
    
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

