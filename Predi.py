import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Chargement des fichiers du modèle
model = joblib.load("C:/Users/LENOVO/Documents/CHATBOT/Application prediction credit/model1.joblib")
scaler = joblib.load("C:/Users/LENOVO/Documents/CHATBOT/Application prediction credit/scaler (1).joblib")
features = joblib.load("C:/Users/LENOVO/Documents/CHATBOT/Application prediction credit/features.joblib")

# Configuration de la page
st.set_page_config(page_title="💳 Prédiction Compte Bancaire", layout="centered")
st.title("💡 Prédiction de l'utilisation d’un compte bancaire")

st.write("Veuillez remplir les informations ci-dessous :")

# --- Saisie utilisateur ---
age = st.slider("Âge", 15, 100, 30)
cellphone = st.radio("Avez-vous un téléphone ?", ["Yes", "No"])
education = st.selectbox("Niveau d'éducation", [
    "No formal education", "Primary education", 
    "Secondary education", "Tertiary education", 
    "Vocational/Specialised training", "Other/Dont know/RTA"
])
marital_status = st.selectbox("Statut matrimonial", [
    "Married/Living together", "Single/Never Married", 
    "Divorced/Seperated", "Widowed", "Dont know"
])
gender = st.selectbox("Sexe", ["Male", "Female"])
household_size = st.slider("Taille du ménage", 1, 20, 5)
country = st.selectbox("Pays", ["Kenya", "Tanzania", "Rwanda", "Uganda"])
location_type = st.selectbox("Lieu de résidence", ["Urban", "Rural"])
job_type = st.selectbox("Type d'emploi", [
    "Formally employed Private", "Formally employed Government", 
    "Self employed", "Informally employed", "Farming and Fishing",
    "Remittance Dependent", "Government Dependent", "Other Income",
    "No Income", "Dont Know/Refuse to answer"
])
relationship_with_head = st.selectbox("Lien avec le chef de ménage", [
    "Head of Household", "Spouse", "Child", "Parent", 
    "Other relative", "Other non-relatives"
])
year = 2018

# Seuil personnalisé
threshold = st.slider("🔧 Choisissez un seuil de probabilité", 0.0, 1.0, 0.5, 0.01)

# --- Construction du DataFrame ---
raw_input = pd.DataFrame([{
    "age_of_respondent": age,
    "cellphone_access": 1 if cellphone == "Yes" else 0,
    "education_level": education,
    "marital_status": marital_status,
    "gender_of_respondent": gender,
    "household_size": household_size,
    "country": country,
    "location_type": location_type,
    "job_type": job_type,
    "relationship_with_head": relationship_with_head,
    "year": year
}])

# Encodage one-hot + alignement des colonnes
input_encoded = pd.get_dummies(raw_input)
input_encoded = input_encoded.reindex(columns=features, fill_value=0)

# Normalisation
X_scaled = scaler.transform(input_encoded.values)

# --- Prédiction ---
if st.button("🔍 Prédire"):
    proba = model.predict_proba(X_scaled)[0][1]
    prediction = 1 if proba >= threshold else 0

    st.markdown("---")
    if prediction == 1:
        st.success(f"✅ Cette personne a probablement un **compte bancaire**.\n\n💡 Probabilité : **{proba:.2%}** (seuil = {threshold})")
    else:
        st.error(f"❌ Cette personne n’a probablement **PAS de compte bancaire**.\n\n📉 Probabilité : **{proba:.2%}** (seuil = {threshold})")
