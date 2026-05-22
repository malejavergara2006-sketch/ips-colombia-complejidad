import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Cargar datos
df = pd.read_csv("Listado_de_IPS_en_Colombia_según_su_nivel_de_complejidad_20260508.csv")
df_limpio = df[df['nivel'].notna()].copy()
df_limpio = df_limpio[['depa_nombre','muni_nombre','nombre_prestador','nivel','caracter','ese']]
df_limpio = df_limpio[df_limpio['caracter'].notna()].reset_index(drop=True)

# Codificación
le_depa = LabelEncoder()
le_caracter = LabelEncoder()
le_ese = LabelEncoder()
df_limpio['depa_cod'] = le_depa.fit_transform(df_limpio['depa_nombre'])
df_limpio['caracter_cod'] = le_caracter.fit_transform(df_limpio['caracter'])
df_limpio['ese_cod'] = le_ese.fit_transform(df_limpio['ese'])
df_limpio['nivel_cod'] = df_limpio['nivel'].astype(int)

# Entrenar modelo
X = df_limpio[['depa_cod','caracter_cod','ese_cod']]
y = df_limpio['nivel_cod']
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

# Interfaz
st.title("🏥 IPS Colombia — Predictor de Nivel de Complejidad")
st.write("Analiza la cobertura de salud en Colombia según el nivel de complejidad de las IPS.")

st.sidebar.header("Selecciona los datos de la IPS")

departamento = st.sidebar.selectbox("Departamento", sorted(df_limpio['depa_nombre'].unique()))
caracter = st.sidebar.selectbox("Tipo de institución", sorted(df_limpio['caracter'].unique()))
ese = st.sidebar.radio("¿Es Empresa Social del Estado?", ['SI', 'NO'])

if st.sidebar.button("Predecir Nivel"):
    depa_cod = le_depa.transform([departamento])[0]
    caracter_cod = le_caracter.transform([caracter])[0]
    ese_cod = le_ese.transform([ese])[0]
    
    pred = modelo.predict([[depa_cod, caracter_cod, ese_cod]])[0]
    
    if pred == 1:
        st.success("✅ Nivel 1 — Atención Básica (medicina general, vacunación)")
    elif pred == 2:
        st.warning("⚠️ Nivel 2 — Atención Intermedia (algunas especialidades)")
    else:
        st.error("🔴 Nivel 3 — Alta Complejidad (cirugías especializadas, UCI)")

st.subheader("📊 Distribución de IPS por Nivel")
conteo = df_limpio['nivel_cod'].value_counts().sort_index()
st.bar_chart(conteo)