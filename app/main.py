import streamlit as st
import numpy as np
import pandas as pd
import requests
import json
import pickle

# Cargar el dataframe con las características
# (Reemplazar con la ruta adecuada para el dataframe si es necesario)
df = pd.read_pickle("../datos/clean.pkl")

# Cargar modelos y transformadores
with open('../models/target.pkl', 'rb') as file:
    target_encoder = pickle.load(file)
with open('../models/onehot.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)
with open('../models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Diccionario para codificar la educación
education_dict = {'Below College': 0, 'College': 1, 'Bachelor': 2, 'Master': 3, 'Doctor': 4}

# Título de la app
st.title('Employee Attrition Predictor')
st.write("Esta aplicación predice si un empleado tiene riesgo de dejar la empresa, basado en ciertas características.")

# Crear un formulario para la entrada de las características
with st.form("predict_form"):
    age = st.slider('Edad', min_value=df["Age"].astype(int).min(), max_value=df["Age"].astype(int).max(), step=1)
    distance_from_home = st.slider('Distancia desde casa (km)', min_value=df["DistanceFromHome"].astype(int).min(), max_value=df["DistanceFromHome"].astype(int).max(), step=1)
    monthly_income = st.number_input('Ingreso mensual ($)', min_value=df["MonthlyIncome"].astype(int).min(), max_value=df["MonthlyIncome"].astype(int).max(), step=100)
    num_companies_worked = st.slider('Numero de empresas trabajadas', min_value=df["NumCompaniesWorked"].astype(int).min(), max_value=df["NumCompaniesWorked"].astype(int).max(), step=1)
    percent_salary_hike = st.slider('Aumento porcentual de salario (%)', min_value=df["PercentSalaryHike"].astype(int).min(), max_value=df["PercentSalaryHike"].astype(int).max(), step=1)
    total_working_years = st.slider('Años totales trabajados', min_value=df["TotalWorkingYears"].astype(int).min(), max_value=df["TotalWorkingYears"].astype(int).max(), step=1)
    training_times_last_year = st.slider('Cantidad de entrenamientos el último año', min_value=df["TrainingTimesLastYear"].astype(int).min(), max_value=df["TrainingTimesLastYear"].astype(int).max(), step=1)
    years_at_company = st.slider('Años en la compañía', min_value=df["YearsAtCompany"].astype(int).min(), max_value=df["YearsAtCompany"].astype(int).max(), step=1)
    years_since_last_promotion = st.slider('Años desde la última promoción', min_value=df["YearsSinceLastPromotion"].astype(int).min(), max_value=df["YearsSinceLastPromotion"].astype(int).max(), step=1)
    years_with_curr_manager = st.slider('Años con el actual gerente', min_value=df["YearsWithCurrManager"].astype(int).min(), max_value=df["YearsWithCurrManager"].astype(int).max(), step=1)
    environment_satisfaction = st.selectbox('Satisfacción con el ambiente (1-4)', [1, 2, 3, 4])
    job_satisfaction = st.selectbox('Satisfacción con el trabajo (1-4)', [1, 2, 3, 4])
    work_life_balance = st.selectbox('Balance trabajo-vida (1-4)', [1, 2, 3, 4])
    job_involvement = st.selectbox('Involucramiento en el trabajo (1-4)', [1, 2, 3, 4])
    business_travel = st.selectbox('Viajes de negocio', df["BusinessTravel"].unique())
    department = st.selectbox('Departamento', df["Department"].unique())
    education = st.selectbox('Nivel de educación', ['Below College', 'College', 'Bachelor', 'Master', 'Doctor'])
    education_field = st.selectbox('Campo educativo', df["EducationField"].unique())
    job_role = st.selectbox('Rol de trabajo', df["JobRole"].unique())
    marital_status = st.selectbox('Estado civil', df["MaritalStatus"].unique())
    gender = st.selectbox('Genero', df["Gender"].unique())
    job_level = st.selectbox('Nivel de trabajo', [1, 2, 3, 4, 5])
    stock_option_level = st.selectbox('Nivel de opciones de acciones', [0, 1, 2, 3])
    performance_rating = st.selectbox('Calificación de desempeño', [3, 4])

    # Botón para enviar el formulario
    submit_button = st.form_submit_button("Predecir")

# Procesar la predicción al enviar el formulario
if submit_button:
    # Transformar la entrada a un formato que el modelo pueda entender
    input_data = {
        'Age': float(age),
        'BusinessTravel': str(business_travel),
        'Department': str(department),
        'DistanceFromHome': float(distance_from_home),
        'Education': education_dict[education],
        'EducationField': str(education_field),
        'Gender': str(gender),
        'JobLevel': str(job_level),
        'JobRole': str(job_role),
        'MaritalStatus': str(marital_status),
        'MonthlyIncome': float(monthly_income),
        'NumCompaniesWorked': float(num_companies_worked),
        'PercentSalaryHike': float(percent_salary_hike),
        'StockOptionLevel': str(stock_option_level),
        'TotalWorkingYears': float(total_working_years),
        'TrainingTimesLastYear': float(training_times_last_year),
        'YearsAtCompany': float(years_at_company),
        'YearsSinceLastPromotion': float(years_since_last_promotion),
        'YearsWithCurrManager': float(years_with_curr_manager),
        'EnvironmentSatisfaction': str(environment_satisfaction),
        'JobSatisfaction': str(job_satisfaction),
        'WorkLifeBalance': str(work_life_balance),
        'JobInvolvement': str(job_involvement),
        'PerformanceRating': str(performance_rating),
    }

    # Convertir input_data en un DataFrame
    input_df = pd.DataFrame(input_data, index=[0])
    
    # Asegurarse de que todas las columnas necesarias están presentes
    input_df = input_df
    
    # Aplicar transformación de encoding y escalado

    # Convertir el DataFrame procesado a un diccionario para enviarlo a la API
    


    
    input_json = json.dumps(input_df.iloc[0].to_dict())

    st.write(input_df.dtypes)

    # Llamar a la API Flask para realizar la predicción
    api_url = 'http://127.0.0.1:5000/predict'  # Cambiar al endpoint de la API Flask si es necesario
    response = requests.post(api_url, json=input_json)

    if response.status_code == 200:
        result = response.json()
        if result['attrition_risk'] == 'Yes':
            st.warning(f'Este empleado tiene riesgo de dejar la empresa. Probabilidad: {result["probability"]:.2f}')
        else:
            st.success(f'Este empleado tiene baja probabilidad de dejar la empresa. Probabilidad: {result["probability"]:.2f}')
    else:
        st.error('Error en la solicitud a la API Flask. Por favor, intente de nuevo.')
