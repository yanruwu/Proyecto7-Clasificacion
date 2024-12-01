import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle

# Configurar el título general de la app
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

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
education_dict = {'Below College': "1", 'College': "2", 'Bachelor': "3", 'Master': "4", 'Doctor': "5"}
performance_dict = {"Excellent" : "3", "Outstanding" : "4"}
involvement_dict = {"Low" : "1", "Medium" : "2", "High" : "3", "Very High" : "4"}
worklife_dict = {"Bad" : "1", "Good" : "2", "Better" : "3", "Best" : "4", "unknown" : "unknown"}
jobsatisfaction_dict = {"Low" : "1", "Medium" : "2", "High" : "3", "Very High" : "4", "unknown" : "unknown"}
envsatisfaction_dict = {"Low" : "1.0", "Medium" : "2.0", "High" : "3.0", "Very High" : "4.0", "unknown" : "unknown"}

# Crear las páginas de la aplicación
st.sidebar.title("Navegación")
st.sidebar.markdown("---")
page = st.sidebar.radio("Ir a", ["Inicio" ,"Datos del DataFrame", "Modelo de Predicción"])

# Página de Inicio
if page == "Inicio":
    st.title("Bienvenido al Employee Attrition Predictor")
    st.write("Esta aplicación fue desarrollada para predecir la posible deserción de empleados basándose en diversas características laborales y personales.")
    st.write("## Acerca del Proyecto")
    st.write("Este proyecto utiliza modelos de machine learning para identificar empleados que puedan estar en riesgo de dejar la empresa. Utilizando datos como la satisfacción laboral, los ingresos, el nivel de involucramiento, entre otros, se puede predecir si un empleado podría tener la intención de renunciar.")
    
    st.write("## Objetivos del Proyecto")
    st.write("- Predecir la deserción de empleados con una alta precisión.")
    st.write("- Ayudar a las empresas a tomar decisiones informadas sobre retención de talento.")
    st.write("- Proporcionar una herramienta interactiva para explorar los datos de empleados y analizar patrones.")
    
    # st.image("../images/employee_retention.jpg", caption="Retención de Talento", use_container_width =True)
    # st.image("../images/attrition_factors.png", caption="Factores que contribuyen a la deserción de empleados", use_column_width=True)
    
    st.write("## Navegación de la Aplicación")
    st.write("Utiliza el menú de la barra lateral para navegar entre las diferentes secciones de la aplicación:")
    st.write("- **Inicio**: Esta página, donde puedes leer sobre el proyecto.")
    st.write("- **Modelo de Predicción**: Prueba el modelo para predecir la probabilidad de deserción de un empleado dado sus datos.")
    st.write("- **Datos del DataFrame**: Explora el DataFrame utilizado para entrenar el modelo y consulta el análisis estadístico de los datos.")

    st.write("## Datos del Proyecto")
    st.write("[Repositorio en GitHub](https://github.com/yanruwu/Proyecto8-Clasificacion) y [Mi perfil en LinkedIn](https://www.linkedin.com/in/yanruwujin/)")

# Página de Predicción
elif page == "Modelo de Predicción":
    # Título de la app
    st.title('Employee Attrition Predictor')
    st.write("Esta aplicación predice si un empleado tiene riesgo de dejar la empresa, basado en ciertas características.")

    # Crear un formulario para la entrada de las características
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider('Edad', min_value=df["Age"].min(), max_value=df["Age"].max(), step=1)
            distance_from_home = st.slider('Distancia desde casa (km)', min_value=df["DistanceFromHome"].min(), max_value=df["DistanceFromHome"].max(), step=1)
            monthly_income = st.number_input('Ingreso mensual (₹)', min_value=df["MonthlyIncome"].min(), max_value=df["MonthlyIncome"].max(), step=100)
            num_companies_worked = st.slider('Numero de empresas trabajadas', min_value=df["NumCompaniesWorked"].min(), max_value=df["NumCompaniesWorked"].max(), step=1)
            percent_salary_hike = st.slider('Aumento porcentual de salario (%)', min_value=df["PercentSalaryHike"].min(), max_value=df["PercentSalaryHike"].max(), step=1)
            total_working_years = st.slider('Años totales trabajados', min_value=df["TotalWorkingYears"].min(), max_value=df["TotalWorkingYears"].max(), step=1)
            training_times_last_year = st.slider('Cantidad de entrenamientos el último año', min_value=df["TrainingTimesLastYear"].min(), max_value=df["TrainingTimesLastYear"].max(), step=1)
            years_at_company = st.slider('Años en la compañía', min_value=df["YearsAtCompany"].min(), max_value=df["YearsAtCompany"].max(), step=1)
            years_since_last_promotion = st.slider('Años desde la última promoción', min_value=df["YearsSinceLastPromotion"].min(), max_value=df["YearsSinceLastPromotion"].max(), step=1)
            years_with_curr_manager = st.slider('Años con el actual gerente', min_value=df["YearsWithCurrManager"].min(), max_value=df["YearsWithCurrManager"].max(), step=1)
        with col2:
            environment_satisfaction = st.selectbox('Satisfacción con el ambiente', envsatisfaction_dict.keys())
            job_satisfaction = st.selectbox('Satisfacción con el trabajo', jobsatisfaction_dict.keys())
            work_life_balance = st.selectbox('Balance trabajo-vida', worklife_dict.keys())
            job_involvement = st.selectbox('Involucramiento en el trabajo', involvement_dict.keys())
            business_travel = st.selectbox('Viajes de negocio', df["BusinessTravel"].unique())
            department = st.selectbox('Departamento', df["Department"].unique())
            education = st.selectbox('Nivel de educación', ['Below College', 'College', 'Bachelor', 'Master', 'Doctor'])
            education_field = st.selectbox('Campo educativo', df["EducationField"].unique())
            job_role = st.selectbox('Rol de trabajo', df["JobRole"].unique())
            marital_status = st.selectbox('Estado civil', df["MaritalStatus"].unique())
            gender = st.selectbox('Genero', df["Gender"].unique())
            job_level = st.selectbox('Nivel de trabajo', [1, 2, 3, 4, 5])
            stock_option_level = st.selectbox('Nivel de opciones de acciones', [0, 1, 2, 3])
            performance_rating = st.selectbox('Calificación de desempeño', performance_dict.keys())

        # Botón para enviar el formulario
        submit_button = st.form_submit_button("Predecir")

    # Procesar la predicción al enviar el formulario
    if submit_button:
        # Transformar la entrada a un formato que el modelo pueda entender
        input_data = {
            'Age': age,
            'BusinessTravel': business_travel,
            'Department': department,
            'DistanceFromHome': distance_from_home,
            'Education': education_dict[education],
            'EducationField': education_field,
            'Gender': gender,
            'JobLevel': str(job_level),
            'JobRole': job_role,
            'MaritalStatus': marital_status,
            'MonthlyIncome': monthly_income,
            'NumCompaniesWorked': num_companies_worked,
            'PercentSalaryHike': percent_salary_hike,
            'StockOptionLevel': str(stock_option_level),
            'TotalWorkingYears': total_working_years,
            'TrainingTimesLastYear': training_times_last_year,
            'YearsAtCompany': years_at_company,
            'YearsSinceLastPromotion': years_since_last_promotion,
            'YearsWithCurrManager': years_with_curr_manager,
            'EnvironmentSatisfaction': envsatisfaction_dict[environment_satisfaction],
            'JobSatisfaction': jobsatisfaction_dict[job_satisfaction],
            'WorkLifeBalance': worklife_dict[work_life_balance],
            'JobInvolvement': involvement_dict[job_involvement],
            'PerformanceRating': performance_dict[performance_rating],
        }

        # Convertir input_data en un DataFrame
        input_df = pd.DataFrame(input_data, index=[0])
        input_json = input_df.iloc[0].to_dict()

        # Llamar a la API Flask para realizar la predicción
        api_url = 'http://127.0.0.1:5000/predict' 
        response = requests.post(api_url, json=input_json)   

        if response.status_code == 200:
            result = response.json()
            if result['attrition_risk'] == 'Yes':
                st.warning(f'Este empleado tiene riesgo de dejar la empresa. Probabilidad: {result["probability"]:.2f}')
            else:
                st.success(f'Este empleado tiene baja probabilidad de dejar la empresa. Probabilidad: {result["probability"]:.2f}')
        else:
            st.error(f'Error en la solicitud a la API Flask. Código de error: {response.status_code}, Detalles: {response.text}')

# Página de Datos del DataFrame
elif page == "Datos del DataFrame":
    st.title("Datos del DataFrame")
    st.write("Aquí se encuentran los datos del DataFrame utilizado para el análisis.")
    
    # Aplicar estilos al DataFrame para colorear según 'Attrition'
    def highlight_attrition(val):
        color = 'red' if val == 'Yes' else 'green'
        return f'background-color: {color}'
    
    # Aplicar el estilo y mostrar el DataFrame
    if 'Attrition' in df.columns:
        styled_df = df.style.applymap(highlight_attrition, subset=['Attrition'])
        st.dataframe(styled_df)
    else:
        st.dataframe(df)

    st.write("Resumen Estadístico:")
    attrition_filter = st.selectbox(label = "Attrition", options = ["Both", "Yes", "No"])
    if attrition_filter == "Both":
        st.write(df.describe())
    else:
        st.write(df[df["Attrition"] == attrition_filter].describe())