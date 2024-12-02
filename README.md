# 🚀 Clasificación de Empleados: Predicción de Attrition

## Descripción del Proyecto 📊

Este proyecto tiene como objetivo clasificar empleados con base en la probabilidad de *attrition* (rotación o deserción) dentro de una empresa. Utilizando datos reales de recursos humanos, se desarrollan modelos de machine learning capaces de identificar patrones que conducen a la deserción de empleados, permitiendo a la empresa tomar decisiones proactivas y mejorar la retención de talento. El repositorio completo se encuentra disponible en [GitHub](https://github.com/yanruwu/Proyecto8-Clasificacion).

### Proceso del Proyecto 🚀

1. **Análisis Exploratorio de Datos (EDA)**:
   - El primer paso fue realizar un análisis exploratorio de datos (EDA) para comprender mejor la naturaleza de los datos y tratar valores nulos e inconsistencias. Para ello, se utilizó `pandas` para el manejo de datos, y `matplotlib` y `seaborn` para generar visualizaciones que permitieran identificar patrones y correlaciones entre variables relevantes.

2. **Preprocesamiento de Datos**:
   - En la etapa de preprocesamiento, se llevaron a cabo varias actividades fundamentales para preparar los datos. Se empleó `support_prep.py` en la carpeta `src/` para realizar la codificación de variables categóricas (one-hot encoding con el apoyo del archivo `onehot.pkl` y target encoding, dependiendo de si la variable era nominal u ordinal, utilizando `target.pkl`), la normalización de las características (feature scaling) con el escalador almacenado en `scaler.pkl`, y la detección y tratamiento de outliers para garantizar que el modelo no se viera afectado por valores extremos. También se emplearon técnicas para eliminar duplicados y gestionar el desbalanceo en los datos.

3. **Iteración de Modelos**:
   - Se llevaron a cabo diferentes iteraciones de modelos de machine learning. Se entrenaron varios modelos, incluyendo `Logistic Regression`, `Decision Tree`, `Random Forest`, `Gradient Boosting`, `XGBoost` y `Support Vector Classifier (SVC)`. Se realizaron tres fases de iteraciones:
     - Con los datos originales.
     - Eliminando duplicados.
     - Gestionando el desbalanceo mediante técnicas de remuestreo (ver `model_resampled.pkl`).
   - Se utilizó `GridSearchCV` para optimizar los hiperparámetros de cada modelo y encontrar la configuración óptima para maximizar métricas como `accuracy`, `precision`, `recall`, y `F1-score`.

4. **Selección del Mejor Modelo**:
   - Tras evaluar el rendimiento de todos los modelos, se seleccionó el mejor para su implementación. El modelo seleccionado fue `XGBoost`, el cual mostró el mejor rendimiento en las métricas consideradas. Este modelo se utilizó para crear una API mediante `Flask`, ubicada en la carpeta `api/`. Esta API permite acceder a las predicciones de forma sencilla y escalable.

5. **Creación de Webapp**:
   - Posteriormente, se desarrolló una aplicación web con `Streamlit` (`app/main.py`) que llama a la API de Flask y presenta de manera amigable los resultados a los usuarios. Esto permite a los departamentos de recursos humanos visualizar fácilmente las predicciones y factores asociados al attrition de los empleados.
   - También se intentó implementar una aplicación usando `Reflex` como frontend (código en `reflexapp/`), pero este intento no tuvo éxito. Sin embargo, se considera retomarlo como parte de los próximos pasos del proyecto.

## Estructura del Proyecto 🗂️
```
├── api/                        # Código relacionado con la API del proyecto
│   └── main.py                 # Código en Flask para la creación de la API
├── app/                        # Aplicación frontend para visualizar los resultados
│   └── main.py                 # Código en Streamlit de la webapp que llama a la API de Flask
├── datos/                      # Archivos CSV y datos en crudo
│   ├── clean.pkl               # Datos preprocesados en formato pickle
│   ├── diccionario-datos.xlsx  # Diccionario de datos con la descripción de cada columna
│   ├── employee_survey_data.csv # Datos de encuestas realizadas a empleados
│   ├── general_data.csv        # Datos generales de los empleados
│   ├── manager_survey_data.csv # Datos de encuestas realizadas a los gerentes
│   ├── prepped.pkl             # Datos preparados para el modelado en formato pickle
│   ├── prepped_nodup.pkl       # Datos preparados sin duplicados en formato pickle
├── models/                     # Modelos entrenados y sus configuraciones
│   ├── model_resampled.pkl     # Modelo entrenado utilizando técnicas de remuestreo
│   ├── onehot.pkl              # Codificador one-hot usado en el preprocesamiento
│   ├── scaler.pkl              # Escalador utilizado para normalizar los datos
│   ├── target.pkl              # Datos del target preprocesado para el modelado
├── img/                        # Imágenes
├── notebooks/                  # Notebooks Jupyter para EDA y desarrollo de modelos
│   ├── 1-eda.ipynb             # Análisis exploratorio de los datos
│   ├── 2-preprocess.ipynb      # Limpieza y preparación de los datos
│   ├── 3.1-models.ipynb        # Entrenamiento inicial de modelos
│   ├── 3.2-models_nodup.ipynb  # Modelos entrenados sin duplicados
│   ├── 3.3-models_nodup_balanced.ipynb # Modelos entrenados sin duplicados con datos balanceados
│   └── api_tests.ipynb         # Pruebas para la API utilizando Jupyter
├── reflexapp/                  # Código para la implementación de Reflex como frontend
├── src/                        # Código fuente principal para el preprocesamiento y modelado
│   ├── support_exchangerate.py # Funciones de soporte para el manejo de tasas de cambio
│   ├── support_models.py       # Funciones de soporte para la creación y evaluación de modelos
│   ├── support_plots.py        # Funciones para la generación de gráficos y visualizaciones
│   └── support_prep.py         # Funciones para la preparación y limpieza de los datos
├── environment.yml             # Archivo de configuración para gestionar dependencias del entorno
└── README.md                   # Documentación del proyecto
```

## Instalación y Requisitos ⚙️

Para configurar el entorno de desarrollo y asegurarse de que todas las dependencias necesarias estén instaladas, se deben seguir estos pasos:

### Requisitos

- Python 3.8 o superior 🐍
- [Anaconda](https://www.anaconda.com/products/distribution) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (opcional, pero recomendado)

### Paquetes Necesarios

El proyecto utiliza los siguientes paquetes:

- [`pandas`](https://pandas.pydata.org/pandas-docs/stable/): Para la manipulación y análisis de datos.
- [`numpy`](https://numpy.org/doc/stable/): Para operaciones numéricas y manejo de arrays.
- [`scikit-learn`](https://scikit-learn.org/stable/): Para la creación y evaluación de modelos de machine learning.
- [`category_encoders`](https://pypi.org/project/category-encoders/): Para el target encoding.
- [`matplotlib`](https://matplotlib.org/stable/users/index.html): Para la visualización de datos.
- [`seaborn`](https://seaborn.pydata.org/): Para visualización estadística de datos.
- [`imblearn`](https://imbalanced-learn.org/stable/): Para la gestión de desbalance.
- [`shap`](https://shap.readthedocs.io/en/latest/): Para plots de resumen tipo SHAP.
- [`streamlit`](https://docs.streamlit.io/): Para creación de webapp desde Python.
- [`flax`](https://flask.palletsprojects.com/en/stable/): Para la creación de APIs desde Python.

### Instalación

1. **Clonar el repositorio:**

   ```bash
   git clone https://github.com/yanruwu/Proyecto8-Clasificacion
   cd Proyecto8-Clasificacion
   ```

2. **Crear un entorno virtual:**

   Para crear el entorno de Conda, usar el siguiente comando:

   ```bash
   conda env create -f environment.yml
   ```

   O si se prefiere usar venv:

   ```bash
   python -m venv venv
   source venv/bin/activate  # En macOS/Linux
   venv\Scripts\activate     # En Windows
   ```

### Ejecutar la API y la Webapp

- **API (Flask)**:

  - Navegar al directorio `api` y ejecutar `main.py` con el siguiente comando:
    ```bash
    python main.py
    ```

- **Webapp (Streamlit)**:

  - Navegar al directorio `app` y ejecutar `main.py` con el siguiente comando:
    ```bash
    streamlit run main.py
    ```


### Conclusiones

Este proyecto ha permitido identificar y entender los principales factores que influyen en la retención de empleados, utilizando un modelo óptimo basado en **XGBoost**. Después de eliminar duplicados y balancear las clases con **SMOTETomek**, el modelo ha mostrado una capacidad notable para predecir la probabilidad de **attrition**, proporcionando información útil sobre los factores más determinantes para la retención.

- Las características más importantes para la predicción de la attrition son **Estado Civil**, **Frecuencia de Viajes**, **Años Totales de Experiencia** y **Satisfacción con el Ambiente Laboral**. Estas variables indican que la estabilidad personal, la carga de viajes, la experiencia acumulada y el nivel de satisfacción con el entorno laboral juegan un papel crucial en la decisión de los empleados de permanecer en la empresa.

- El **SHAP Summary Plot** nos ha permitido entender cómo las diferentes características afectan la predicción del modelo. Factores como una mayor **satisfacción laboral**, **menos empresas previas**, **edad avanzada** y **promociones recientes** tienden a tener un impacto positivo en la retención, mientras que una **falta de promoción** y **baja satisfacción con el ambiente** tienen un efecto negativo.

![Texto Alternativo](img\shap.png)

- Aunque el modelo XGBoost presenta un **recall** alto y proporciona buenas predicciones, también se observó cierta tendencia al **overfitting** cuando se entrenó con datos balanceados, lo que implica que podría ser necesario ajustar mejor los hiperparámetros o explorar otros enfoques para mejorar la capacidad de generalización.

| Dataset | Precision | Accuracy | Recall | F1 Score | Kappa |
|---------|-----------|----------|--------|----------|-------|
| train   | 1.000000  | 0.999441 | 0.998889 | 0.999444 | 0.998882 |
| test    | 0.921127  | 0.897001 | 0.865079 | 0.892224 | 0.793783 |



- Además de la iteración con datos balanceados y sin duplicados, se realizaron otras iteraciones con el conjunto completo de datos y con datos solo sin duplicados. Estas iteraciones permitieron observar las diferencias en el rendimiento del modelo, evidenciando que la eliminación de duplicados y el balanceo afectan significativamente la capacidad de generalización y la estabilidad de las predicciones.

Este análisis podría ser una base sólida para implementar políticas que mejoren la satisfacción laboral y reduzcan la rotación de personal.



## Próximos Pasos 🔍

1. **Implementar un Dashboard**:
   - Desarrollar un dashboard interactivo para visualizar la probabilidad de attrition y los factores relevantes en tiempo real.

2. **Explorar el Uso de Reflex**:
   - Se intentará nuevamente implementar una aplicación utilizando Reflex como frontend. El objetivo es mejorar la interfaz y la experiencia del usuario, considerando las limitaciones encontradas previamente.
3. **Implementación de exchange**:
    - Implementar una API que permita seleccionar el tipo de moneda que se usa en la predicción de la permanencia (soporte)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para mejorar el proyecto, se puede abrir un pull request o una issue.

