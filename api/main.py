from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Cargar pickles
# Encoding
with open('../models/target.pkl', 'rb') as file:
    target = pickle.load(file)
with open('../models/onehot.pkl', 'rb') as file:
    onehot = pickle.load(file)
# Scaling
with open('../models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
# Model
with open('../models/model_resampled.pkl', 'rb') as file:
    model = pickle.load(file)

cat_diff  = ['BusinessTravel',
            'Department',
            'EducationField',
            'JobRole',
            'MaritalStatus',
            'EnvironmentSatisfaction',
            'JobSatisfaction',
            'WorkLifeBalance',
            'JobInvolvement']

cat_no_diff = ['Education', 'Gender', 'JobLevel', 'StockOptionLevel','PerformanceRating']

# Inicializar la app de Flask
app = Flask(__name__)

# Ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos de la solicitud JSON
    input_data = pd.DataFrame(request.get_json(), index=[0])

    if not request:
        return jsonify({'error': 'No input data provided'}), 400

    # Realizar la predicción
    res_target = target.transform(input_data)
    res_onehot = pd.DataFrame(onehot.transform(res_target[cat_no_diff]).toarray(), columns = onehot.get_feature_names_out())
    res_encoded = pd.concat([res_target.drop(columns = cat_no_diff), res_onehot], axis = 1)
    res_scaled = pd.DataFrame(scaler.transform(res_encoded.drop(columns = res_onehot.columns)), columns = scaler.get_feature_names_out())
    res_scaled = pd.concat([res_scaled, res_onehot], axis = 1)
    prediction = model.predict(res_scaled)
    prediction_probability = model.predict_proba(res_scaled)[0, 1]

    # Devolver el resultado en formato JSON
    result = {
        'attrition_risk': 'Yes' if prediction == 1 else 'No',
        'probability': round(float(prediction_probability), 2)
    }

    return jsonify(result)

# Iniciar la aplicación
if __name__ == '__main__':
    app.run(debug=True)
