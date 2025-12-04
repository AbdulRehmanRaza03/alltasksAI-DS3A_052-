import pickle
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
processed_data = pd.read_csv('Winter_Fashion_Trends_Dataset.csv')

# Define categorical and numeric columns
categorical_columns = ['Brand','Category','Color','Material','Style','Gender','Season','Trend_Status']
numeric_columns = ['Popularity_Score','Customer_Rating']

# LabelEncoders for categorical features
label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    label_encoders[col].fit(processed_data[col])

# Load trained model
model_svc = pickle.load(open('model_svc.pkl', 'rb'))

# Feature order for model
model_feature_order = numeric_columns + categorical_columns

@app.route('/')
def index():
    sample_data = {col: processed_data[col].unique() for col in categorical_columns}
    return render_template('index.html', sample_data=sample_data, numeric_columns=numeric_columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {col: request.form.get(col) for col in categorical_columns}
        input_data.update({col: float(request.form.get(col, 0)) for col in numeric_columns})

        # Encode categorical features
        for col in categorical_columns:
            input_data[col] = label_encoders[col].transform([input_data[col]])[0]

        input_df = pd.DataFrame([input_data])
        input_df = input_df[model_feature_order]

        prediction = model_svc.predict(input_df)
        predicted_price = prediction[0]

        sample_data = {col: processed_data[col].unique() for col in categorical_columns}
        return render_template('index.html', predicted_price=predicted_price, sample_data=sample_data, numeric_columns=numeric_columns)
    except Exception as e:
        sample_data = {col: processed_data[col].unique() for col in categorical_columns}
        return render_template('index.html', error=str(e), sample_data=sample_data, numeric_columns=numeric_columns)

if __name__ == '__main__':
    app.run(debug=True)
