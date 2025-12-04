from sklearn.svm import SVR  # Regression version for continuous price
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

df = pd.read_csv('Winter_Fashion_Trends_Dataset.csv')

categorical_columns = ['Brand','Category','Color','Material','Style','Gender','Season','Trend_Status']
numeric_columns = ['Popularity_Score','Customer_Rating']

# Encode categorical columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[numeric_columns + categorical_columns]
y = df['Price(USD)']

model_svc = SVR(kernel='rbf')
model_svc.fit(X, y)

pickle.dump(model_svc, open('model_svc.pkl','wb'))

print("Model trained and saved successfully!")
