import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

df = pd.read_csv('cleaned_data.csv')

features = ["MA_10", "MA_50", "Volatility", "Volume"]
target = "Close"

x = df[features]
y = df[target]

feature_length = int(len(df)*0.8)
x_train = x[:feature_length]
y_train = y[:feature_length]

#model = RandomForestRegressor(
 #   n_estimators= 200,
 #   random_state= 42
 #)
model = LinearRegression()
model.fit(x_train, y_train)

with open("Linear_Regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to Linear_Regression_model.pkl")