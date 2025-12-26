import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

df = pd.read_csv('cleaned_data.csv')
features = ["MA_10", "MA_50", "Volatility", "Volume"]
target = "Close"

x = df[features]
y = df[target]
feature_length = int(len(df)*0.8)
x_test = x[feature_length:]
y_test = y[feature_length:]
with open("Linear_Regression_model.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(x_test)

MAE = mean_absolute_error(y_test, predictions)
RMSE = root_mean_squared_error(y_test, predictions)

print("mae:", MAE)
print("rmse:", RMSE)

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label = 'Actual Prices')
plt.plot(predictions, label = 'Predicted Prices')
plt.legend()
plt.title('Actual vs Predicted Stock Prices')
plt.show()