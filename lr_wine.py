import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------
# Load dataset
# --------------------------------

df = pd.read_csv("winequality_red.csv", sep=",")

print(df.head())
print(df.shape)
print(df.columns)


# --------------------------------
# Separate X and y
# --------------------------------

X = df.drop("quality", axis=1)
y = df["quality"]


# --------------------------------
# Train test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1234
)


# --------------------------------
# Create Linear Regression model
# --------------------------------

model = LinearRegression()


# --------------------------------
# Train model
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# Prediction
# --------------------------------

y_pred = model.predict(X_test)


# --------------------------------
# Evaluation
# --------------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# --------------------------------
# Save model
# --------------------------------

obj = {
    "model": model,
    "columns": list(X.columns)
}

joblib.dump(obj, "winequality.joblib")

print("Model saved successfully!")