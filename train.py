import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load datasets
train = pd.read_csv("E:/House/dataset/train.csv")
test = pd.read_csv("E:/House/dataset/test.csv")

# Save target and apply log transform
y = np.log1p(train["SalePrice"])
train.drop(["SalePrice"], axis=1, inplace=True)

# Combine train and test for preprocessing
data = pd.concat([train, test], sort=False)
data.drop(["Id"], axis=1, inplace=True)

# One-hot encoding for categorical variables
data = pd.get_dummies(data)

# Fill missing values with mean
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split back the data
X_train = data_imputed[:len(y)]
X_test = data_imputed[len(y):]

# Train-validation split
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)

# -----------------------------
# 1. Lasso Regression
# -----------------------------
lasso = Lasso(alpha=0.001, max_iter=10000, random_state=42)
lasso.fit(X_train_split, y_train_split)
lasso_preds = lasso.predict(X_val)
lasso_rmse = np.sqrt(mean_squared_error(y_val, lasso_preds))
print(f"Lasso RMSE: {lasso_rmse:.4f}")

# -----------------------------
# 2. Random Forest Regressor
# -----------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_split, y_train_split)
rf_preds = rf.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
print(f"Random Forest RMSE: {rf_rmse:.4f}")

# -----------------------------
# 3. XGBoost Regressor
# -----------------------------
xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
xgb.fit(X_train_split, y_train_split)
xgb_preds = xgb.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_preds))
print(f"XGBoost RMSE: {xgb_rmse:.4f}")

# Save the XGBoost model
joblib.dump(xgb, 'E:/House/dataset/house_prediction_model.pkl')
print("✅ XGBoost model saved as 'house_prediction_model.pkl'")

# -----------------------------
# Feature Importance - XGBoost
# -----------------------------
plt.figure(figsize=(10, 6))
xgb_importance = pd.Series(xgb.feature_importances_, index=X_train.columns)
xgb_importance.nlargest(20).plot(kind='barh')
plt.title("Top 20 Feature Importances - XGBoost")
plt.tight_layout()
plt.show()

# -----------------------------
# Residual Plot - XGBoost
# -----------------------------
residuals = y_val - xgb_preds
plt.figure(figsize=(8, 6))
sns.scatterplot(x=xgb_preds, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Log SalePrice")
plt.ylabel("Residuals")
plt.title("Residual Plot - XGBoost")
plt.show()

# -----------------------------
# Learning Curve - XGBoost
# -----------------------------
train_errors, val_errors = [], []
n_estimators_range = range(50, 500, 50)

for m in n_estimators_range:
    model = XGBRegressor(n_estimators=m, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train_split, y_train_split)
    train_pred = model.predict(X_train_split)
    val_pred = model.predict(X_val)
    train_errors.append(np.sqrt(mean_squared_error(y_train_split, train_pred)))
    val_errors.append(np.sqrt(mean_squared_error(y_val, val_pred)))

plt.plot(n_estimators_range, train_errors, label="Training RMSE")
plt.plot(n_estimators_range, val_errors, label="Validation RMSE")
plt.xlabel("Number of Trees")
plt.ylabel("RMSE")
plt.legend()
plt.title("Learning Curve - XGBoost")
plt.grid(True)
plt.show()

# -----------------------------
# Final Prediction on Test Set
# -----------------------------
final_preds = xgb.predict(X_test)
final_preds = np.expm1(final_preds)

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": final_preds
})
submission.to_csv("E:/House/dataset/submission.csv", index=False)
print("✅ Submission file 'submission.csv' created.")

# -----------------------------
# Loading the Saved Model and Using it for Future Predictions
# -----------------------------
# Load the saved XGBoost model
xgb_loaded = joblib.load('E:/House/dataset/house_prediction_model.pkl')

# Use the loaded model for predictions
final_preds_loaded = xgb_loaded.predict(X_test)
final_preds_loaded = np.expm1(final_preds_loaded)

# Create submission file using the loaded model
submission_loaded = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": final_preds_loaded
})
submission_loaded.to_csv("E:/House/dataset/submission_loaded.csv", index=False)
print("✅ Submission file 'submission_loaded.csv' created using the loaded model.")
