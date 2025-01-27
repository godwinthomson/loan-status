import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.header("Loan_Status_Prediction")

# Load the dataset
df = pd.read_csv(r'C:\Users\admin\Documents\Projects\Loan Status Prediction Project\Loan_Prediction_dataset.csv')

# Basic data exploration
print(df.head())
print(df.info())
print(df.describe())

# Step 1: Data Preprocessing
# Handle missing values (e.g., drop or fill missing values)
df = df.dropna()  # Drop rows with missing values (or alternatively df.fillna() to impute)

# Encode categorical variables (if any) - Assume 'Gender' is categorical
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Split features and target
X = df.drop('Loan_Status', axis=1)  # Features (input variables)
y = df['Loan_Status']  # Target variable

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature Scaling (SVM is sensitive to feature scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the SVM model
svm = SVC(kernel='linear')  # You can also try 'rbf', 'poly', 'sigmoid' for different kernels
svm.fit(X_train_scaled, y_train)

# Step 5: Predict on the test set
y_pred = svm.predict(X_test_scaled)

# Step 6: Evaluation

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Approved', 'Approved'], yticklabels=['Not Approved', 'Approved'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Optional: Hyperparameter Tuning (for better performance)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf'],  # Kernel type
    'gamma': ['scale', 'auto'],  # Kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid’
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Best Parameters from Grid Search
print('Best Parameters:', grid_search.best_params_)

# Re-train the model with the best parameters
best_svm = grid_search.best_estimator_
best_svm.fit(X_train_scaled, y_train)

# Re-evaluate with the best model
y_pred_best = best_svm.predict(X_test_scaled)
print(f'Accuracy (with tuned parameters): {accuracy_score(y_test, y_pred_best):.4f}')
