# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('Heart-attack-dataset.csv')

# Features and Target
X = df[['Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
        'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


#predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred)*100, "%")
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

#user input prediction
user_age = int(input("Enter age: "))
user_gender = int(input("Enter gender (1: Male, 0:Female): "))
user_hr = int(input("Enter heart rate: "))
user_sbp = int(input("Enter systolic blood pressure: "))
user_dbp = int(input("Enter diastolic blood pressure: "))
user_sugar = int(input("Enter blood sugar level: "))
user_ckmb = float(input("Enter CK-MB: "))
user_troponin = float(input("Enter troponin: "))

#convert into dataframe to fit in model
user_data = pd.DataFrame([[user_age, user_gender, user_hr, user_sbp, user_dbp, user_sugar, user_ckmb, user_troponin]], columns=['Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
                                  'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin'])

#make prediction
user_data_pred = model.predict(user_data)[0]

# Output result
# print(user_data_pred)
if user_data_pred == 1:
    print("Prediction: High risk of heart attack.")
else:
    print("Prediction: Low risk of heart attack.")