import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('train.csv')


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  
df['Embarked'] = label_encoder.fit_transform(df['Embarked']) 


X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_model.fit(X_train, y_train)


lr_y_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_y_pred)
print(f"Logistic Regression Model Accuracy: {lr_accuracy * 100:.2f}%")
print("Logistic Regression Classification Report:\n", classification_report(y_test, lr_y_pred))


joblib.dump(lr_model, 'titanic_lr_model.pkl')
print("Logistic Regression Model saved as 'titanic_lr_model.pkl'")


train_accuracy = lr_model.score(X_train, y_train)
test_accuracy = lr_accuracy


plt.figure(figsize=(8, 6))
plt.bar(['Training Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy], color=['skyblue', 'salmon'])
plt.ylabel('Accuracy')
plt.ylim(0, 1)  
plt.title('Training vs Test Accuracy (Logistic Regression)')
plt.show()


correct_predictions = np.sum(lr_y_pred == y_test)
incorrect_predictions = np.sum(lr_y_pred != y_test)


plt.figure(figsize=(8, 6))
plt.pie([correct_predictions, incorrect_predictions], labels=['Correct', 'Incorrect'], 
        autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'salmon'])
plt.title('Correct vs Incorrect Predictions on Test Set (Logistic Regression)')
plt.show()


lr_cm = confusion_matrix(y_test, lr_y_pred)


plt.figure(figsize=(6, 5))
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()