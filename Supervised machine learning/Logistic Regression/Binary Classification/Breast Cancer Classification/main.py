import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

data = pd.read_csv(r'data\Cancer_Data.csv')

print (data.info())
print(data.head())
print(data.describe())
print(data.isna().sum())
print(f'Dataset shape:  {data.shape}')
print(f'Dataset Columns : {data.columns} ')

data=data.drop(['id','Unnamed: 32'],axis=1)

data['diagnosis']=data['diagnosis'].map({
    'M': 1,
    'B':0
})

X=data.drop(['diagnosis'],axis=1)
y=data['diagnosis']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)

scaler=StandardScaler()
X_train_scaled =scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model=LogisticRegression(max_iter=1000)
model.fit(X_train_scaled,y_train)

y_pred=model.predict(X_test_scaled)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1score=f1_score(y_test,y_pred)


print(f'Accuracy = {accuracy:.2f}')
print(f'Precision = {precision:.2f}')
print(f'Recall = {recall:.2f}')
print(f'F1_Score = {f1score:.2f}')

results=pd.DataFrame({
    'Actual Diagnosis':y_test.values,
    'Predicted Diagnosis':y_pred
})
print(results.head(10))

ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
plt.title('Confusion Matrix')
plt.savefig("images/confusion_matrix.png", dpi=300)
plt.show()

