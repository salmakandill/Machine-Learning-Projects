import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


data=pd.read_csv(r'data/medical_insurance.csv')

print(data.info())
print (data.head())
print(data.describe())
print(data.isnull().sum())
print(f"Dataset Shape : {data.shape}")
print(f"Columns : {list(data.columns)}")


print(data['alcohol_freq'].value_counts())

data['alcohol_freq'].fillna(data['alcohol_freq'].mode()[0],inplace=True)

categorical_columns=[
    'sex',
    'region',
    'urban_rural',
    'education',
    'marital_status',
    'employment_status',
    'smoker',
    'alcohol_freq',
    'plan_type',
    'network_tier'
]


X=data.drop([
    'annual_medical_cost',
    'person_id',
    'risk_score',
    'annual_premium',
    'monthly_premium',
    'claims_count',
    'avg_claim_amount',
    'total_claims_paid'],axis= 1)
X=pd.get_dummies(
    X,
    columns= categorical_columns,
    drop_first=True,
    dtype=int

)


y=data['annual_medical_cost']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=25)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
score=model.score(X_test,y_test)
r2=r2_score(y_test,y_pred)
rmse=np.sqrt(mse)

weights= pd.DataFrame({
    'Features':X.columns,
    'Weights':model.coef_
})
bias=model.intercept_

print(weights.head(20))
print('Bias = ',bias)
print(f'MAE = {mae:.2f}')
print(f'MSE = {mse:.2f}')
print(f'RMSE = {rmse:.2f}')
print('Score: ', score)
print(f'R2 = {r2:.4f}')

results=pd.DataFrame({
    'Actual Cost':y_test.values,
    'Predicted Cost':y_pred
})
print(results.head(10))


plt.figure(figsize=(8,6))

error = y_test - y_pred

plt.scatter(y_pred, error)

plt.axhline(y=0,color='red')

plt.xlabel('Predicted Cost')
plt.ylabel('Error')
plt.title('Error Plot')

plt.savefig('images/predicted_cost.png',dpi=300)
plt.show()