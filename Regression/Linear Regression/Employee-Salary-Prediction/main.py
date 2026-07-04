import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


data=pd.read_csv(r'data\Employers_data.csv')

print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())


X=data[['Experience_Years']]
y=data['Salary']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=True,random_state=25)
model = LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
score=model.score(X_test,y_test)
r2=r2_score(y_test,y_pred)
rmse=np.sqrt(mse)

weight=model.coef_[0]
bias=model.intercept_

print('Weight = ',weight)
print('Bias = ',bias)
print(f'MAE = {mae:.2f}')
print(f'MSE = {mse:.2f}')
print(f'RMSE = {rmse:.2f}')
print('Score: ', score)
print(f'R2 = {r2:.4f}')

print('Actual Salary = ',y_test)
print('Predicted Salary = ',y_pred)


y_all=model.predict(X)

plt.figure(figsize=(10,6))

plt.scatter(X,y,label='Actual Salary')
plt.plot(X,y_all,color='red', linewidth=2,label='Regression Line')

plt.xlabel('Experience Years')
plt.ylabel('Salary')
plt.title('Employee Salary Prediction')

plt.legend()
plt.show()