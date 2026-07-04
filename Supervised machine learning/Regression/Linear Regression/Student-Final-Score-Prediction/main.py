import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import( 
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


data=pd.read_csv('data\student_performance_interactions.csv')

print(data.info())
print (data.head())
print(data.describe())
print(data.isnull().sum())
print(f"Dataset Shape : {data.shape}")
print(f"Columns : {list(data.columns)}")

X=data[['daily_study_hours']]
y=data['final_score']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=25)

model=LinearRegression()
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

results=pd.DataFrame({
    'Actual Final Score':y_test.values,
    'Predicted Final Score':y_pred
})
print(results.head(10))

y_all=model.predict(X)

plt.figure(figsize=(10,6))

plt.scatter(X,y,label='Actual Final Score')
plt.plot(X,y_all,color='red', linewidth=2,label='Regression Line')

plt.xlabel('Daily Study Hours')
plt.ylabel('Final Score')
plt.title('Final Score Prediction')

plt.legend()
plt.show()