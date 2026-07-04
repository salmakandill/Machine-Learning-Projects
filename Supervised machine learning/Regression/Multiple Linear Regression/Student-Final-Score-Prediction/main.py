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


data=pd.get_dummies(
    data,
    columns=['parent_education_level', 'study_environment'],
    drop_first=True,
    dtype=int
)

X=data.drop(['student_id' ,'final_score','grade','pass_fail'],axis=1)
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
    'Actual Final Score':y_test.values,
    'Predicted Final Score':y_pred
})
print(results.head(10))


plt.figure(figsize=(10,6))

plt.scatter(y_test,y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(),y_test.max()],
    color='red',
    linewidth=2
)

plt.xlabel('Actual Final Score')
plt.ylabel('Predicted Final Score')
plt.title('Actual vs Predicted Final Score')

plt.show()
plt.savefig('images/predicted_score.png',dpi=300)