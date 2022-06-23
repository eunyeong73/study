# sklearn.datasets.fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing



#1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


print(x)
print(y)
print(x.shape, y.shape)        # (20640, 8) (20640,)

print(datasets.feature_names)
print(datasets.DESCR)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, 
    # shuffle=True,
    random_state=66)


datasets = fetch_california_housing()
x = datasets.data
y = datasets.target



#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=200
          )



#4. 평가, 에측

y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  0.6437079906463623
# r2스코어 :  0.5308833557326117
