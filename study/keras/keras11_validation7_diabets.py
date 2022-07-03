# sklearn.datasets.load_diabetes
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)     #(442, 10) (442,)

print(datasets.feature_names)
print(datasets.DESCR)

#[실습]
# R2 0.62 이상

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=61, train_size=0.6)


datasets = load_diabetes()
x = datasets.data
y = datasets.target


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=100, validation_split=0.21) 


#4. 평가, 예측

y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# loss :  1678.2406005859375
# r2스코어 :  0.6653577770621634

##############validation 이후

# loss :  3069.047607421875
# r2스코어 :  0.5074052635374344