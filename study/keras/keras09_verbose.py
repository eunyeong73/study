from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score
# sklearn.datasets.fetch_california_housing

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, 
    # shuffle=True,
    random_state=66)


datasets = load_boston()
x = datasets.data
y = datasets.target

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

import time

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()
print(start_time)    #1656032953.0888371
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=0)
end_time = time.time() - start_time


print("걸린 시간 : ", end_time)

"""

verbose 0일 때, 걸린 시간 :  8.863617897033691 / 출력 없다.
verbose 1일 때, 걸린 시간 :  10.5017728805542 / 잔소리 많다.
verbose 2일 때, 걸린 시간 :  9.12317967414856 / 진행바(프로그래스바)가 사라짐.
verbose 3, 4, 5... 일 때, 걸린 시간 :  8.98381233215332 / epoch만 나온다.

"""
