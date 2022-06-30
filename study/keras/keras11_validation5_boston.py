from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
# sklearn.datasets.fetch_california_housing

#1. 데이터
datasets = load_boston() #변수 만들기
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)     #(506, 13) (506,)

print(datasets.feature_names)
print(datasets.DESCR)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, train_size=0.8)

#얘는 한줄로 있어도 괜찮음. 아래 fit에서 validation_split을 추가할 거니까.

datasets = load_boston()
x = datasets.data
y = datasets.target

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(5))
model.add(Dense(190))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(75))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=20, validation_split=0.25)


#4. 평가, 예측
y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# loss :  17.007692337036133
# r2스코어 :  0.7941383640471288



#########################validation 적용 후 

# loss :  19.434534072875977
# r2스코어 :  0.7674820334470275  