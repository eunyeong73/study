import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법을 찾아라.
x_train = x[:7] 
x_test = x[7:]
y_train = y[:7]
y_test = y[7:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, 
    train_size=0.7, 
    # shuffle=True,
    random_state=66
)

print(x_train) #[2 7 6 3 4 8 5]
print(x_test) #[ 1  9 10]
print(y_train) 
print(y_test) 

# x_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10])
# y_train = np.array([1,2,3,4,5,6,7])
# y_test = np.array([8,9,10])


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=400, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('11의 예측값 : ', result)

# x sizes: 3
# y sizes: 7
# Make sure all arrays contain the same number of samples.

# loss :  3.031649096259942e-13
# 11의 예측값 :  [[11.000001]]