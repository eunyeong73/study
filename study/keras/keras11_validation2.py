from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

# #1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))


#슬라이싱하기
x_train = x[:10] 
y_train = y[:10]
x_test = x[10:13]
y_test = y[10:13]
x_val = y[13:17]
y_val = y[13:17]

print(x_train) 
print(y_train) 
print(x_test) 
print(y_test) 
print(y_val) 
print(x_val) 


##직접 복습

'''
# x_train = np.array(range(1, 11)) #훈련
# y_train = np.array(range(1, 11))
# x_test = np.array([11,12,13]) #평가 (evaluate)
# y_test = np.array([11,12,13])
# x_val = np.array([14,15,16]) #검증
# y_val = np.array([14,15,16])

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)


#val은 문제집 개념으로 실전(test_set)에 들어가기 전에 한 번 확인하는 용도.

'''