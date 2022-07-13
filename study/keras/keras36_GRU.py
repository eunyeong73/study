import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) #x 가 아니라 실질적으로 datasets
# y = ?

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4,5,6,7,8,9,10])

# x의 shape = (행, 열, 몇 개씩 자르는 지!!!) - 암기
# RNN의 shape는 3차원

print(x.shape, y.shape)    #(7, 3) (7,)

x = x.reshape(7,3,1)
print(x.shape)   #(7, 3, 1)



#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape=(3,1)))  #inputs=[batch, timesteps, feature]
# model.add(SimpleRNN(units=10, input_length=3, input_dim=1))
# model.add(SimpleRNN(32))
# input shape 할 때는 항상 행을 뺌, 모델을 구성하는 데 행은 크게 관여하지 않는다
# RNN은 2차원으로 내려오기 때문에 Dense로 받을 수 있다
# SimpleRNN은 연속으로 받을 수 없음.

model.add(GRU(units = 10, input_shape=(3,1)))

## input_shape를 input_length와 input_dim으로 나누어 쓸 수 있다
#가독성을 위해 순서를 지켜 쓰자.
model.add(Dense(70, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(1))

model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #딱 떨어지는 값이 아니기에 회귀-> mse
model.fit(x,y,epochs=1000)

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)



#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1)   #[8,9,10]의 모양 [[[8], [9], [10]]]
result = model.predict(y_pred)

print('loss : ', loss)
print('[8,9,10]의 결과 : ', result)


# loss :  0.0004501897201407701
# [8,9,10]의 결과 :  [[10.858174]]  

