from tkinter import SE
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D, Flatten
# from tensorflow.keras.layers import Bidirectional

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) #x 가 아니라 실질적으로 datasets
# y = ?

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4,5,6,7,8,9,10])

# x의 shape = (행, 열, 몇 개씩 자르는 지!!!) - 암기
# RNN의 shape는 3차원

print(x.shape, y.shape)    #(7, 3) (7,)
x = x.reshape(7,3,1)
print(x.shape)   


#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(64, input_shape=(3,1)))
# model.add(LSTM(10, input_shape=(3,1), return_sequences=False))
model.add(Conv1D(10, 2, input_shape=(3,1)))  #(filter, kernel_size, input_shape)
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))

model.summary() #LSTM : 517 / conv1D = 97


# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 3, 10)             120

#  bidirectional (Bidirectiona  (None, 10)               160
#  l)

#  dense (Dense)               (None, 75)                825

#  dense_1 (Dense)             (None, 1)                 76

# =================================================================
# Total params: 1,181
# Trainable params: 1,181
# Non-trainable params: 0
# _________________________________________________________________

#bidirectional의 파라미터가 160인 이유? 
# ==> [ 10 + 5(SimpleRNN) + 1(Biase) ] * 2(bidirection) * 5(simpleRNN)




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #딱 떨어지는 값이 아니기에 회귀-> mse
model.fit(x,y,epochs=2000)

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1)   #[8,9,10]의 모양 [[[8], [9], [10]]]
result = model.predict(y_pred)

print('loss : ', loss)
print('[8,9,10]의 결과 : ', result)


# BEFORE
# loss :  0.0004501897201407701
# [8,9,10]의 결과 :  [[10.858174]]  


# Bidirectional
# loss :  2.7026211682823487e-05
# [8,9,10]의 결과 :  [[10.639596]]


#Conv1D
# loss :  3.421172323214705e-08
# [8,9,10]의 결과 :  [[10.999808]]

