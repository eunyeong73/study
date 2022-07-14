import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Bidirectional

#1. 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12], 
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])


print(x.shape, y.shape)    #(13, 3) (13,)


x = x.reshape(13,3,1)
print(x.shape)   #(13, 3, 1)



#2. 모델 구성
model = Sequential()
model.add(Bidirectional(SimpleRNN(64, return_sequences=True), input_shape=(3,1)))
model.add(LSTM(10))
model.add(Dense(3, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(1))


#input_shape를 SimpleRNN 안에 넣는 것이 아니라, 바깥으로 빼주어야 한다
#return_sequence는 RNN에서 제공하는 것, 차원이 안 바뀌게 해주는 역할




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=1000)

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1,
                restore_best_weights=True)


#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([50,60,70]).reshape(1,3,1)   #[8,9,10]의 모양 [[[8], [9], [10]]]
result = model.predict(y_pred)

print('loss : ', loss)
print('[50,60,70]의 결과 : ', result)


# BEFORE
# loss :  0.00020856699848081917
# [50,60,70]의 결과 :  [[79.20117]] 


# Bidirectional 이후
# loss :  0.0001588205195730552
# [50,60,70]의 결과 :  [[78.162964]]
