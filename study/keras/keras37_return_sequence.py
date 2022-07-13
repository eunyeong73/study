import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM


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
model.add(LSTM(units = 10, return_sequences=True, input_shape=(3,1)))
model.add(LSTM(5, return_sequences=False))
model.add(Dense(1))

model.summary()

# LSTM 두 개 이상 엮어야 할 때,

# return_sequences=True로 설정하면 3차원으로 떨어짐. (차원 자체가 하나가 늘어남)
# cf. false인 경우, 차원 때문에 dimension 오류가 나타남.



# _________________________________________________________________       
# Layer (type)                 Output Shape              Param #
# =================================================================       
# lstm (LSTM)                  (None, 3, 10)             480
# _________________________________________________________________       
# lstm_1 (LSTM)                (None, 5)                 320
# _________________________________________________________________       
# dense (Dense)                (None, 1)                 6
# =================================================================       
# Total params: 806
# Trainable params: 806
# Non-trainable params: 0
# _________________________________________________________________      

# 원래는 (N,3,1) ===> (N,3,10)
# 가장 끝에 필터 부분만 1에서 10으로 바뀐 것.




"""

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=700)



from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1,
                restore_best_weights=True)



#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([50,60,70]).reshape(1,3,1)   #[8,9,10]의 모양 [[[8], [9], [10]]]
result = model.predict(y_pred)

print('loss : ', loss)
print('[50,60,70]의 결과 : ', result)



# GRU
# loss :  5.269555185805075e-05
# [50,60,70]의 결과 :  [[78.48957]]


# loss :  0.025023754686117172
# [50,60,70]의 결과 :  [[78.26048]]

# 데이터가 적은 건 GRU가 더 좋다


"""