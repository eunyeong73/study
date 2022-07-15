from re import I
from sympy import O
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers import Conv1D, Input, LSTM, Reshape
# from keras.keras33_cnn_hamsu1_mnist import Conv2D_2, Maxp1
from tensorflow.keras.datasets import mnist
import numpy as np

#Conv2D = 2D 이미지 관련
#Flatten 납작하게 하는 것 쭉 늘려서
#Maxpooling2D 큰 놈 골라내는 것.




#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)   #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)

# reshape 이용해서 (60000, 28, 28)와 (28, 28, 1)을 맞추기. 순서와 위치는 그대로. 
# 하지만 모양은 바꿀 수 있음.


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape)

print(np.unique(y_train, return_counts=True))

#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64)) 
# ----> 다중분류임 (unique를 찍어보았을 때, output이 10, 즉 2 이상이었으므로)





#2. 모델 구성

# model = Sequential()
# model.add(Dense(units=10, input_shape=(3,)))   #(batch_size, input_dim)
# model.summary()
# (input_dim + bias) * units = summary Params 갯수(Dense 모델)


# model.add(Conv2D(filters = 64, kernel_size=(3,3),              # 출력 (N, 28, 28, 64) 패딩이 세임이므로
#                  padding='same',
#                  input_shape=(28, 28, 1))) # (batch_size, rows, cols, channels)  
# model.add(MaxPooling2D())                                      # 출력 (N, 14,14,64)
# model.add(Conv2D(32, (3,3), 
#                  padding='valid',       # padding은 valid가 디폴트
#                  activation='relu'))                           # 출력 (N, 12, 12, 32)
# model.add(Conv2D(7, (3,3), 
#                  padding='valid', activation='relu'))          #(N, 10, 10, 7)
# model.add(Flatten())                                           # 출력 (N, 700) Flatten한 후 차원이 두 개 작아짐
# model.add(Dense(100, activation='relu'))                       # 출력 (N, 100) 
# # 위와 같이 2차원 데이터 후 Conv1D 쓰고 싶으면? 중간에서 "Reshape" 가능
# model.add(Reshape(target_shape=(100, 1)))                      # (N,100,1)
# # 레이어 상에서 하는 reshape. 실질적으로 연산하는 건 아니다.
# model.add(Conv1D(10,kernel_size = 3))                          # (N, 98, 10) 3차원 출력
# # 따라서 LSTM 받을 수 O 입력은 3차원 출력 2차원
# model.add(LSTM(16))                                            # (N, 16)
# # 출력이 2차원이기 때문에 Dense가 받아들일 수 있음
# model.add(Dense(32, activation='relu'))                        # 출력 (N, 32)
# model.add(Dense(10, activation='softmax'))                     # 출력 (N, 10)
# #원핫의 갯수와 10(유닛)의 갯수는 동일함



#함수로 바꾸어보자
input1=Input(shape=(28,28,1))
Conv2D_1 = Conv2D(64, 4, padding = 'same')(input1)
Maxp1 = MaxPooling2D()(Conv2D_1)
Conv2D_2 = Conv2D(32, 3, padding='valid', activation = 'relu')(Maxp1)
Conv2D_3 = Conv2D(7, 3, padding='valid', activation='relu')(Conv2D_2)
flatten = Flatten()(Conv2D_3)
dense1 = Dense(100)(flatten)
reshape = Reshape(target_shape=(100,1))(dense1)
Conv1D_1 = Conv1D(10, 3)(reshape)
LSTM = LSTM(16)(Conv1D_1)
dense2 = Dense(32, activation='relu')(LSTM)
output1 = Dense(1, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1)



model.summary()


# flatten과 reshape는 연산 과정 없이 순서와 내용은 바뀌지 않은 채 모양만 바꾸어 준다



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)


import time
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 

end_time = time.time() - start_time

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)
print('걸린 시간 : ', end_time)


# loss :  0.07294125109910965
# accuracy :  0.9850999712944031


# reshape 후
# loss :  0.0
# accuracy :  0.11349999904632568
# 걸린 시간 :  671.4747502803802

