#1. 데이터

from random import random
import numpy as np


x1_datasets= np.array([range(100), range(301, 401)])   #삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201), range(411, 511), range(150, 250)])   #원유, 돈육, 밀
x1 = np.transpose(x1_datasets)  #가로세로 바꾸기 위해. 선택적.
x2 = np.transpose(x2_datasets)

print(x1.shape, x2.shape)       #(100, 2) (100, 3)

y= np.array(range(2001, 2101))    #금리    (100,)


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, train_size=0.7, random_state=66)

print(x1_train.shape, x1_test.shape)   #(70, 2) (30, 2)
print(x2_train.shape, x2_test.shape)   #(70, 3) (30, 3)
print(y_train.shape, y_test.shape)     #(70,) (30,)


#2. 모델 구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input


#2-1. 모델_1
input1 = Input(shape=(2,))   #input_shape를 넣어 주어야 함
dense1 = Dense(70, activation='relu', name='ey1')(input1)
dense2 = Dense(50, activation='relu', name='ey2')(dense1)
dense3 = Dense(45, activation='relu', name='ey3')(dense2)
output1 = Dense(50, activation='relu', name='out_ey1')(dense3)



#2-2. 모델_2
input2 = Input(shape=(3,))   #input_shape를 넣어 주어야 함
dense11 = Dense(140, activation='relu', name='ey11')(input2)
dense12 = Dense(120, activation='relu', name='ey12')(dense11)
dense13 = Dense(80, activation='relu', name='ey13')(dense12)
dense14 = Dense(140, activation='relu', name='ey14')(dense13)
output2 = Dense(100, activation='relu', name='out_ey2')(dense14)


#두 모델을 합친다

from tensorflow.python.keras.layers import concatenate, Concatenate  #소문자는 함수 대문자는 클래스
merge1 = concatenate([output1, output2], name='merge1')   #list의 append 개념
merge2 = Dense(50, activation='relu', name='mg1')(merge1)
merge3 = Dense(70, name = 'mg3')(merge2)
last_output = Dense(1, name = 'last')(merge3)

model = Model(inputs=[input1, input2], outputs=last_output) #두 개 이상은 리스트 -> []

model.summary()    #() 까먹지 말자!!!


# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected 
# to
# ==================================================================================================
# input_2 (InputLayer)            [(None, 3)]          0

# __________________________________________________________________________________________________
# input_1 (InputLayer)            [(None, 2)]          0

# __________________________________________________________________________________________________
# ey11 (Dense)                    (None, 11)           44          input_2[0][0]
# __________________________________________________________________________________________________
# ey1 (Dense)                     (None, 1)            3           input_1[0][0]
# __________________________________________________________________________________________________
# ey12 (Dense)                    (None, 12)           144         ey11[0][0]
# __________________________________________________________________________________________________
# ey2 (Dense)                     (None, 2)            4           ey1[0][0] 

# __________________________________________________________________________________________________
# ey13 (Dense)                    (None, 13)           169         ey12[0][0]
# __________________________________________________________________________________________________
# ey3 (Dense)                     (None, 3)            9           ey2[0][0] 

# __________________________________________________________________________________________________
# ey14 (Dense)                    (None, 14)           196         ey13[0][0]
# __________________________________________________________________________________________________
# out_ey1 (Dense)                 (None, 10)           40          ey3[0][0] 

# __________________________________________________________________________________________________
# out_ey2 (Dense)                 (None, 10)           150         ey14[0][0]
# __________________________________________________________________________________________________
# merge1 (Concatenate)            (None, 20)           0           out_ey1[0][0]
#                                                                  out_ey2[0][0]
# __________________________________________________________________________________________________
# mg1 (Dense)                     (None, 2)            42          merge1[0][0]
# __________________________________________________________________________________________________
# mg3 (Dense)                     (None, 3)            9           mg1[0][0] 

# __________________________________________________________________________________________________
# last (Dense)                    (None, 1)            4           mg3[0][0] 

# ==================================================================================================
# Total params: 814
# Trainable params: 814
# Non-trainable params: 0
# __________________________________________________________________________________________________






#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                      filepath='./_ModelCheckPoint/keras43_ModelCheckPoint.hdf5')

import time
start_time = time.time()
hist = model.fit([x1_train, x2_train], y_train, epochs=500, 
                 batch_size=5, validation_split=0.2, 
                 callbacks=[earlyStopping, mcp], 
                 verbose=1) 
end_time = time.time() - start_time



# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)   #두 개 이상은 리스트 따라서 x1_test와 x2_test를 []로 묶음
print('loss : ', loss)
print("걸린시간 : ", end_time)


y_predict = model.predict([x1_test, x2_test])
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)



# loss :  0.011147699318826199
# 걸린시간 :  6.436637878417969



# loss :  0.017568130046129227
# 걸린시간 :  4.269522190093994
# r2 스코어 : 0.9999798965989839