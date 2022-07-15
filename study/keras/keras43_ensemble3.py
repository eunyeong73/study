#1. 데이터

from random import random
import numpy as np


###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감###################



x1_datasets= np.array([range(100), range(301, 401)])   #삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201), range(411, 511), range(150, 250)])   #원유, 돈육, 밀
x3_datasets= np.array([range(100, 200), range(1301, 1401)])   # 우리반 IQ, 우리반 키
x1 = np.transpose(x1_datasets)  #가로세로 바꾸기 위해. 선택적.
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

print(x1.shape, x2.shape, x3.shape)       #(100, 2) (100, 3)  (100, 2)

y1= np.array(range(2001, 2101))    #금리    (100,)
y2= np.array(range(201, 301))    #금리

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, x3, y1, y2, train_size=0.7, random_state=66)

print(x1_train.shape, x1_test.shape)   #(70, 2) (30, 2)
print(x2_train.shape, x2_test.shape)   #(70, 3) (30, 3)
print(x3_train.shape, x3_test.shape)   #(70, 2) (30, 2)
print(y1_train.shape, y1_test.shape)    #(70,) (30,)
print(y2_train.shape, y2_test.shape)    #(70,) (30,)





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
dense11 = Dense(14, activation='relu', name='ey11')(input2)
dense12 = Dense(12, activation='relu', name='ey12')(dense11)
dense13 = Dense(10, activation='relu', name='ey13')(dense12)
dense14 = Dense(14, activation='relu', name='ey14')(dense13)
output2 = Dense(50, activation='relu', name='out_ey2')(dense14)



#2-3. 모델_3
input3 = Input(shape=(2,))   #input_shape를 넣어 주어야 함
dense31 = Dense(10, activation='relu', name='ey31')(input3)
dense32 = Dense(10, activation='relu', name='ey32')(dense31)
dense33 = Dense(15, activation='relu', name='ey33')(dense32)
dense34 = Dense(10, activation='relu', name='ey34')(dense33)
output3 = Dense(10, activation='relu', name='out_ey3')(dense33)



#세 모델을 합친다 (절대 순서 바꾸지 말 것)
# Concatenate
from tensorflow.python.keras.layers import concatenate, Concatenate  #소문자는 함수 대문자는 클래스
merge1 = concatenate([output1, output2, output3], name='merge1')   #list의 append 개념
merge2 = Dense(50, activation='relu', name='mg1')(merge1)
merge3 = Dense(70, name = 'mg2')(merge2)
last_output1 = Dense(1, name = 'last')(merge3)


merge11 = concatenate([output1, output2, output3], name='merge11')   #list의 append 개념
merge12 = Dense(50, activation='relu', name='mg11')(merge1)
merge13 = Dense(70, name = 'mg12')(merge2)
last_output2 = Dense(1, name = 'last2')(merge3)

model = Model(inputs=[input1, input2, input3], outputs=[last_output1, last_output2]) #두 개 이상은 리스트 -> []

model.summary()    #() 까먹지 말자!!!





#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

save_filepath = './_ModelCheckPoint/' + current_name + '/'
load_filepath = './_ModelCheckPoint/' + current_name + '/'

# model = load_model(load_filepath + '0708_1753_0011-0.0731.hdf5')

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([save_filepath, date, '_', filename])
                      )


import time
start_time = time.time()
hist = model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=1000, 
                 batch_size=5, validation_split=0.2, 
                 callbacks=[earlyStopping, mcp], 
                 verbose=1) 
end_time = time.time() - start_time



# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y1_test)   #두 개 이상은 리스트 따라서 x1_test와 x2_test를 []로 묶음
loss2 = model.evaluate([x1_test, x2_test, x3_test], y2_test)
print('loss : ', loss)
print('loss2 : ', loss2)

print("걸린시간 : ", end_time)

y_predict1, y_predict2 = model.predict([x1_test, x2_test, x3_test])

from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y_predict1)
r2_2 = r2_score(y2_test, y_predict2)
print('r2 스코어(y1_test) :', r2_1)
print('r2 스코어(y2_test) :', r2_2)


# loss :  [3239911.5, 0.05818764492869377, 3239911.5]
# loss2 :  [3239900.25, 3239900.25, 0.032691437751054764]
# 걸린시간 :  9.054180145263672
# r2 스코어(y1_test) : 0.9999334401738946
# r2 스코어(y2_test) : 0.9999626119600765