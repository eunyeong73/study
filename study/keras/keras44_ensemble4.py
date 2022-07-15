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




#1. 데이터

x1_datasets= np.array([range(100), range(301, 401)])   #삼성전자 종가, 하이닉스 종가
x1 = np.transpose(x1_datasets)  #가로세로 바꾸기 위해. 선택적.

print(x1.shape)       #(100, 2) 

y1= np.array(range(2001, 2101))    #금리    (100,)
y2= np.array(range(201, 301))    #금리

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1, y2, train_size=0.7, random_state=66)

print(x1_train.shape, x1_test.shape)   #(70, 2) (30, 2)
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




#세 모델을 합친다 (절대 순서 바꾸지 말 것)
# Concatenate
from tensorflow.python.keras.layers import concatenate, Concatenate  #소문자는 함수 대문자는 클래스
merge1 = concatenate([output1], name='merge1')   #list의 append 개념
merge2 = Dense(50, activation='relu', name='mg1')(merge1)
merge3 = Dense(70, name = 'mg2')(merge2)
last_output = Dense(1, name = 'last')(merge3)



#분기
#2-4. output 모델1
output41 = Dense(10)(last_output)
output42 = Dense(10)(output41)
last_output2 = Dense(1)(output42)

#2-5. output 모델2
output51 = Dense(110)(last_output)     #이번에도 last_output을 받아들임
output52 = Dense(110)(output51)
output53 = Dense(110)(output52)
last_output3 = Dense(1)(output53)


model = Model(inputs=[input1], outputs = [last_output2, last_output3])



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
hist = model.fit(x1_train, [y1_train, y2_train], epochs=1000, 
                 batch_size=5, validation_split=0.2, 
                 callbacks=[earlyStopping, mcp], 
                 verbose=1) 
end_time = time.time() - start_time


# 4. 평가, 예측
loss = model.evaluate(x1_test, y1_test)   #두 개 이상은 리스트! 따라서 x1_test와 x2_test를 []로 묶음
loss2 = model.evaluate(x1_test, y2_test)
print('loss : ', loss)
print('loss2 : ', loss2)

print("걸린시간 : ", end_time)

y_predict1, y_predict2 = model.predict([x1_test])

from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y_predict1)
r2_2 = r2_score(y2_test, y_predict2)
print('r2 스코어(y1_test) :', r2_1)
print('r2 스코어(y2_test) :', r2_2)


# loss :  [3209058.25, 43.9569206237793, 3209014.25]
# loss2 :  [3219904.5, 3219176.5, 727.9028930664062]
# 걸린시간 :  5.62761378288269
# r2 스코어(y1_test) : 0.9497284619734491
# r2 스코어(y2_test) : 0.16752324579537903


# 월요일 종일 해서 화요일 개장가(시가 09시)를 맞추는 시험 / 수요일은 종가(오후 3시)
# 월요일 아침에 데이터를 주실 것.(며칠, 언제인지는 나와있지 않음..)
# 소스파일과 가중치를 제출해야 함


