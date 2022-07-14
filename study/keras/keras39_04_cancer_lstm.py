import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
from sklearn.preprocessing import MaxAbsScaler, RobustScaler



#1. 데이터
datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data    #(569, 30) 
y = datasets.target  #(569,)
print(x.shape, y.shape)
print(y)


x = x.reshape(569,30,1)
print(x.shape)



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=31, train_size=0.8)

#########################스케일링########################

# #scaler = MaxAbsScaler()
# scaler = RobustScaler()
# # scaler = StandardScaler()
# # # scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)




#2. 모델구성
model = Sequential()
model.add(LSTM(units = 10, input_shape=(30,1)))
model.add(Dense(5, activation='linear'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='linear'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu')) # 히든에서만 쓸 수 있음. 중간에서만 사용. 평타는 치는 좋은 애 adam.같은 애.
model.add(Dense(30, activation='linear'))
model.add(Dense(1, activation='sigmoid')) #이진 분류에서는 마지막에는 sigmoid 활성화 함수 사용


#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=40, mode='min', verbose=1,
                restore_best_weights=True)
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse']) 
# 이진 분류에서는 loss=binary_crossentropy 사용
# 평가지표 accuracy 넣음
# 두 개 이상은 리스트. 위가 리스트 형태니까 넣어줘도 가능함.
# 메트릭스는 추가지표 넣을 때 쓰는 것.

import time


import datetime
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

print(date)


filepath = './_ModelCheckPoint/k25/04'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                      filepath= "".join([filepath, 'k25_', date, '_', filename])
                      )

#초 단위로 epoch가 갱신이 될 때마다 저장될 것.



start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, 
                 batch_size=5, validation_split=0.2, 
                 callbacks=[earlyStopping, mcp], 
                 verbose=1) 

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
y_predict = np.round(y_predict, 0)
#**반올림 하는 함수

#####[과제 1. accuracy score 완성하기]
from sklearn.metrics import r2_score, accuracy_score
#r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)


#loss :  [0.2671935558319092, 0.8947368264198303, 0.07989775389432907]
# acc스코어 :  0.8947368421052632