from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import pandas as pd
from sklearn import datasets



#1. 데이터
datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data    #(569, 30) 
y = datasets.target  #(569,)
print(x.shape, y.shape)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=31, train_size=0.8)

print(x_train.shape, y_train.shape)  #(455, 30) (455,)
print(x_test.shape, y_test.shape)   #(114, 30) (114,)

#########################스케일링########################

#scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler = StandardScaler()
# # scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


###################리세이프#######################
x_train = x_train.reshape(455, 6, 5, 1)
x_test = x_test.reshape(114, 6, 5, 1)
print(x_train.shape)
print(np.unique(y_train, return_counts=True))
#################################################




#2. 모델구성
model = Sequential()
model.add(Conv2D(filters = 200, kernel_size=(3,3),
                   padding='same',
                   input_shape=(6,5,1)))
model.add(Conv2D(filters = 200, kernel_size=(3,3),
                   padding='same'))
model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=40, mode='min', verbose=1,
                restore_best_weights=True)
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse']) 


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




# loss :  [0.10504604876041412, 0.9649122953414917, 0.03183398023247719]    
# acc스코어 :  0.9649122807017544