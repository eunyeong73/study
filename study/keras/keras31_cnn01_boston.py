from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time


# sklearn.datasets.fetch_california_housing

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


# print(np.unique(x_train, return_counts=True))

print(x_train.shape, y_train.shape)  #(404, 13) (404,)
print(x_test.shape, y_test.shape)   #(102, 13) (102,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# (404, 13) (102, 13)

###################리세이프#######################
x_train = x_train.reshape(404, 13, 1, 1)
x_test = x_test.reshape(102, 13, 1, 1)
print(x_train.shape)
print(np.unique(y_train, return_counts=True))
#################################################




# 2. 모델구성
# input1 = Input(input_shape=(13,1,1))
# dense1 = Dense(32)(input1)
# dense2 = Dense(32)(dense1)
# drop1 = Dropout(0.2)(dense2)
# dense3 = Dense(32)(drop1)
# drop2 = Dropout(0.1)(dense3)
# dense4 = Dense(32)(drop2)
# drop3 = Dropout(0.1)(dense4)
# dense5 = Dense(32)(dense4)
# output1 = Dense(1)(dense5)
# model = Model(inputs=input1, outputs=output1)

#2. 모델구성

model = Sequential()
model.add(Conv2D(filters = 200, kernel_size=(3,3),
                   padding='same',
                   input_shape=(13,1,1)))
 # padding : 커널 사이즈대로 자르다보면 가생이는 중복되서 분석을 못해주기때문에 행렬을 키워주는것, 패딩을 입혀준다? 이런 너낌
 # kernel_size = 이미지 분석을위해 2x2로 잘라서 분석하겠다~
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
model.add(Dense(1))
model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, 
                 batch_size=5, validation_split=0.2, 
                 callbacks=[earlyStopping], 
                 verbose=1) 

end_time = time.time() - start_time



# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)



y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)


# loss :  23.304153442382812
# r2 스코어 : 0.721185248040112