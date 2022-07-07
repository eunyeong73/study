# sklearn.datasets.fetch_california_housing
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from yaml import load
from scipy import rand
from sklearn import datasets


#1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


print(x)
print(y)
print(x.shape, y.shape)        # (20640, 8) (20640,)

print(datasets.feature_names)
print(datasets.DESCR)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, 
    # shuffle=True,
    random_state=66)


#########################스케일링########################


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)




#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(1))




import time

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()  # 2022-07-07 17:21:50.245623
date = date.strftime("%m%d_%H%M")  # 0707_1723

print(date)
#자료형 형태로 출력됨 


filepath = './_ModelCheckPoint/k25/02/'
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

y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)



loss :  0.5582231879234314
r2스코어 :  0.5931821607008438