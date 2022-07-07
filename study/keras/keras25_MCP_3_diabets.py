# sklearn.datasets.load_diabetes
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
import numpy as np

from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1.데이터


datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)     #(442, 10) (442,)

print(datasets.feature_names)
print(datasets.DESCR)





from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, 
    # shuffle=True,
    random_state=86)

##################스케일링

scaler = StandardScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#######################스케일링#########################


datasets = load_diabetes()
x = datasets.data
y = datasets.target


input1 = Input(shape=(10,))
dense1 = Dense(5)(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(10)(dense2)
dense4 = Dense(10)(dense3)
dense5 = Dense(10)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import time


import datetime
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

print(date)


filepath = './_ModelCheckPoint/k25/03/'
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


loss :  1720.9794921875
r2스코어 :  0.6568356034764296