from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import time


#이상치를 잘 잡아내는 스케일러를 골라낼 것.



#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets['target']


#잘못됨
# print(np.min(x))    #0.0
# print(np.max(x))    #711.0
# x = (x - np.min(x)) / (np.max(x) - np.min(x))
# print(x[:10]) 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, 
    # shuffle=True,
    random_state=66)


###############################스케일링

# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

###################################



#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))





#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')



import datetime
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

print(date)


filepath = './_ModelCheckPoint/k25/01'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                      filepath= "".join([filepath, 'k25_', date, '_', filename])
                      )

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


