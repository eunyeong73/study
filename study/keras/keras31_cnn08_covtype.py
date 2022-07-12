import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.metrics import r2_score, accuracy_score
import time
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import tensorflow as tf
tf.random.set_seed(66)



#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)   #(581012, 54) (581012,)
print(np.unique(y, return_counts=True))     #[1 2 3 4 5 6 7]



y = pd.get_dummies(y) #섞기 전에 하기


# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)   #(581012, 8)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66)


print(x_train.shape, y_train.shape)  #(464809, 54) (464809, 7)
print(x_test.shape, y_test.shape)   #(116203, 54) (116203, 7)



###################리세이프#######################
x_train = x_train.reshape(464809, 54, 1, 1)
x_test = x_test.reshape(116203, 54, 1, 1)
print(x_train.shape)
print(np.unique(y_train, return_counts=True))


#########################스케일링########################


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#######################스케일링#########################



#모델구성
model = Sequential()
model.add(Conv2D(filters = 200, kernel_size=(3,3),
                   padding='same',
                   input_shape=(54,1,1)))
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
model.add(Dense(7, activation='softmax'))
model.summary()




#3. 컴파일, 훈련

earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
#다중분류는 로스에 categorical_crossentropy만 씀

hist = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 

import time


import datetime
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

print(date)


filepath = './_ModelCheckPoint/k25/08/'
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
                 batch_size=75, validation_split=0.2, 
                 callbacks=[earlyStopping, mcp], 
                 verbose=1) 




end_time = time.time()
print("걸린시간 : ", end_time)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)



print("============================================")

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

y_test = model.predict(x_test)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)


# loss :  0.5375658869743347
# accuracy :  0.7674328684806824



# dropout 후
# loss :  0.5756451487541199
# accuracy :  0.7602729797363281