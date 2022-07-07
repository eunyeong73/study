#다중 분류



import numpy as np
from sklearn.datasets import load_iris
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import r2_score, accuracy_score
import time
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(66)



#1. 데이터
datasets = load_iris()
print(datasets.DESCR)    #(150, 4)
print(datasets.feature_names)
x=datasets['data']
y=datasets['target']
print(x)
print(y)
print(x.shape, y.shape)     #(150, 4) (150,)

#(150,)를 (150,3)으로 만들어주어야 하는데, 원핫인코딩을 사용할 것.add(

print("y의 라벨값: ", np.unique(y)) #y의 라벨값:  [0 1 2]


###########################

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)   #(150,3)

##########################

# y = pd.get_dummies(y)

###########################

from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
y = datasets.target.reshape(-1,1)
oh.fit(y)
y = oh.transform(y).toarray()

#######################


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66)

print(y_train)
print(y_test)

#########################스케일링########################

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#######################스케일링#########################


# #모델구성
# model = Sequential()
# model.add(Dense(5, input_dim=4))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='sigmoid'))
# model.add(Dense(3, activation='softmax'))


input1 = Input(shape=(4,))
dense1 = Dense(5)(input1)
dense2 = Dense(5, activation='relu')(dense1)
dense3 = Dense(10, activation='sigmoid')(dense2)
dense4 = Dense(10, activation='sigmoid')(dense3)
dense5 = Dense(5, activation='sigmoid')(dense4)
dense6 = Dense(5, activation='relu')(dense5)
output1 = Dense(3, activation='softmax')(dense6)
model = Model(inputs=input1, outputs=output1)



#다중분류는 softmax(꼭 마지막에 넣어야 함)



#3. 컴파일, 훈련

earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
#다중분류는 로스에 categorical_crossentropy만 씀
hist = model.fit(x_train, y_train, epochs=1000, batch_size=50, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 


import time


import datetime
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

print(date)


filepath = './_ModelCheckPoint/k25/05/'
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



model.save("./_save/keras23_11_save_model.h5")

# model = load_model("./_save/keras23_11_save_model.h5")



#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)


results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])


# print("==================y_test[:5]===============")
# print(y_test[:5])
# print("==================y_pred====================")
# y_pred = model.predict(x_test[:5])
# print(y_pred)



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






# loss :  0.48687151074409485
# accuracy :  0.6333333253860474


