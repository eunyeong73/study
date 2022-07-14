import numpy as np
from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, Input, Dropout
from sklearn.metrics import r2_score, accuracy_score
import time
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(66)




#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)   #(1797, 64) (1797,)
print(np.unique(y, return_counts=True))     #[0 1 2 3 4 5 6 7 8 9]



from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  #(1797, 10)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66)

#########################스케일링########################


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#######################스케일링#########################



print(x_train.shape, x_test.shape)  #(1437, 64) (360, 64)


x_train = x_train.reshape(1437,64,1)
x_test = x_test.reshape(360,64,1)




# 모델구성
model = Sequential()
model.add(Conv1D(10,2,input_shape=(64,1)))  #(filter, kernel_size, input_shape)
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))




#3. 컴파일, 훈련

earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
#다중분류는 로스에 categorical_crossentropy만 씀

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1000, batch_size=50, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 

import time


import datetime
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

print(date)


filepath = './_ModelCheckPoint/k25/07/'
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


model.save("./_save/keras23_13_save_model.h5")
# model = load_model("./_save/keras23_13_save_model.h5")


end_time = time.time() - start_time
print("걸린시간 : ", end_time)


#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])



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



#  loss :  0.399832159280777
# accuracy :  0.9055555462837219

# dropout 후
# loss :  0.2788817584514618
# accuracy :  0.9333333373069763


# Conv1D
# loss :  0.17543727159500122      
# accuracy :  0.9583333134651184  
# 걸린시간 :  5.3510823249816895
# acc스코어 :  1.0