#다중 분류

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from pathlib import Path




#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target


from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
y = datasets.target.reshape(-1,1)
oh.fit(y)
y = oh.transform(y).toarray()


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )



scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)  # (120, 4) (30, 4)


###################리세이프#######################
x_train = x_train.reshape(120, 2, 2, 1)
x_test = x_test.reshape(30, 2, 2, 1) 
print(x_train.shape)
print(np.unique(y_train, return_counts=True))
#################################################

print(x_train.shape, x_test.shape)  # (455, 6, 5, 1) (114, 6, 5, 1) 

start_time = time.time()





#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=80, kernel_size=(2,2), strides=1, padding='same', input_shape=(2, 2, 1)))
model.add(Conv2D(100, (1,1),padding='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(90, (1,1),padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(70, (1,1),padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))





#다중분류는 softmax(꼭 마지막에 넣어야 함)



#3. 컴파일, 훈련

import datetime

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filepath = './_ModelCheckPoint/k25/05/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
#                       save_best_only=True, filepath= "".join([filepath, 'k25_',date, '_', filename]))
log = model.fit(x_train, y_train, epochs=100, batch_size=100, callbacks=[Es], validation_split=0.2)





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


from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

y_test = model.predict(x_test)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)


# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

# y_predict = model.predict(x_test)
# y_predict = tf.argmax(y_predict, axis=1)
# y_test = tf.argmax(y_test, axis=1)

# acc_sc = accuracy_score(y_test, y_predict)
# print('acc스코어 : ', acc_sc)



loss :  0.14388753473758698
accuracy :  0.9333333373069763 

