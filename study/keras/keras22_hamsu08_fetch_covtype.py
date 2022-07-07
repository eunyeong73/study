import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential, Model
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
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)   #(581012, 54) (581012,)
print(np.unique(y, return_counts=True))     #[1 2 3 4 5 6 7]

# (array([1, 2, 3, 4, 5, 6, 7]), 
#  array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))

print(y)
print(y.shape)


y = pd.get_dummies(y) #섞기 전에 하기


# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)   #(581012, 8)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66)

print(y_train)
print(y_test)


#########################스케일링########################


scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#######################스케일링#########################



# #모델구성
# model = Sequential()
# model.add(Dense(5, input_dim=54))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(50, activation='sigmoid'))
# model.add(Dense(7, activation='softmax'))

#다중분류는 softmax(꼭 마지막에 넣어야 함)

input1 = Input(shape=(54,))
dense1 = Dense(5)(input1)
dense2 = Dense(50, activation='relu')(dense1)
dense3 = Dense(10, activation='linear')(dense2)
dense4 = Dense(10, activation='relu')(dense3)
dense5 = Dense(50, activation='sigmoid')(dense4)
output1 = Dense(7, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)
model.summary()

#3. 컴파일, 훈련

earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
#다중분류는 로스에 categorical_crossentropy만 씀

hist = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 

#batch_size의 디폴트가 32

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


# # 스케일링 전
# loss :  0.6768789291381836
# accuracy :  0.7079077363014221



# # RobustScaler
# loss :  0.5375658869743347
# accuracy :  0.7674328684806824


# + 함수형
# loss :  0.5375658869743347
# accuracy :  0.7674328684806824