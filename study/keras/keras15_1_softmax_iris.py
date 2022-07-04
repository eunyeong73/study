#다중 분류



import numpy as np
from sklearn.datasets import load_iris
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
import time
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping


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




#모델구성
model = Sequential()
model.add(Dense(5, input_dim=4))
model.add(Dense(5, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

#다중분류는 softmax(꼭 마지막에 넣어야 함)



#3. 컴파일, 훈련

earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
#다중분류는 로스에 categorical_crossentropy만 씀
hist = model.fit(x_train, y_train, epochs=1000, batch_size=50, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 


end_time = time.time()
print("걸린시간 : ", end_time)


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



#######################

# y_predict = model.predict(x_test)
# y_predict = tf.argmax(y_predict, axis=1) #1은 행끼리 비교, 0은 열끼리 비교
# print(y_predict)


# y_test = tf.argmax(y_test, axis=1)
# print(y_test)

# acc_sc = accuracy_score(y_test, y_predict)
# print('acc 스코어 : ', acc_sc)

#######################






# 걸린시간 :  1656899481.2329924
# accuracy :  0.7820217152563335

# # ######원핫인코딩 후
# loss :  1.095748782157898
# accuracy :  0.3333333432674408

# ####더미 후
# loss :  0.1483854204416275
# accuracy :  0.9666666388511658