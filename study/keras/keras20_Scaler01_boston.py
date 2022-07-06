from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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


################################스케일링

# scaler = StandardScaler()
# #scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(np.min(x_train))    #0.0
# print(np.max(x_train))    #1.0

# print(np.min(x_test))    #-1.3404255319148937
# print(np.max(x_test))    #1.2654320987654315

# # a=0.1   #0.0
# # b=0.2   #1.0000000000000002
# # print(a+b)  #0.30000000000000004

#######################################


#scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train))    #0.0
print(np.max(x_train))    #1.0

print(np.min(x_test))    #-1.3404255319148937
print(np.max(x_test))    #1.2654320987654315


####################################



#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)



#4. 평가, 예측
y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)




# #보스턴 세 가지 비교
# 1. 스케일러 하기 전
# 2. 민맥스
# 3. 스탠다드 -> 이렇게 세 가지 비교해보기.

# #1. 스케일링 전 
# loss :  21.56123924255371
# r2스코어 :  0.7390220920908033


#2. 민맥스 후 
# #loss :  21.85053825378418
# r2스코어 :  0.7355204019766544


#3.스탠다드 후
# loss :  17.013078689575195
# r2스코어 :  0.7940731489379143


#4. MaxAbsScaler 후
# loss :  16.555875778198242
# r2스코어 :  0.7996071450582565


# #5. RobustScaler 후
# loss :  23.212514877319336
# r2스코어 :  0.7190350055640977