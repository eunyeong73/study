# sklearn.datasets.fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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


# scaler = StandardScaler()
# # scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(np.min(x_train))    #0.0
# print(np.max(x_train))    #1.0

# print(np.min(x_test))    #-1.3404255319148937
# print(np.max(x_test))    #1.2654320987654315


#######################스케일링#########################


################################스케일링

scaler = MaxAbsScaler()
#scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#######################################



#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=50
          )



#4. 평가, 예측

y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)



# 1. 스케일링 전
# loss :  0.6435720920562744
# r2스코어 :  0.5309822269166493

# # 2. 민맥스 후
# loss :  0.5511255860328674
# r2스코어 :  0.5983548207486957

# # 3. 스탠다드 후
# loss :  0.5460681915283203
# r2스코어 :  0.6020404651444411



# #4. MaxAbsScaler 후
# loss :  0.5893654227256775
# r2스코어 :  0.5704865822857943


# #5. RobustScaler 후
# loss :  0.5692400336265564
# r2스코어 :  0.5851535423983332