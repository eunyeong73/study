#데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
print(train_set)
print(train_set.shape)      #[1459 rows x 10 columns] (1459, 10)

test_set = pd.read_csv(path + 'test.csv',       #예측에서 사용할 예정
                       index_col=0)

print(test_set)
print(test_set.shape)       #(715, 9)

print(train_set.columns)
print(train_set.info())
print(train_set.describe())


# ##### 결측치 처리를 할 것 1. 제거 #####
print(train_set.isnull().sum())   #널의 갯수를 구한다.
train_set = train_set.dropna()
print(train_set.isnull().sum())
print(train_set.shape)       #((1328, 10))
# ####################################


# from pandas import DataFrame
# DataFrame.fillna(0)


x = train_set.drop(['count'], axis=1)
print(x)   #[1459 rows x 9 columns]
print(x.columns)
print(x.shape)   #(1459, 9)

y = train_set['count']
print(y)
print(y.shape)      #(1459,) 스칼라 1459개, 백터는 1개


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.9, shuffle=True, random_state=31)


#2. 모델구성
model = Sequential()
model.add(Dense(6, input_dim=9))
model.add(Dense(10))
model.add(Dense(39))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=700, batch_size=400)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))    # 루트씌운다.

test_set=test_set.fillna(0)

# #얘는 결측치를 0으로 채워넣는 것.


result = pd.read_csv(path + 'submission.csv', index_col=0)
y_summit = model.predict(test_set)
result['count'] = y_summit
result.to_csv(path + 'submission.csv', index=True)






# #########    summit의 관하여.
y_summit = model.predict(test_set)
print(y_summit.shape)     # (715, 1)

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)






########## .to_csv()를 사용해서
### submission.csv를 완성할 것


# loss :  35.030609130859375
# RMSE :  44.07803578076409

# random state 를 31로 수정, batch_size 올려서 수정함


#y_predict = model. predict(test_set)