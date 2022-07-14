#데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


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

#########################스케일링########################


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape, x_test.shape)   #(1195, 9) (133, 9)


x_train = x_train.reshape(1195,9,1)
x_test = x_test.reshape(133, 9, 1)




#2. 모델구성
model = Sequential()
model.add(Conv1D(10, 2, input_shape=(9,1)))
model.add(Flatten())
model.add(Dense(39))
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dropout(0.3))
model.add(Dense(40))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=400)



import time


import datetime
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

print(date)


filepath = './_ModelCheckPoint/k25/09/'
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
end_time = time.time() - start_time


model.save("./_save/keras23_15_save_model.h5")
# model = load_model("./_save/keras23_15_save_model.h5")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

#########################스케일링########################

scaler = StandardScaler()
scaler.fit(test_set)
test_set = scaler.transform(test_set)

#test_set이 주어진 따릉이 문제에서는 따로 이와 관련하여 스케일링을 진행해주어야 한다.

#######################스케일링#########################


y_predict = model.predict(x_test)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))    # 루트씌운다.



result = pd.read_csv(path + 'submission.csv', index_col=0)
y_summit = model.predict(test_set)
result['count'] = y_summit
result.to_csv(path + 'submission.csv', index=True)



# #########    summit의 관하여.
y_summit = model.predict(test_set)
print(y_summit.shape)     # (715, 1)

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

print("걸린시간 : ", end_time)


# loss :  35.516090393066406
# RMSE :  46.3098884582757


# dropout 후
# loss :  35.35393142700195
# RMSE :  46.049063129065935

# Conv1D
# loss :  35.546180725097656       
# RMSE :  46.9885192183362
# 걸린시간 :  7.077710151672363   