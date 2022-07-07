# sklearn.datasets.fetch_california_housing
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
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


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)




# # #2. 모델구성
# # model = Sequential()
# # model.add(Dense(5, input_dim=8))
# # model.add(Dense(50))
# # model.add(Dense(30))
# # model.add(Dense(50))
# # model.add(Dense(5))
# # model.add(Dense(1))



# input1 = Input(shape=(8,))
# dense1 = Dense(5)(input1)
# dense2 = Dense(50)(dense1)
# dense3 = Dense(30)(dense2)
# dense4 = Dense(50)(dense3)
# dense5 = Dense(5)(dense4)
# output1 = Dense(1)(dense5)
# model = Model(inputs=input1, outputs=output1)



# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=100, batch_size=50
#           )

#model.save("./_save/keras23_8_save_model.h5")

model = load_model("./_save/keras23_8_save_model.h5")

#4. 평가, 예측

y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)



loss :  0.5582231879234314
r2스코어 :  0.5931821607008438