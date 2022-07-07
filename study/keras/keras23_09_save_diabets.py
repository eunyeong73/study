# sklearn.datasets.load_diabetes
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
import numpy as np

from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1.데이터


datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)     #(442, 10) (442,)

print(datasets.feature_names)
print(datasets.DESCR)





from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, 
    # shuffle=True,
    random_state=86)

##################스케일링

scaler = StandardScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#######################스케일링#########################


datasets = load_diabetes()
x = datasets.data
y = datasets.target



# #2. 모델구성
# model = Sequential()
# model.add(Dense(5, input_dim=10))
# model.add(Dense(3))
# model.add(Dense(3))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))


input1 = Input(shape=(10,))
dense1 = Dense(5)(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(10)(dense2)
dense4 = Dense(10)(dense3)
dense5 = Dense(10)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=5)

model.save("./_save/keras23_9_save_model.h5")

# model = load_model("./_save/keras23_9_save_model.h5")

#4. 평가, 예측

y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


loss :  1720.9794921875
r2스코어 :  0.6568356034764296