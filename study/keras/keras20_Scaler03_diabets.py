# sklearn.datasets.load_diabetes
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

# scaler = StandardScaler()
#scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train))    #0.0
print(np.max(x_train))    #1.0

print(np.min(x_test))    #-1.3404255319148937
print(np.max(x_test))    #1.2654320987654315


#######################스케일링#########################


datasets = load_diabetes()
x = datasets.data
y = datasets.target



#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=5)



#4. 평가, 예측

y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)



# 스케일링 전
# loss :  1674.0389404296875
# r2스코어 :  0.6661955716435914


# MaxAbsScaler
# loss :  1699.51220703125
# r2스코어 :  0.6611161659824072


# #민맥스
# loss :  1667.4312744140625
# r2스코어 :  0.6675131567305108


# 스탠다드
# loss :  1720.8211669921875
# r2스코어 :  0.6568671323943486


# RobustScaler
# loss :  1708.8218994140625
# r2스코어 :  0.6592598026482427