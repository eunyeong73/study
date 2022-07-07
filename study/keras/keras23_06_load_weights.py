from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MinMaxScaler

# sklearn.datasets.fetch_california_housing

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)

print(datasets.feature_names)
print(datasets.DESCR)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, 
    # shuffle=True,
    random_state=66)



scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



datasets = load_boston()
x = datasets.data
y = datasets.target

# # 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

# model = load_model("./_save/keras23_1_save_model.h5")

# model.save_weights("./_save/keras23_5_save_weights1.h5")

model.load_weights("./_save/keras23_5_save_weights2.h5")



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# from tensorflow.python.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
#                 restore_best_weights=True)

# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=270, 
#                  batch_size=5, validation_split=0.2, 
#                  verbose=1) 

# model.save_weights("./_save/keras23_5_save_mode2.h5")




# end_time = time.time() - start_time

# train_size의 비율 0.2이므로 validation은 전체에서 16%

# model = load_model("./_save/keras23_3_save_model.h5")


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)



y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)



# loss :  11.814218521118164
# 걸린시간 : 17.40452003479004
# r2 스코어 : 0.8586527493840406

#model 다음에 세이브하면 모델까지 저장됨
#fit 다음 SAVE_MODEL 은 모델과 가중치까지 저장됨

# loss :  8.854361534118652
# r2 스코어 : 0.8940649802304067


# # 가중치 저장 후-------------------------------
# loss :  13.247673034667969
# r2 스코어 : 0.8415026592609032


# loss :  13.247673034667969
# r2 스코어 : 0.8415026592609032
