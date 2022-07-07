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


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, 
    # shuffle=True,
    random_state=66)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



datasets = load_boston()
x = datasets.data
y = datasets.target




# # 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()  # 2022-07-07 17:21:50.245623
date = date.strftime("%m%d_%H%M")  # 0707_1723

print(date)
#자료형 형태로 출력됨 


filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                      filepath= "".join([filepath, 'k24_', date, '_', filename])
                      )

#초 단위로 epoch가 갱신이 될 때마다 저장될 것.



start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, 
                 batch_size=5, validation_split=0.2, 
                 callbacks=[earlyStopping, mcp], 
                 verbose=1) 

end_time = time.time() - start_time

model.save('./_save/keras24_3_save_model.h5')




# 4. 평가, 예측


print("===========================1. 기본 출력=============================")

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)

"""

print("===========================2. load_model 출력=============================")
model2 = load_model('./_save/keras24_3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)

y_predict2 = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2 스코어 :', r2)





print("===========================3. ModelCheckPoint 출력=============================")
model3 = load_model('./_ModelCheckPoint/keras24_ModelCheckPoint3.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss3 : ', loss3)

y_predict2 = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2 스코어 :', r2)


"""