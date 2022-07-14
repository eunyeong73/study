from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, Flatten
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


print(x_train.shape)  #(404, 13)
print(x_test.shape)  #(102, 13)

x_train = x_train.reshape(404,13,1)
x_test = x_test.reshape(102, 13, 1)



# # 2. 모델구성
model = Sequential()
model.add(Conv1D(10, 2, input_shape=(13,1)))  #(filter, kernel_size, input_shape)
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                      filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, 
                 batch_size=5, validation_split=0.2, 
                 callbacks=[earlyStopping, mcp], 
                 verbose=1) 
end_time = time.time() - start_time



# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("걸린시간 : ", end_time)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)


# Conv1D 
# loss :  11.628548622131348
# 걸린시간 :  8.345824003219604
# r2 스코어 : 0.8608741469150711