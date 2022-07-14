from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np


from sklearn import datasets
from tensorflow.keras.utils import to_categorical



    
#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)   #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)


print(np.unique(y_train, return_counts=True))

#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],dtype=int64))

#원 핫 전에 하기.



y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(60000, 28*28, 1)
x_test = x_test.reshape(10000, 28*28, 1)

print(x_train.shape)



#2. 모델 구성

model = Sequential()

model.add(LSTM(units = 5, input_shape=(28*28,1)))
# model.add(Conv2D(filters = 64, kernel_size=(3,3),   # 출력 (N, 28, 28, 64) 패딩이 세임이므로
# #                  padding='same')) # (batch_size, rows, cols, channels)  

# model.add(MaxPooling2D())
model.add(Dropout(0.3))
# model.add(Conv2D(64, (2,2), 
#                  padding='valid',   #디폴트
#                  activation='relu'))  # 출력 (N, 3, 3, 7)
model.add(Flatten())  # 출력 (N, 63)
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

model.summary()



#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 




#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)



# LSTM 전
# loss :  0.05844346806406975
# accuracy :  0.891700029373169

# LSTM 후
