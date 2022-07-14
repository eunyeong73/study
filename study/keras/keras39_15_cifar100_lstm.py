from keras.datasets import cifar100
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv2D, Flatten, Dropout
import numpy as np
from tensorflow.keras.utils import to_categorical



#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)   
print(x_test.shape, y_test.shape)     


print(np.unique(y_train, return_counts=True))


x_train = x_train.reshape(50000, 32*32, 3)
x_test = x_test.reshape(10000, 32*32, 3)
print(x_train.shape)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print(np.unique(y_train, return_counts=True))

#(array([0., 1.], dtype=float32), array([540000,  60000], dtype=int64))PS C:\study> 


#2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(32*32, 3))) 
model.add(Flatten()) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 



#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)



#LSTM 전
# loss :  4.605264663696289
# accuracy :  0.009999999776482582