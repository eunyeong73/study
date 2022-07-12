from keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
import numpy as np
from tensorflow.keras.utils import to_categorical



#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)   
print(x_test.shape, y_test.shape)     


print(np.unique(y_train, return_counts=True))


x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)
print(x_train.shape)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print(np.unique(y_train, return_counts=True))

#(array([0., 1.], dtype=float32), array([540000,  60000], dtype=int64))PS C:\study> 


#2. 모델구성
model = Sequential()

model.add(Dense(64, input_shape=(28*28*1,))) 
model.add(Flatten()) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

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


# loss :  0.4555760622024536
# accuracy :  0.8435999751091003

