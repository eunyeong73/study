from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist, cifar10
import numpy as np

from sklearn import datasets
from tensorflow.keras.utils import to_categorical



    
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)    #(10000, 32, 32, 3) (10000, 1)



y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)


print(x_train.shape)
print(np.unique(y_train, return_counts=True))





#2. 모델 구성

model = Sequential()


model.add(Conv2D(filters = 64, kernel_size=(2,2),  
                 padding='same',
                 input_shape=(32, 32, 3))) # (batch_size, rows, cols, channels)  
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), 
                 padding='valid',   #디폴트
                 activation='relu')) 
model.add(Conv2D(32, (3,3), padding='valid', activation='relu'))
model.add(Flatten()) 
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()




#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 




#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)



# loss :  1.242660403251648
# accuracy :  0.5611000061035156

