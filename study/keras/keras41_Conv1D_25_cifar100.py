from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.datasets import mnist, cifar100
import numpy as np

from sklearn import datasets
from tensorflow.keras.utils import to_categorical



    
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)   #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)     #(10000, 32, 32, 3) (10000, 1)





#################################유니크#############################
print(np.unique(y_train, return_counts=True))


# (array([ 0,  1,  2,  3,  4,  5,  6,  7, 
#  8,  9, 10, 11, 12, 13, 14, 15, 16,     
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,      
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,      
#        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,      
#        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,      
#        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))

#################################유니크





y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(50000, 32*32, 3)
x_test = x_test.reshape(10000, 32*32, 3)



#2. 모델 구성

model = Sequential()

  
model.add(Conv1D(filters = 64, kernel_size = 3, 
                 padding='same',
                 input_shape=(32*32, 3)))
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(Conv1D(70, 2, 
                 padding='valid',   
                 activation='relu')) 
model.add(Conv1D(85, 2, 
                 padding='valid', activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten()) 
model.add(Dense(100, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()




#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.3, 
                 callbacks=[earlyStopping], verbose=1) 
end_time = time.time() - start_time



#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)
print("걸린시간 : ", end_time)


#Conv2D
# loss :  4.605504035949707
# accuracy :  0.009999999776482582

#Conv1D
# loss :  4.605262279510498        
# accuracy :  0.009999999776482582 
# 걸린시간 :  759.5793659687042 