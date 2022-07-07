import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터    
x = np.array([range(10), range(21, 31), range(201, 211)])
# print(range(10))
# for i in range(10):
# print(i)
    
print(x.shape)  #(3, 10)
x = np.transpose(x)
print(x.shape)  #(10, 3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])
y = np.transpose(y)
print(y.shape)


#2.모델구성
model=Sequential()
# model.add(Dense(10, input_dim=3))
model.add(Dense(10, input_shape=(3,)))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))
model.summary()



# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 10)                40
# _________________________________________________________________
# dense_1 (Dense)              (None, 5)                 55
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 18
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 4
# =================================================================
# Total params: 117
# Trainable params: 117
# Non-trainable params: 0
# _________________________________________________________________
# PS C:\study>
