import numpy as np

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
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

# model=Sequential()
# # model.add(Dense(10, input_dim=3))
# model.add(Dense(10, input_shape=(3,)))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(1))


##############################함수형 모델


input1 = Input(shape=(3,))
dense1 = Dense(10)(input1)
dense2 = Dense(5, activation='relu')(dense1)
dense3 = Dense(3, activation='sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)
model.summary()



#input 명시, input shape 정의, dense 구성함, 전 레이어를 뒤에 명시.
#레이어 재사용 가능
#함수형은 그냥 Model 인풋하면 ok



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x, y, epochs=100, batch_size=1)









