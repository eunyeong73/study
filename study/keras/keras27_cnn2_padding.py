from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

#Conv2D = 2D 이미지 관련
#Flatten 납작하게 하는 것 쭉 늘려서
#Maxpooling2D 큰 놈 골라내는 것.



model = Sequential()
# model.add(Dense(units=10, input_shape=(3,)))   #(batch_size, input_dim)
# model.summary()
# (input_dim + bias) * units = summary Params 갯수(Dense 모델)

  
model.add(Conv2D(filters = 64, kernel_size=(3,3),   # 출력 (N, 28, 28, 64) 패딩이 세임이므로
                 padding='same',
                 input_shape=(28, 28, 1))) # (batch_size, rows, cols, channels)  

model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), 
                 padding='valid',   #디폴트
                 activation='relu'))  # 출력 (N, 3, 3, 7)
model.add(Flatten())  # 출력 (N, 63)

#padding의 의의 : kernel_size에 상관없이. 원래 Input_shape를 그대로 유지하고 싶을 때 쓰는 거.



#4차원을 2차원으로 만들기.   (N,4,3,2)를 (N,4*3*2)로 만듦. 위에서는 3*3*7

#예를 들어 (N,4,3,2)를 쭉 늘려서 쫙 핀 데이터(24, N장)로 만들면 (N,24)
#대신 순서와 데이터를 바꾸면 안 됨. 따라서 Flatten 사용(길게 늘이기 위해서)

model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
#(kernel_size * channels + bias) * filters = summary Params 갯수(CNN 모델)



#kernel_size : 이미지 자르는 규격
#input_shape : 장 수, 가로, 세로, 1(흑백)/2(컬러)





