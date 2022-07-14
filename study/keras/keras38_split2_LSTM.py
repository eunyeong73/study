import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM



#1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96,106))    #결과 값은 100~106까지
size = 5    #x는 4개, y는 1개

print(x_predict.shape)   #(10,)


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)   #(6,5)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape)   #(96, 4) (96,)


x = x.reshape(96,4,1)
print(x.shape)     #(96, 4, 1)


ccc = split_x(x_predict, 4)
print(ccc.shape)   #(7, 4)




#2. 모델 구성
model = Sequential()
model.add(LSTM(units = 10, input_shape=(4,1)))
model.add(Dense(70, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(47, activation='relu'))
model.add(Dense(1))

model.summary()





#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #딱 떨어지는 값이 아니기에 회귀-> mse
model.fit(x,y,epochs=100)

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)





#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = ccc.reshape(7,4,1)
result = model.predict(y_pred)

print('loss : ', loss)
print('[ccc]의 결과 : ', result)


#  [[ 99.632935]  
#  [100.22998 ]
#  [100.70619 ]
#  [101.15499 ]
#  [101.60197 ]
#  [102.04795 ]
#  [102.49367 ]]


#3차원으로 바꾸기 위해서 reshape 해야 함.







# 목적 : 시계열 데이터를 자르기, 결과치를 확인해보면 어떻게 잘리는 지 확인할 수 있음.

#def split_x(dataset, size): 에서 dataset=x, size=y로 보아도 무방함
#range(11) = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (range는 0이상 11미만이기 때문!)

#for in 함수 : 되돌려주는 함수

# [:,:] => 모든 행, 모든 열
# [:, :-1] => 모든 행, 가장 마지막 열 제외
# [:, -1] => 모든 행, 가장 마지막 열만


# append 옆에 붙이는 함수