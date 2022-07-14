import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, SimpleRNN, LSTM



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
print(bbb.shape)   #(96,5)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape)   #(96, 4) (96,)

#x를 reshape를 해줄 필요 없음(어차피 2차원이니까)

ccc = split_x(x_predict, 4)
print(ccc.shape)   #(7, 4)



#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_shape=(4,)))
model.add(Dense(70, activation='relu'))
model.add(Flatten()) 
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
# y_pred = ccc.reshape(7,4,1)
result = model.predict(ccc)

print('loss : ', loss)
print('[ccc]의 결과 : ', result)


# LSTM일 때
#  [[ 99.632935]  
#  [100.22998 ]
#  [100.70619 ]
#  [101.15499 ]
#  [101.60197 ]
#  [102.04795 ]
#  [102.49367 ]]


# DNN에서. DNN은 output shape가 1, 2차원 상관 없다
# [[100.22468]
#  [101.23369]
#  [102.24268]
#  [103.25167]
#  [104.26321]
#  [105.27904]
#  [106.29487]]
 
 