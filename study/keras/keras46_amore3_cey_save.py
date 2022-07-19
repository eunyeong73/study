import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import time
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
path = './_data/test_amore_0718/'
dataset_sam = pd.read_csv(path + '삼성전자220718.csv', thousands=',', encoding='cp949')
dataset_amo = pd.read_csv(path + '아모레220718.csv', thousands=',', encoding='cp949')

print(dataset_sam.columns)

dataset_sam = dataset_sam.drop(['전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)
dataset_amo = dataset_amo.drop(['전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)

# dataset_amo.info()
# dataset_sam.info()
dataset_sam = dataset_sam.fillna(0)
dataset_amo = dataset_amo.fillna(0)

dataset_sam = dataset_sam.loc[dataset_sam['일자']>="2018/05/04"] # 액면분할 이후 데이터만 사용
dataset_amo = dataset_amo.loc[dataset_amo['일자']>="2018/05/04"] # 삼성의 액면분할 날짜 이후의 행개수에 맞춰줌
print(dataset_amo.shape, dataset_sam.shape) # (1035, 11) (1035, 11)

dataset_sam = dataset_sam.sort_values(by=['일자'], axis=0, ascending=True) # 오름차순 정렬
dataset_amo = dataset_amo.sort_values(by=['일자'], axis=0, ascending=True)
print(dataset_amo.head) # 앞 다섯개만 보기

feature_cols = ['시가', '고가', '저가', '거래량', '기관', '외국계', '종가']
label_cols = ['종가']

dataset_sam = dataset_sam[feature_cols]
dataset_amo = dataset_amo[feature_cols]
dataset_sam = np.array(dataset_sam)
dataset_amo = np.array(dataset_amo) # ':'을 쓰기 위해서(데이터 프레임에서는 적용 불가.)

# 시계열 데이터 만드는 함수
dataX = [] # 입력으로 사용될 Sequence Data
dataY = [] # 출력(타켓)으로 사용

def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1 : y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

SIZE = 3
COLSIZE = 3
x1, y1 = split_xy(dataset_amo, SIZE, COLSIZE)
x2, y2 = split_xy(dataset_sam, SIZE, COLSIZE)
print(x1.shape, y1.shape) # (1030, 3, 7) (1031, 3)


x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y1, test_size=0.2, shuffle=False)


# data 스케일링
scaler = MinMaxScaler()
print(x1_train.shape, x1_test.shape) # (824, 3, 6) (207, 3, 6)
print(x2_train.shape, x2_test.shape) # (824, 3, 6) (207, 3, 6)
print(y_train.shape, y_test.shape) # (824, 3) (207, 3)


x1_train = x1_train.reshape(824*3,6)
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(207*3,6)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(824*3,6)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(207*3,6)
x2_test = scaler.transform(x2_test)

# Conv1D에 넣기 위해 3차원화
x1_train = x1_train.reshape(824, 3, 6)
x1_test = x1_test.reshape(207, 3, 6)
x2_train = x2_train.reshape(824, 3, 6)
x2_test = x2_test.reshape(207, 3, 6)

# 2. 모델구성
# 2-1. 모델1
input1 = Input(shape=(3, 6))
dense1 = Conv1D(64, 2, activation='relu', name='d1')(input1)
dense2 = LSTM(40, activation='relu', name='d2')(dense1)
dense3 = Dense(33, activation='relu', name='d3')(dense2)
dense4 = Dense(24, activation='relu', name='d4')(dense3)
dense5 = Dense(22, activation='relu', name='d5')(dense4)
output1 = Dense(32, activation='relu', name='out_d1')(dense5)

# 2-2. 모델2
input2 = Input(shape=(3, 6))
dense11 = Conv1D(64, 2, activation='relu', name='d11')(input2)
dense12 = LSTM(40, activation='tanh', name='d12')(dense11)
dense13 = Dense(34, activation='relu', name='d13')(dense12)
dense14 = Dense(30, activation='relu', name='d14')(dense13)
output2 = Dense(16, activation='relu', name='out_d2')(dense14)

from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='m1')
merge2 = Dense(50, activation='relu', name='mg2')(merge1)
merge3 = Dense(30, name='mg3')(merge2)
merge4 = Dense(70, name='mg4')(merge3)
last_output = Dense(1, name='last')(merge4)

model = Model(inputs=[input1, input2], outputs=[last_output])

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
fit_log = model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=64, callbacks=[Es], validation_split=0.1)
end_time = time.time()
model.save('./_test/keras46_jongga6.h5')

# model = load_model('./_test/keras46_siga3.h5')

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('prdict: ', predict[-1:]) # 제일 마지막에 나온거 하나 슬라이싱
print('걸린 시간: ', end_time-start_time)



# # keras46_jongga5.h5
# loss:  26575560.0
# prdict:  [[135029.53]]
# 걸린 시간:  38.43431234359741

# keras46_jongga6