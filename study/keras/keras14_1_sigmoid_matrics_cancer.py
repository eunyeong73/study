from re import X
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data    #(569, 30) 
y = datasets.target  #(569,)
print(x.shape, y.shape)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=31, train_size=0.8)


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=30))
model.add(Dense(5, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(10, activation='relu')) # 히든에서만 쓸 수 있음. 중간에서만 사용. 평타는 치는 좋은 애 adam.같은 애.
model.add(Dense(30, activation='linear'))
model.add(Dense(1, activation='sigmoid')) #이진 분류에서는 마지막에는 sigmoid 활성화 함수 사용


#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=40, mode='min', verbose=1,
                restore_best_weights=True)
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse']) 
# 이진 분류에서는 loss=binary_crossentropy 사용
# 평가지표 accuracy 넣음
# 두 개 이상은 리스트. 위가 리스트 형태니까 넣어줘도 가능함.
# 메트릭스는 추가지표 넣을 때 쓰는 것.

hist = model.fit(x_train, y_train, 
                 epochs=1000, 
                 batch_size=50, 
                 validation_split=0.2, 
                 callbacks=[earlyStopping], 
                 verbose=1) 

#4. 평가, 예측



loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family']="Malgun Gothic"
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('diabets loss와 val 비교')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()




y_predict = model.predict(x_test)

#####[과제 1. accuracy score 완성하기]
from sklearn.metrics import r2_score, accuracy_score
#r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)

#하나는 소수점이고 또 하나는 1,0,1,0이니까 나머지 하나 반올림 처리 하면 될듯.




# loss :  0.11455793678760529
# r2스코어 :  0.5272171653174207



#activation 활성화 함수 : 모든 레이어에 강림할 수 있는 함수 => 한정시키는 역할. 
# -> 하이퍼 파라미터 튜닝의 한 종류가 되기도 함.
#엑티베이션 함수에 통과시킴으로써 활성화 함수가 원하는 수에 수렴시키는 함수.
#sigmoid 함수 : 0과 1사이의 수로 수렴하는 것. (한정시킴)

#우리는 항상 엑티베이션 함수를 쓰고 있었음. [[[  activation='linear'  ]]] 그냥 선이라는 뜻

#결과치 0,1을 원했을 때 sigmoid에서 반올림해주면 해결됨.

#이진 분류에서는 컴파일 loss='binary_crossentropy'를 씀. 0과 1로 분류할 때.
#이진 분류에서는 마지막에는 sigmoid 활성화 함수 사용
 # ==> 반드시!!
 
 
 
# loss :  0.1809016317129135
# r2스코어 :  0.7458561824112064