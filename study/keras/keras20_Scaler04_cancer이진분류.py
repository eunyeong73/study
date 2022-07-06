import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
from sklearn.preprocessing import MaxAbsScaler, RobustScaler



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

#########################스케일링########################

#scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler = StandardScaler()
# # scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#######################스케일링#########################




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

y_predict = model.predict(x_test)
y_predict = np.round(y_predict, 0)
#**반올림 하는 함수

#####[과제 1. accuracy score 완성하기]
from sklearn.metrics import r2_score, accuracy_score
#r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)


# # 1. 스케일링 전
# loss :  [0.22274695336818695, 0.8859649300575256, 0.07069212943315506]
# acc스코어 :  0.8859649122807017


# 2. 민맥스 후
#loss :  [0.10774346441030502, 0.9561403393745422, 0.03176480159163475]
# acc스코어 :  0.956140350877193

# # # 3. 스탠다드 후
# loss :  [0.08896598219871521, 0.9736841917037964, 0.023911558091640472]
# acc스코어 :  0.9736842105263158


# # MaxAbsScaler 후
# loss :  [0.1204063817858696, 0.9561403393745422, 0.03568060323596001]
# acc스코어 :  0.956140350877193


# #  RobustScaler 후
# loss :  [0.10248217731714249, 0.9649122953414917, 0.027276020497083664]
# acc스코어 :  0.9649122807017544

