from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score
# sklearn.datasets.fetch_california_housing

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)

print(datasets.feature_names)
print(datasets.DESCR)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, 
    # shuffle=True,
    random_state=66)


datasets = load_boston()
x = datasets.data
y = datasets.target

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(5, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

import time

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse']) 

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

# ---> earlystopping에 대한 정의.

#결과값에서 가장 마지막 수에서 11번째 역순이 가장 작은 수.



start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=5, validation_split=0.2, 
                 callbacks=[earlyStopping],
                 verbose=1) 
end_time = time.time()



# train_size의 비율 0.2이므로 validation은 전체에서 16%

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

#print("============================================")
print(hist) # <tensorflow.python.keras.callbacks.History object at 0x0000020EE9092160>

#print("============================================")
print(hist.history) #이 안에 loss와 val.loss가 있음.

#print("============================================")
print(hist.history['loss'])

#print("============================================")
print(hist.history['val_loss'])


print("걸린시간 : ", end_time)


y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)




#그림 그리기

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family']="Malgun Gothic"
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('보스턴 loss와 val 비교')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()




#딕셔너리 키(히스토리), 벨류(리스트)




#과적합 이슈 -> loss와 val이 차이가 나는 것이 더 안 좋음.(loss의 성능이 더 좋다고 하더라도)


# #0701 
# 걸린시간 :  1656648498.5567405
# r2스코어 :  0.5296591948260512


#0701 숙제 피드백
#r2 스코어 마이너스 값이 계속 나옴. 추가한다고 더 좋아지지는 않는 건가라는 생각이 듦. 또한 earlystopping도 잡히지 않음.

# 걸린시간 :  1656821812.3661463
# r2스코어 :  -5.634823528805671