# sklearn.datasets.load_diabetes
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)     #(442, 10) (442,)

print(datasets.feature_names)
print(datasets.DESCR)

#[실습]
# R2 0.62 이상

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, 
    # shuffle=True,
    random_state=86)


datasets = load_diabetes()
x = datasets.data
y = datasets.target


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                restore_best_weights=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse']) 
hist = model.fit(x_train, y_train, epochs=1000, batch_size=30, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 

#false - 제일 작은 값에서 10번 더 간 값(끊긴 값)을 가져옴
#true - 끊어졌을 때 기준 제일 작은 loss에서 가장 작은 weight를 가져옴.



#4. 평가, 예측


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

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  1678.2406005859375
# r2스코어 :  0.6653577770621634


# ###################### early stopping 사용 후
# loss :  1652.318603515625
# r2스코어 :  0.6705266489092283


# #0701숙제 피드백
# r2스코어 :  -4.4666294410240495
# earlyStopping이 잡히지 않음.