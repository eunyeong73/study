# sklearn.datasets.fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)        # (20640, 8) (20640,)

print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=31, train_size=0.8)

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(5, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련


from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                restore_best_weights=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse']) 
hist = model.fit(x_train, y_train, epochs=500, batch_size=50, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 



#4. 평가, 예측


y_predict = model.predict(x_test)

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
plt.title('캘리포니아 loss와 val 비교')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)



# loss :  0.6248257160186768
# r2스코어 :  0.5557542345237354


######################validation 이후

# loss :  0.647396981716156
# r2스코어 :  0.5133872313137797



######################early stopping 이후


# loss :  0.6347912549972534
# r2스코어 :  0.5162551328338514


# ###0701숙제 피드백
# loss :  2.428748846054077
