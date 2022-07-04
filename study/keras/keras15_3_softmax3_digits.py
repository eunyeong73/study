import numpy as np
from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
import time
from tensorflow.python.keras.callbacks import EarlyStopping


import tensorflow as tf
tf.random.set_seed(66)




#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)   #(1797, 64) (1797,)
print(np.unique(y, return_counts=True))     #[0 1 2 3 4 5 6 7 8 9]

# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
#  array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], 
#        dtype=int64))



# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[0])
# plt.show()
# #그냥 숫자 보기 위해서 하는 것.


from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  #(1797, 10)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66)

print(y_train)
print(y_test)



#모델구성
model = Sequential()
model.add(Dense(5, input_dim=64))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

#다중분류는 softmax(꼭 마지막에 넣어야 함)

#3. 컴파일, 훈련

earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
#다중분류는 로스에 categorical_crossentropy만 씀
hist = model.fit(x_train, y_train, epochs=1000, batch_size=50, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 


end_time = time.time()
print("걸린시간 : ", end_time)


#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])



print("============================================")

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

y_test = model.predict(x_test)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)




# loss :  0.29993587732315063
# accuracy :  0.9166666865348816
# acc스코어 :  1.0

