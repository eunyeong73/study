#https://www.kaggle.com/competitions/bike-sharing-demand/code


#캐글 BIKE 문제풀이
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os
print(os.listdir("./_data/kaggle_bike/"))

cnt=train_df['count'].values
q99=np.percentile(cnt,[99])


train_df=train_df[train_df['count']<q99[0]]
sns.distplot(train_df['count'])
plt.show()

#from scipy.stats import boxcox
train_df['count']=train_df['count'].apply(lambda x:np.log(x))
#train_df['count']=boxcox(train_df['count'])[0]
sns.distplot(train_df['count'])
plt.show()
print (train_df['count'])


'''
#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
print(train_set)
print(train_set.shape)     # [10886 rows x 11 columns]


test_set = pd.read_csv(path + 'test.csv',       #예측에서 사용할 예정
                       index_col=0)

'''

print(test_set)
print(test_set.shape)     # [8 rows x 11 columns]

print(train_set.columns)
print(train_set.info())
print(train_set.describe())


test_set.head()   #[8 rows x 11 columns]


train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')

train_df.describe()
train_df.info()

x = train_set.drop(['count','casual','registered'], axis=1)  
#test와 train의 열을 맞추기 위해서 train에서 세 가지 열을 날림.
print(x)
print(x.columns)
print(x.shape)          # 10886 , 8

y = train_set['count']
print(y)
print(y.shape)          # 10886 , 


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.9, shuffle=True, random_state=31)



#2. 모델구성
model = Sequential()
model.add(Dense(6, input_dim=8))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=450)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


y_predict = model.predict(x_test)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))    # 루트씌운다.


# test_set=test_set.fillna(0)

# #얘는 결측치를 0으로 채워넣는 것.



result = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
y_summit = model.predict(test_set)
result['count'] = y_summit
result.to_csv(path + 'sampleSubmission.csv', index=True)

