#[과제]
#[3가지 원핫인코딩 방식을 비교할 것]


#1. pandas의 get_dummies
#2. tensorflow의 to_categorical
#3. sklearn의 OneHotEncoder


#미세한 차이를 정리하기.



#1. pandas의 get_dummies

y = pd.get_dummies(y)



#2. tensorflow의 to_categorical

from sklearn import datasets
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)

#데이터의 시작에 관계없이 무조건 0으로 시작하도록 만듦

#3. sklearn의 OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder
print(y.shape)
y = datasets.target.reshape(-1,1)
print(y.shape)
oh.fit(y)
y = oh.transform(y).toarray()
print(y)
print(y.shape)


# 원핫(One-Hot) 인코딩이라는 말처럼 이 기술은 데이터를 수많은 0과 한개의 1의 값으로 데이터를 구별하는 인코딩이다. 
#  0으로 이루어진 벡터에 단 한개의 1의 값으로 해당 데이터의 값을 구별하는 것이 원핫 인코딩이다.