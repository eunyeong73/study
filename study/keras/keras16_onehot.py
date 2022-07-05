#[과제]
#[3가지 원핫인코딩 방식을 비교할 것]


#1. pandas의 get_dummies
#2. tensorflow의 to_categorical
#3. sklearn의 OneHotEncoder


#미세한 차이를 정리하기.



#1. pandas의 get_dummies

y = pd.get_dummies(y)
#카테고리 아이디 별로 그대로 정리


#2. tensorflow의 to_categorical

from sklearn import datasets
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)

#데이터의 시작에 관계없이 무조건 0으로 시작하도록 만듦 -> 무조건 0부터 칼럼 생성
#만약 데이터 레이블이 3부터 있다면 0, 1, 2 체킹 칼럼이 생성되어 버림

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
# 0으로 이루어진 벡터에 단 한개의 1의 값으로 해당 데이터의 값을 구별하는 것이 원핫 인코딩이다.
# get_dummies와 같은 방식. 백터였던 y데이터를 행렬로 변환하는 작업 후 사용
# 다른 두가지는 백터도 칼럼 레이블로 인식해주는데, 얘는 못 함.