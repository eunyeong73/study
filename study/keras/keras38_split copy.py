import numpy as np

a = np.array(range(1, 11))
size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)   #(6,5)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape)   #(6,4) (6,)



# 목적 : 시계열 데이터를 자르기, 결과치를 확인해보면 어떻게 잘리는 지 확인할 수 있음.

#def split_x(dataset, size): 에서 dataset=x, size=y로 보아도 무방함
#range(11) = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (0이상 11미만이기 때문!)

#for in 함수 : 되돌려주는 함수

# [:,:] => 모든 행, 모든 열
# [:, :-1] => 모든 행, 가장 마지막 열 제외
# [:, -1] => 모든 행, 가장 마지막 열만


