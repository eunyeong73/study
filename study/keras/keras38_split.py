import numpy as np

a = np.array(range(1, 11))
size = 5

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