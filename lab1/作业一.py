import numpy as np
import pyswarms as ps
np.random.seed(2022)
#1.简单函数书写
def rows(a,n):return a[n,:]
print(rows(np.random.randint(1,7,[5,5]),[0,2,4]))
print("-*-"*25,"第一题","-*-"*25)

#2.简单函数书写
def columns(a,n):return a[:,n]
print(columns(np.random.randint(1,7,[5,5]),[0,2,4]))
print("-*-"*25,"第二题","-*-"*25)

#3.按照题目模拟即可
a1 = np.random.randint(0,10,[10,7])
a2 = np.random.randint(0,10,[7,10])
a = np.matmul(a1,a2)
r = np.linalg.matrix_rank(a)
d = np.linalg.det(a)
P,L,Q = np.linalg.svd(a)
print(L)

print("a的秩为:",r)
print("-*-"*25,"第三题","-*-"*25)

#4.找出秩为7（上面的r等于7）的索引，不考虑循环，大炮打鸟，直接使用PSO寻找
print("a为非奇异矩阵" if r==a.shape[0] else "a为奇异矩阵")
#适应度函数
def func(x,m=a):
    x = [np.round(i).tolist() for i in x]
    for i in range(len(x)):
        for j in range(len(x[0])):x[i][j] = int(x[i][j])
    print(x)
    return -1*np.linalg.matrix_rank(rows(a,x))
#参数设置
dim,swarm_size = 7,50
options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,dimensions=dim,bounds=[[0 for i in range(0,7)],
                                    [9 for i in range(0,7)]],options=options)
cost, pos = optimizer.optimize(func,iters=1000)
print(f"秩为7的元素索引是:{[round(i) for i in pos]}","通过计算秩为:",np.linalg.matrix_rank(a[[round(i) for i in pos]]))
b = a[[round(i) for i in pos]]
print("-*-"*25,"第四题","-*-"*25)

#5.同理使用上面的方法，这里不继续写，得到结果直接写入就行，这里有多个结果，选取了第一个
c = b[:,[0,1,2,3,4,5,6]]
print("矩阵C的秩序:",np.linalg.matrix_rank(c),c.shape)
print("-*-"*25,"第五题","-*-"*25)

#6.逆矩阵计算
cr = np.linalg.inv(c)
print("c与cr的矩阵乘积:",np.matmul(c,cr))
print("cr与c的矩阵乘积:",np.matmul(cr,c))
print("-*-"*25,"第六题","-*-"*25)

#7.特征值计算
eig_value,eig_vector = np.linalg.eig(c)
print("特征值:",eig_value,"特征向量:",eig_vector)
#验证,发现得到的向量相同，证明是其特征值和特征向量
print(c.dot(eig_vector[:,0]),eig_value[0]*eig_vector[:,0])
print("-*-"*25,"第七题","-*-"*25)

#8.SVD分解
P,L,Q = np.linalg.svd(a) #svd分解
#验证P,Q为正交矩阵,也可验各行、列是单位向量且两两正交
print(f"矩阵P为正交矩阵" if np.round(np.linalg.det(P@P.T))==1 else f"矩阵P不为正交矩阵")
print(f"矩阵Q为正交矩阵" if np.round(np.linalg.det(Q@Q.T))==1 else f"矩阵Q不为正交矩阵")
#验证PLQ=a,看两者是否相等
print(P.dot(np.diag(L)).dot(Q),a)
print("-*-"*25,"第八题","-*-"*25)

#9.按照第八题的方法验证QR分解
Q,R = np.linalg.qr(a)
print(f"矩阵Q为正交矩阵" if np.round(np.linalg.det(Q@Q.T))==1 else f"矩阵Q不为正交矩阵")
print(Q@R)
print("-*-"*25,"第九题","-*-"*25)