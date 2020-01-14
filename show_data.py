#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


inFile = open('/home/lhw/uisee/face-2/r.txt', 'r')#以只读方式打开某fileName文件
File = open('/home/lhw/uisee/face-2/r_ind.txt', 'r')#以只读方式打开某fileName文件
#定义两个空list，用来存放文件中的数据
x = []
y = []
z = []
x1 = []
y1 = []
z1 = []
for line in inFile:
	trainingSet = line.split( ) #对于每一行，按','把数据分开，这里是分成两部分
	x.append(float(trainingSet[0])) #第一部分，即文件中的第一列数据逐一添加到list X 中
	y.append(float(trainingSet[1])) #第二部分，即文件中的第二列数据逐一添加到list y 中
	z.append(float(trainingSet[2])) #第二部分，即文件中的第二列数据逐一添加到list y 中
for line1 in File:
	trainingSet1 = line1.split( ) #对于每一行，按','把数据分开，这里是分成两部分
	x1.append(float(trainingSet1[0])) #第一部分，即文件中的第一列数据逐一添加到list X 中
	y1.append(float(trainingSet1[1])) #第二部分，即文件中的第二列数据逐一添加到list y 中
	z1.append(float(trainingSet1[2])) #第二部分，即文件中的第二列数据逐一添加到list y 中
#转化为array数组，便于列数据的切片获取
x=np.array(x)
y=np.array(y)
z=np.array(z)

x1=np.array(x1)
y1=np.array(y1)
z1=np.array(z1)
#x=x.astype('float32')
#y=y.astype('float32')
#z=z.astype('float32')


#ax = plt.figure().add_subplot(111, projection = '3d')
#ax=plt.axes(projection='3d')
ax = plt.subplot(111, projection='3d')

#ax.scatter(x,y,z,s=1,c='y')
ax.scatter(x1,y1,z1,s=20,c='r')
#ax.set_zlabel('z')
#ax.set_ylabel('y')
#ax.set_xlabel('x')



plt.show()

