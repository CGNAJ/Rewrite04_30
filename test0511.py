import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math
from mpl_toolkits.mplot3d import Axes3D

event = pd.read_csv("./data2021/05_09/510afterfilter.csv",header = None)
PATH = "./data2021/05_09/s1.pth"

thresholdv=16
input_dim = 3
fc1 = 20
fc2 = 20
fc3 = 20
output_dim = 1

epoch1 = 1
eventnumber = 266840
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
scan_arr4 = [[],[], [], [],[], [],[],[], [],[], [],[], [],[],[],[],[], [],[],[],[],[], [],[],[],[],[], [],[],[],[],[], [],[],[],[],[], [],[],[],[],[], [], [], [], [], [], [], [], [], [], []]
regionj=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,60,70,80,85]
check_number = 0

#从.csv读取数据并转换为list数据类型
def prepare_data():
	testDm = (event.values[:, 43:46])
	testDm = np.array(testDm)
	'''for i in range(eventnumber):
		trainD[i][0]=((trainD[i][0]-trainD[i][1]*6.803/7.481) * 0.03 + 0.015) / 0.06
		trainD[i][2] = ((trainD[i][2] - trainD[i][1] * 9.835 / 7.481) * 0.03 + 0.015) / 0.45
		trainD[i][1]=((trainD[i][1]*0.03+0.015)-3.8)/5.2'''
	trainTr = 100/event.values[:, 42:43]
	testT = trainTr.astype(np.float32)
	testDm = testDm.astype(np.float32)
	testDm = torch.from_numpy(testDm)
	testT = torch.from_numpy(testT)
	return [testDm,testT]

#读入一个事例（30个数据点），分析出其中的x坐标
def x_crd(x_arr):
	x_out = []
	for i in range(len(x_arr)):
		if x_arr[i]!=0:
			x_out.append(x_arr[i]*0.03)
	return x_out

#读入一个事例（30个数据点），分析出其中的y坐标
def y_crd(x_arr):
	y_out = []
	y_RPC = [RPC_1_1_Y[0],RPC_1_2_Y[0],RPC_2_1_Y[0],RPC_2_2_Y[0],RPC_3_1_Y[0],RPC_3_2_Y[0]]
	for i in range(len(x_arr)):
		if x_arr[i]!=0:
			y_out.append(y_RPC[i%30//5])
	return y_out

#注意：同RPC的两层探测器被人为增大了距离，防止图片的距离太近，分不清两层的区别
#实际y坐标为RPC1: 6.8, RPC2: 7.478, RPC3: 9.832
#同RPC的两层探测器间距：0.006
RPC_1_X = [0,9.147]
RPC_1_1_Y = [6.8,6.8]
#RPC_1_1_Y = [6.7,6.7]
RPC_1_2_Y = [6.806,6.806]

RPC_2_X = [0,9.66]
RPC_2_1_Y = [7.478,7.478]
#RPC_2_1_Y = [7.378,7.378]
RPC_2_2_Y = [7.484,7.484]

RPC_3_X = [0,12.267]
RPC_3_1_Y = [9.832,9.832]
#RPC_3_1_Y = [9.732,9.732]
RPC_3_2_Y = [9.838,9.838]

#weight和bias的初始化函数
def weight_init(length,width):
	K = torch.zeros(length,width)
	for i in range(length):
		for j in range(width):
			if(i%2==0):
				K[i,j] = 0.5
			else:
				K[i,j] = -0.5
	return K

def bias_init(length):
	B = torch.zeros(length)
	return B

def minmaxscaler(data):
	min = np.amin(data)
	max = np.amax(data)
	return (data - min) / (max - min)

def featurescaler(data):
	mu = np.mean(data)
	std = np.std(data)
	return (data - mu) / std

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.fc1 = nn.Linear(input_dim,fc1)
		self.fc2 = nn.Linear(fc1,fc2)
		self.fc3 = nn.Linear(fc2,fc3)
		self.fc4 = nn.Linear(fc3,output_dim)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x

def learn_reg(testDm,testT):
	print(testDm)
	print(testT)
	pred = net(testDm).detach().numpy()
	print(pred)
	plt.figure()
	plt.scatter(x=100 / testT, y=100 / pred, marker='o', s=0.1)
	plt.xlabel('input: transverse momentum [GeV]')
	plt.ylabel('result: transverse momentum [GeV]')
	plt.show()
	testT = testT.numpy()
	for j in range(52):
		n=1
		r=0
		for i in range(eventnumber):
			if regionj[j]<100/testT[i]<regionj[j+1] or -regionj[j+1]<100/testT[i]<-regionj[j]:
				n=n+1
				if thresholdv<100/pred[i] or 100/pred[i]<-1*thresholdv:
					r=r+1
		scan_arr4[j] = round(r/ n, 3)
	plot_x2=[2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5,30.5,31.5,32.5,33.5,34.5,35.5,36.5,37.5,38.5,39.5,40.5,41.5,42.5,43.5,44.5,45.5,46.5,47.5,48.5,49.5,55,65,75,82.5]
	plot_y4 = scan_arr4
	plt.figure(figsize=(8,5))
	ln5, = plt.plot(plot_x2, plot_y4, color='green', linewidth=2.0, linestyle='--')
	plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,55, 60,65, 70,75, 80, 85],
			   [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,55, 60,65, 70,75, 80, 85])
	plt.legend(handles=[ ln5], labels=['MU16'], loc='lower right')
	plt.title("ratio for NN| RPC receiving eff:70%")
	plt.xlabel('transverse momentum [GeV]')
	plt.ylabel('ratio')
	plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,1.0])
	plt.grid(linestyle=":")
	plt.show()
	print(scan_arr4)



if __name__ == "__main__":
	data = prepare_data()
	net = Net()
	net.load_state_dict(torch.load(PATH))
	learn_reg(data[0],data[1])