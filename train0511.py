import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
event = pd.read_csv("./data2021/05_09/285train4.csv",header = None)
PATH = "./data2021/05_09/510trains4.pth"

shrinkage = 64
input_dim = 3
fc1 = 20
fc2 = 20
fc3 = 20
output_dim = 1

learning_rate1 = 0.001
learning_rate2 = 0.0001
learning_rate3 = 0.00001
learning_rate4 = 0.00001
 
epoch1 = 3000
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

#从.csv读取数据并转换为list数据类型
def prepare_data():
	trainD = (event.values[:,45:48])
	trainTr = 100/event.values[:,48:49]
	trainT = np.zeros([trainTr.shape[0],output_dim])
	trainD = trainD.astype(np.float32)
	trainT = trainTr.astype(np.float32)
	trainDm = trainD
	trainTm = trainT
	trainDm = torch.from_numpy(trainDm)
	trainD = torch.from_numpy(trainD)
	trainTm = torch.from_numpy(trainTm)
	return [trainD,trainDm,trainT,trainTm]

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

def learn_reg(epoch,learning_rate,trainD,trainDm,trainT,trainTm):
	loss_func = nn.MSELoss()
	optimizer = torch.optim.SGD(net.parameters(),lr = learning_rate)
	print(trainDm)
	print(trainTm)
	TotalLoss = np.zeros((epoch1))
	for epo in range(epoch):
		prediction = net(trainDm)
		loss = loss_func(prediction, trainTm)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('Epoch [{}/{}], Loss: {:.4f}'.format(epo+1, epoch, loss.data.numpy()))
		TotalLoss[epo] = loss.data.numpy()
	pred = net(trainDm).detach().numpy()
	print(pred)
	plt.figure()
	plt.scatter(x = trainT, y = pred, marker = 'o', s = 0.1)
	plt.xlabel('TrainData(1/Gev)')
	plt.ylabel('Prediction(1/Gev)')
	plt.figure()
	plt.scatter(x = np.arange(0, epoch1), y = TotalLoss, s = 0.1)
	plt.show()

if __name__ == "__main__":
	data = prepare_data()
	net = Net()
	learn_reg(epoch1,learning_rate1,data[0],data[1],data[2],data[3])
	learn_reg(epoch1,learning_rate2,data[0],data[1],data[2],data[3])
	learn_reg(epoch1,learning_rate3,data[0],data[1],data[2],data[3])
	torch.save(net.state_dict(), PATH)