#coding=utf-8
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
	dataMat = []
	labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split() # [feat1	feat2	label]
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

def sigmoid(inX):
	return 1.0 / (1 + exp(-inX))

# 批梯度上升
def gradAscent(dataMatIn, classLabels):
	dataMat= mat(dataMatIn) # 100行3列
	labelMat = mat(classLabels).transpose() # 100行1列
	m, n = shape(dataMat)
	alpha = 0.001 # learning rate
	maxCycles = 500
	weights = ones((n, 1))
	for k in range(maxCycles):
		h = sigmoid(dataMat * weights)  # 返回的是列向量，100行1列
		error = (labelMat - h)
		weights = weights + alpha * dataMat.transpose() * error
	return weights

# 随机梯度上升
def stocGraAscent0(dataMatIn, classLabels):
	dataMat = array(dataMatIn)
	m, n = shape(dataMat)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMat[i] * weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMat[i]
	return weights 

# 改进的随机梯度上升
def stocGraAscent1(dataMatIn, classLabels, numIter = 150):
	dataMat = array(dataMatIn)
	m, n = shape(dataMat)
	weights = ones(n)
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4 / (1.0 + j + i) + 0.01
			randIndex = int(random.uniform(0, len(dataIndex)))
			h = sigmoid(sum(dataMat[randIndex] * weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMat[randIndex]
			del(dataIndex[randIndex])
	return weights


# 画出决策边界线
def plotBestFit(weights):
	#weights = weights.getA() # 转换为array
	dataMat, labelMat = loadDataSet()
	dataArr = array(dataMat)
	m = shape(dataArr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(m):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i, 1])
			ycord1.append(dataArr[i, 2])
		else:
			xcord2.append(dataArr[i, 1])
			ycord2.append(dataArr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
	ax.scatter(xcord2, ycord2, s = 30, c = 'green')
	x = arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2] # w0 + w1*x + w2*y = 0
	ax.plot(x, y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()


#### 示例：对马死亡进行预测

def clasifyVector(intX, weights):
	prob = sigmoid(sum(intX * weights))
	if prob > 0.5:
		return 1
	else:
		return 0

def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet = []
	trainingLabels = []
	for line in frTrain.readlines():
		curLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(curLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(curLine[21]))
	trainWeights = stocGraAscent1(trainingSet, trainingLabels, 500)
	errorCount = 0.0
	numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1
		curLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(curLine[i]))
		if clasifyVector(array(lineArr), trainWeights) != int(curLine[21]):
			errorCount += 1
	errorRate = float(errorCount) / numTestVec
	print "The error rate of this test is: {errorRate}".format(errorRate = errorRate)
	return errorRate

def multiTest():
	numTests = 10
	errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print "After {numTests} iterations the average error rate is {errorRate}".format(numTests = numTests, 
		errorRate = errorSum / float(numTests))























