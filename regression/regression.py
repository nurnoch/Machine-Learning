#coding=utf-8

import random
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) - 1
	datMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		datMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return datMat, labelMat

# 求出最佳拟合曲线的系数
def standRegres(xArr, yArr):
	xMat = mat(xArr)
	yMat = mat(yArr).T 
	xTx = xMat.T * xMat
	if linalg.det(xTx) == 0.0:
		print "This matrix is singular, cannot inverse"
		return
	ws = xTx.I * (xMat.T * yMat)
	return ws

# 画出最佳拟合直线
def plotBestFitLine(xMat, yMat, ws):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0]) # 数据的散点图
	xCopy = xMat.copy()
	xCopy.sort(0)
	yHat = xCopy * ws
	ax.plot(xCopy[:, 1].tolist(), yHat.tolist())
	plt.show()

# 局部加权线性回归
def lwlr(testPoint, xArr, yArr, k = 1.0):
	xMat = mat(xArr)
	yMat = mat(yArr).T 
	m = shape(xMat)[0]
	weights = mat(eye((m))) # m * m矩阵，一共有m个点
	for j in range(m):
		diffMat = testPoint - xMat[j, :]
		weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k**2))
	xTx = xMat.T * (weights * xMat)
	if linalg.det(xTx) == 0:
		print "This matrix is singular, cannot inverse"
		return
	ws = xTx.I * (xMat.T * (weights * yMat))
	return testPoint * ws

# 对每个点利用lwlr进行预测
def lwlrTest(testArr, xArr, yArr, k = 1.0):
	m = shape(testArr)[0]
	yHat = zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i], xArr, yArr, k)
	return yHat

# lwlr拟合
def plotLwlr(xArr, yArr, yHat):
	xMat = mat(xArr)
	srtInd = xMat[:, 1].argsort(0)
	xSort = xMat[srtInd][:, 0, :]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xSort[:, 1].tolist(), yHat[srtInd].tolist())
	ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s = 2, c = 'red')
	plt.show()

# 误差平方和
def rssError(yArr, yHatArr):
	return ((yArr - yHatArr)**2).sum()

# 给定lambda下，岭回归求解系数
def ridgeRegres(xMat, yMat, lam=0.2):
	xTx = xMat.T * xMat
	denom = xTx + eye(shape(xMat)[1]) * lam
	if linalg.det(denom) == 0:
		print "THis matrix is singular, cannot do inverse"
		return
	ws = denom.I * (xMat.T * yMat)
	return ws

# 在一组lambda上，测试结果
def ridgeTest(xArr, yArr):
	xMat = mat(xArr)
	yMat = mat(yArr).T
	yMean = mean(yMat, 0)
	yMat = yMat - yMean
	# 对特征进行标准化处理，使得每个特征的重要性相等
	xMeans = mean(xMat, 0)
	xVar = var(xMat, 0)
	xMat = (xMat - xMeans) / xVar  # 所有特征减去各自的均值并除以方差
	
	numTestPts = 30  # 30个不同的lambda
	wMat = zeros((numTestPts, shape(xMat)[1]))
	for i in range(numTestPts):
		ws = ridgeRegres(xMat, yMat, exp(i-10))
		wMat[i,:] = ws.T
	return wMat

# 标准化
def regularize(xMat):
	inMat = xMat.copy()
	inMeans = mean(inMat, 0)
	inVar = var(inMat, 0)
	inMat = (inMat - inMeans) / inVar
	return inMat

# 前向逐步回归
def stageWise(xArr, yArr, eps=0.01, numIt=100):
	# 数据标准化，使其满足0均值和单位方差
	xMat = mat(xArr)
	yMat = mat(yArr).T
	yMean = mean(yMat, 0)
	yMat = yMat - yMean
	xMat = regularize(xMat)
	m, n = shape(xMat)
	returnMat = zeros((numIt, n))
	ws = zeros((n, 1))  # n个系数
	wsTest = ws.copy()
	wsMax = ws.copy()
	for i in range(numIt): # 在每一轮迭代
		print ws.T
		lowestError = inf
		for j in range(n):  # 对每一个特征
			for sign in [-1, 1]:  
				wsTest = ws.copy()
				wsTest[j] += eps*sign  # 增大或者减小eps
				yTest = xMat * wsTest
				rssE = rssError(yMat.A, yTest.A)
				if rssE < lowestError:
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy()
		returnMat[i, :] = ws.T
	return returnMat

# 交叉验证测试岭回归
def crossValidation(xArr, yArr, numVal=10):
	m = len(yArr) # 数据点的个数
	indexList = range(m)
	errorMat = zeros((numVal, 30)) # 每一次交叉验证每一个回归的误差
	for i in range(numVal):  # 交叉验证numVal次，这里为10折交叉验证
		# 90%用于训练，10%用于测试
		trainX = []; trainY = []
		testX = []; testY = []
		random.shuffle(indexList)
		for j in range(m):
			if j < 0.9 * m:
				trainX.append(xArr[indexList])
				trainY.append(yArr[indexList])
			else:
				testX.append(xArr[indexList])
				testY.append(yArr[indexList])
		wMat = ridgeTest(trainX, trainY) # 利用当前训练样本和30个不同的lambda得到的30组回归系数
		for k in range(30):
			matTestX = mat(testX); matTrainX = mat(trainX)
			matTestX = regularize(matTestX)
			yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)  # 还原后的预测值
			errorMat[i, k] = rssError(yEst.T.A, array(testY))

	meanErrors = mean(errorMat, 0) # 每一次交叉验证得到的error平均值
	minMean = float(min(meanErrors))
	bestWeights = wMat[nonzero(meanErrors == minMean)]  # 得到error最低的index
	xMat = mat(xArr)
	yMat = mat(yArr).T
	meanX = mean(xMat, 0)
	varX = var(xMat, 0)
	# 还原数据 ？？？
	unReg = bestWeights / varX	
	print "The best model from Ridge Regression is:\n", unReg
	print "with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat)





