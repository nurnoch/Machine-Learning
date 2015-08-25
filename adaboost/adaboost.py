#coding=utf-8
from numpy import *
import matplotlib.pyplot as plt 

def loadSimpleData():
    datMat = matrix([[1., 2.1], [2. , 1.1], [1.3, 1.],
                    [1. , 1.], [2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

# 根据小于或者是大于，对dimen维的各个数进行分类，分别为-1和1两类
def stumpClassify(dataMatrix, dimen, threshVal , threshIneq):
	retArray = ones((shape(dataMatrix)[0], 1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:, dimen] > threshVal] = -1.0
	return retArray

# 找到数据集上的一棵最佳单层决策树
def buildStump(dataArr, classLabels, D):
	dataMatrix = mat(dataArr)
	labelMat = mat(classLabels).T 
	m, n = shape(dataMatrix)
	numSteps = 10.0
	bestStump = {}  # 存储最佳决策树的相关信息
	bestClassEst = mat(zeros((m, 1)))
	minError = inf
	for i in range(n): # 遍历所有的n个特征
		rangeMin = dataMatrix[:, i].min()
		rangMax = dataMatrix[:, i].max()
		stepSize = (rangMax - rangeMin) / numSteps
		for j in range(-1, int(numSteps)+1):
			for inequal in ['lt', 'gt']:
				threshVal = rangeMin + float(j) * stepSize
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
				errArr = mat(ones((m, 1)))
				errArr[predictedVals == labelMat] = 0 # 预测错误为1
				weightedError = D.T * errArr 
				#print "split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
				if weightedError < minError:
					minError = weightedError
					bestClassEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump, minError, bestClassEst # 返回单层决策树， 最小的错误率， 估计的类别向量


# 基于单层决策树的AdaBoost训练
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
	weakClassArr = [] # 弱分类器数组
	m = shape(dataArr)[0]
	D = mat(ones((m, 1)) / m)  # 每个点的当前权重，随着算法运行会增加错误点的权重
	aggClassEst = mat(zeros((m, 1)))  # 每个数据点的类别估计累计值
	for i in range(numIt): # 迭代numIt次，或者错误率为0
		bestStump, error, classEst = buildStump(dataArr, classLabels, D) # 找到当前D下错误率最低的决策树
		print "D: ", D.T
		alpha = float(0.5 * log((1 - error) / max(error, 1e-16))) # 计算分类器的权重
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump) 
		print "classEst: ", classEst.T
		# 更新权重D
		expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
		D = multiply(D, exp(expon))
		D = D / D.sum()
		# 错误率累加计算
		aggClassEst += alpha * classEst
		#print "aggClassEst: ", aggClassEst.T
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1))) # 错误处为1
		errorRate = aggErrors.sum() / m
		print "total error: ", errorRate, "\n"
		if errorRate == 0.0:
			break
	return weakClassArr

# AdaBoost分类函数： 利用多个弱分类器
def adaClassify(datToClass, classifierArr):
	dataMatrix = mat(datToClass)
	m = shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m, 1)))
	for i in range(len(classifierArr)):
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha'] * classEst
		print aggClassEst
	return sign(aggClassEst)

#### 示例：预测马得病的死亡率

def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t'))
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat-1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat, labelMat


# ROC曲线的绘制及AUC计算
def plotROC(predStrengths, classLabels):
	cur = (1.0, 1.0)
	ySum = 0.0
	numPosClas = sum(array(classLabels) == 1.0) # 正例个数
	yStep = 1 / float(numPosClas)
	xStep = 1 / float(len(classLabels) - numPosClas)
	sortedIndicies = predStrengths.argsort()

	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)

	for idx in sortedIndicies.tolist()[0]:
		if classLabels[idx] == 1.0:
			delX = 0
			delY = yStep
		else:
			delX = xStep
			delY = 0
			ySum += cur[1]

		ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b')
		cur = (cur[0] - delX, cur[1] - delY)

	ax.plot([0, 1], [0, 1], 'b--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for AdaBoost Horse Colic Detection System')
	ax.axis([0, 1, 0, 1])
	plt.show()
	print "The Area Under the Curve is: ", ySum * xStep

	
























