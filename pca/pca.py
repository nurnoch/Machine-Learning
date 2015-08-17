#coding=utf-8

from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim = '\t'):
	fr = open(fileName)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	datArr = [map(float, line) for line in stringArr]
	return mat(datArr)

def pca(dataMat, topNfeat = 99999):
	meanVals = mean(dataMat, axis = 0) # 每一列的平均值，即每个特征的均值
	meanRemoved = dataMat - meanVals
	covMat = cov(meanRemoved, rowvar = 0)
	eigVals, eigVects = linalg.eig(mat(covMat))
	eigValIdxs = argsort(eigVals)
	eigValIdxs = eigValIdxs[: -(topNfeat+1): -1] # 后面的topNfeat个
	redEigVects = eigVects[:, eigValIdxs]
	lowDDataMat = meanRemoved * redEigVects
	reconMat = (lowDDataMat * redEigVects.T) + meanVals
	return lowDDataMat, reconMat

def plotData(fileName):
	dataMat = loadDataSet(fileName)
	lowDMat, reconMat = pca(dataMat, 1)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker = '^', s = 90)
	ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker = 'o', s = 50, c = 'red')
	plt.show()

####示例：利用PCA对半导体制造数据降维

# 处理缺失数据
def replaceNanWithMean():
	dataMat = loadDataSet('secom.data', ' ')
	numFeat= shape(dataMat)[1]
	for i in range(numFeat):
		meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0], i])
		dataMat[nonzero(isnan(dataMat[:, i].A))[0], i] = meanVal # nan用均值替换
	return dataMat