#coding=utf-8
from numpy import *

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float, curLine)
		dataMat.append(fltLine)
	return dataMat

# 根据给定的特征和特征值进行二元切分
def binSplitDataSet(dataSet, feature, value):
	mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
	mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
	return mat0, mat1

# 树构建
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	if feat == None:
		return val
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree

# 负责生成叶节点
def regLeaf(dataSet):
	return mean(dataSet[:, -1])

# 误差估计函数
def regErr(dataSet):
	return var(dataSet[:, -1]) * shape(dataSet)[0]

# 寻找最佳切分的特征和特征值
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	# 函数停止时机
	tolS = ops[0] # 容许的误差下降值
	tolN = ops[1] # 切分的最少样本数
	if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # 所有值都相等
		return None, leafType(dataSet)
	m, n = shape(dataSet)
	S = errType(dataSet)
	bestS = inf
	bestIndex = 0
	bestValue = 0
	for featIndex in range(n-1): # 最后一维是y
		for splitVal in set(dataSet[:, featIndex]):
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
				continue
			newS = errType(mat0) + errType(mat1)
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	if (S - bestS) < tolS: # 误差减少不大
		return None, leafType(dataSet)
	mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 切分出的数据集很小
		return None, leafType(dataSet)
	return bestIndex,bestValue














