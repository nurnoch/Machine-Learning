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

# 当前的节点是否为叶节点
def isTree(obj):
	return (type(obj).__name__ == 'dict')

def getMean(tree):
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])
	return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
	if shape(testData)[0] == 0: # 测试集为空
		return getMean(tree)
	if isTree(tree['right']) or isTree(tree['left']):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

	if isTree(tree['left']):
		tree['left'] = prune(tree['left'], lSet)
	if isTree(tree['right']):
		tree['right'] = prune(tree['right'], rSet)

	# 都是叶节点，是否可以进行merge
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
		erorrNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
		treeMean = (tree['left'] + tree['right']) / 2.0
		errorMerge = sum(power(testData[:, -1] - treeMean, 2))
		if errorMerge < erorrNoMerge:
			print "merging"
			return treeMean
		else:
			return tree
	else:
		return tree

#  将数据集格式化为目标变量Y和自变量x
def linearSolve(dataSet):
 	m, n = shape(dataSet)
 	X = mat(ones((m, n)))
 	Y = mat(ones((m, 1)))
 	X[:, 1:n] = dataSet[:, 0:n-1]
 	Y = dataSet[:, -1]
 	xTx = X.T * X
 	if linalg.det(xTx) == 0.0:
 		raise NameError('This matrix is singular, cannot do inverse.')
 	ws = xTx.I * (X.T * Y)
 	return ws, X, Y

# 当需要再切分的时候，负责生成叶节点
def modelLeaf(dataSet):
	ws, X, Y = linearSolve(dataSet)
	return ws

# 计算给定数据集上的误差
def modelErr(dataSet):
	ws, X, Y = linearSolve(dataSet)
	yHat = X * ws
	return sum(power(Y - yHat, 2))


#### 树回归与标准回归的比较

# 回归树
def regTreeEval(model, inDat):
	return float(model)

# 模型树
def modelTreeEval(model, inDat):
	n = shape(inDat)[1]
	X = mat(ones((1, n+1))) 
	X[:, 1:n+1] = inDat
	return float(X * model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
	if not isTree(tree):
		return modelEval(tree, inData)
	if inData[tree['spInd']] > tree['spVal']:
		if isTree(tree['left']):
			return treeForeCast(tree['left'], inData, modelEval)
		else:
			return modelEval(tree['left'], inData)
	else:
		if isTree(tree['right']):
			return treeForeCast(tree['right'], inData, modelEval)
		else:
			return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
	m = len(testData)
	yHat = mat(zeros((m, 1)))
	for i in range(m):
		yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
	return yHat








