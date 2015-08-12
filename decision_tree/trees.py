#coding=utf-8
from math import log
import operator
import pickle

# 计算熵H(Y)
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/ numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

# 生成测试数据
def createDataSet():
	dataSet = [[1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no'],
				[0, 1, 'no'],
				[0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

# 根据给定特征划分数据集
def splitDataSet(dataSet, axis, value):
	retDataSet = []  # 新建一个list，防止原数据集被修改
	for featVec in dataSet:
		if featVec[axis] == value:
			reduceFeatVec = featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reduceFeatVec)
	return retDataSet

# 利用信息增益，选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1 # 最后一个是label
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	for i in xrange(numFeatures):
		featList = [example[i] for example in dataSet] # 每个列的值
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)  # H(Y|X)
		infoGain = baseEntropy - newEntropy # H(Y) - H(Y|X)
		if(infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

# 决定叶节点的类别
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		classCount[vote] = classCount.get(vote, 0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0] # 返回最大类别的标签

# 创建树
def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList): # 当前分类列表只有一种类，类完全相同
		return classList[0]
	if len(dataSet[0]) == 1: # 遍历完所有的属性，返回出现次数最多的类标签
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel: {}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat,value), subLabels)
	return myTree

# 使用决策树进行分类
def classify(inputTree, featLabels, testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict': 
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel

# 存储决策树
def storeTree(inputTree, filename):
	fw = open(filename, 'w')
	pickle.dump(inputTree, fw)
	fw.close()

# 读取存储的决策树
def grabTree(filename):
	fr = open(filename)
	return pickle.load(fr)
	











