#coding=utf-8
from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# kNN算法
def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 计算输入点与其他每个点的距离
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    # 选择距离最近的k个点
    classCount = {}
    for i in xrange(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    # 返回分类标签
    return sortedClassCount[0][0]

####示例1：使用k-近邻算法改进约会网站配对效果

# 读取文本数据输出为训练样本矩阵和类标签向量
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 取出每列的最小值，1X3
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 测试分离器
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in xrange(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :],
            datingLabels[numTestVecs:m], 3)
        print "The classifier came back with: {result}, the real answer is: {lable}".format(result = classifierResult, lable = datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "The total error rate is: {errorRate}".format(errorRate = errorCount/float(numTestVecs))

# 预测函数
def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    percentGame = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentGame, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print "You will probably like this person: {result}".format(result = resultList[classifierResult - 1])

####示例2：识别手写数字

# 将图像转换为测试向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in xrange(32):
        lineStr = fr.readline()
        for j in xrange(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

TRAINING_DIR = 'digits/trainingDigits'
TESTING_DIR = 'digits/testDigits'

# 手写数字识别测试代码
def handwritingClassTest():
    hwLables = []
    trainingFileList = listdir(TRAINING_DIR)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in xrange(m):
        fileNameStr = trainingFileList[i]  # 1_52.txt代表数字1的第52个样本
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLables.append(classNumStr)
        trainingMat[i, :] = img2vector('{trainingDir}/{fileName}'.format(trainingDir
             = TRAINING_DIR, fileName = fileNameStr))
    testFileList = listdir(TESTING_DIR)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in xrange(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('{testingDir}/{fileName}'.format(testingDir
             = TESTING_DIR, fileName = fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLables, 3)
        print "The classifier came back with: {result}, the real answer is: {label}".format(result = classifierResult, label = classNumStr)
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print "\nThe total number of errors is: {errorCount}".format(errorCount = errorCount)
    print "\nTe total error rate is: {errorRate}".format(errorRate = errorCount / float(mTest))




