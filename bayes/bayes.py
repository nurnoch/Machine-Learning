#coding=utf-8
from numpy import *
import re

# 创建实验样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1侮辱性，0非侮辱性
    return postingList,classVec

# 创建在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

# 将输入转换为词向量：词集模型
def setOfWords2Vec(vocabList, inputSet):
	retVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			retVec[vocabList.index(word)] = 1
		else:
			print "The word: {word} is not in the vocabulary!".format(word = word)
	return retVec

# 将输入转换为词向量：词袋模型
def bagOfWords2Vec(vocabList, inputSet):
	retVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			retVec[vocabList.index(word)] += 1
	return retVec


# 朴素贝叶斯分类器训练
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory) / float(numTrainDocs) # p(c1)
	p0Num = ones(numWords) # 类别0，词向量每个位置的单词出现的次数
	p1Num = ones(numWords)
	p0Denom = 2.0  # 分母，出现类别0的单词总数
	p1Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:  
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = log(p1Num / p1Denom)  # log(每个词出现的概率)
	p0Vect = log(p0Num / p0Denom)
	return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类器
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	listOfPosts, listOfClasses = loadDataSet()
	myVocabList = createVocabList(listOfPosts)
	trainMat = []
	for postinDoc in listOfPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	p0V, p1V, pAb = trainNB0(trainMat, listOfClasses)
	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
	testEntry = ['stupid', 'garbage']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

#### 示例：垃圾邮件过滤

# 切分文本
def textParse(bigString):
	listOfTokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 使用朴素贝叶斯进行交叉验证
def spamTest():
	docList = []
	classList = []
	fullText = []
	for i in range(1, 26):
		wordList = textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1) # spam email
		wordList = textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0) # non-spam email

	vocabList = createVocabList(docList)
	trainingSet = range(50) # 一共有50封邮件
	testSet = []
	# 随机构建测试集，选取其中的10个文件，同时从训练集中删除
	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	# 训练
	trainMat = []
	trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
	
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList, docList[docIndex])
		if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1
	print 'The error rate is: ', float(errorCount) / len(testSet)


















































