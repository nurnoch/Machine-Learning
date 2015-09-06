#coding=utf-8

from numpy import *

# 创建测试数据集
def loadDataSet():
	return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# 构建集合大小为1的所有候选集
def createC1(dataSet):
	C1 = []
	for transaction in dataSet:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
	C1.sort()
	return map(frozenset, C1) # turn list to frozenset, so we can use it as a key in dict

# 返回满足最小支持度的项集及其对应的支持度
def scanD(D, Ck, minSupport): # 数据集、候选项集、最低支持度
	ssCnt = {}
	for tid in D: # 遍历所有交易记录
		for can in Ck:  # 遍历所有的候选项集
			if can.issubset(tid): # 当前候选项集是当前记录的一部分
				ssCnt[can] = ssCnt.get(can, 0) + 1;

	numItems = float(len(D))
	retList = []
	supportData = {}
	for key in ssCnt:
		support = ssCnt[key] / numItems
		if support >= minSupport:
			retList.insert(0, key) # insert at pos 0
		supportData[key] = support
	return retList, supportData

# 利用Lk来创建Ck+1
def aprioriGen(Lk, k):
	retList = []
	lenLk = len(Lk)
	for i in range(lenLk): # 比较Lk中的每一个元素与其他元素
		for j in range(i+1, lenLk):
			L1 = list(Lk[i])[:k-2]
			L2 = list(Lk[j])[:k-2]
			L1.sort()
			L2.sort()
			if L1 == L2: # 前k-2个元素都相等，将两集合合并为一个大小为k+1的集合
				retList.append(Lk[i] | Lk[j])
	return retList

# 生成所有候选集列表
def apriori(dataSet, minSupport=0.5):
	C1 = createC1(dataSet)
	D = map(set, dataSet)
	L1, supportData = scanD(D, C1, minSupport)
	L = [L1]
	k = 2
	while(len(L[k-2]) > 0): # 直到项集为空
		Ck = aprioriGen(L[k-2], k)
		Lk, supK = scanD(D, Ck, minSupport)
		supportData.update(supK) # add new support data to dict
		L.append(Lk)
		k += 1
	return L, supportData


# 生成关联规则
def generateRules(L, supportData, minConf=0.7):
	bigRuleList = []
	for i in range(1, len(L)):
		for freqSet in L[i]:
			H1 = [frozenset([item]) for item in freqSet]
			if (i > 1):
				rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
			else:
				calConf(freqSet, H1, supportData, bigRuleList, minConf)
	return bigRuleList

# 生成候选规则集合
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
	m = len(H[0])
	if (len(freqSet) > (m+1)):
		Hmp1 = aprioriGen(H, m+1)
		Hmp1 = calConf(freqSet, Hmp1, supportData, brl, minConf)
		if(len(Hmp1) > 1):
			rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

# 计算置信度
def calConf(freqSet, H, supportData, brl, minConf=0.7):
	prunedH = []
	for conseq in H:
		conf = supportData[freqSet] / supportData[freqSet - conseq]
		if conf >= minConf:
			print freqSet - conseq, '->', conseq, 'conf:', conf
			brl.append((freqSet - conseq, conseq, conf))
			prunedH.append(conseq)
	return prunedH


































