#coding=utf-8
from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


#### 计算相似度的几种方法

def ecludSim(inA, inB):
	return 1.0 / (1.0 + la.norm(inA - inB))

def pearsSim(inA, inB):
	return 0.5 + 0.5 * corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA, inB):
	num = float(inA.T * inB)
	denom = la.norm(inA) * la.norm(inB)
	return 0.5 + 0.5 * (num / denom)

#### 示例：餐馆菜肴推荐

# 给定相似度计算方法simMeas下，用户user对item的预估评分
def standEst(dataMat, user, simMeas, item):
	n = shape(dataMat)[1] # n个物品
	simTotal = 0.0
	ratSimTotal = 0.0
	for j in range(n):
		userRating = dataMat[user, j] # 用户use对j物品的评分
		if userRating == 0:
			continue
		# 对两项物品item和j都有评分的用户
		overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
		if len(overLap) == 0:
			similarity = 0
		else:
			similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
		simTotal += similarity
		ratSimTotal += similarity * userRating
	if simTotal == 0:
		return 0
	else:
		return ratSimTotal / simTotal # 归一化使得评级为1-5

# 推荐引擎: 对用于user，产生N个推荐结果
def recommend(dataMat, user, N = 3, simMeas = cosSim, estMethod = standEst):
	unratedItems = nonzero(dataMat[user, :].A == 0)[1] # 该用户未评分的物品
	if len(unratedItems) == 0:
		print "You rated everything"
	itemScores = []
	for item in unratedItems:
		estimateScore = estMethod(dataMat, user, simMeas, item)
		itemScores.append((item, estimateScore))
	return sorted(itemScores, key = lambda it: it[1], reverse = True)[:N]

# 利用SVD的评分估计
def svdEst(dataMat, user, simMeas, item):
	n = shape(dataMat)[1]
	simTotal = 0.0
	ratSimTotal = 0.0
	U, Sigma, VT = la.svd(dataMat)
	Sig4 = mat(eye(4) * Sigma[:4]) # 中间的sigma对角矩阵
	xformedItems = dataMat.T * U[:, :4] * Sig4.I # 利用U矩阵将物品转换到低维空间中(???)
	for j in range(n):
		userRating = dataMat[user, j] # 用户use对j物品的评分
		if userRating == 0 or j == item:
			continue
		similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
		simTotal += similarity
		ratSimTotal += similarity * userRating
	if simTotal == 0:
		return 0
	else:
		return ratSimTotal / simTotal # 归一化使得评级为1-5

#### 基于SVD的图像压缩

def printMat(inMat, thresh = 0.8):
	for i in range(32):
		for j in range(32):
			if float(inMat[i, j]) > thresh:
				print 1,
			else:
				print 0,
		print ' '

def imgCompress(numSv = 3, thresh = 0.8):
	myl = []
	for line in open('0_5.txt').readlines():
		newRow = []
		for i in range(32):
			newRow.append(int(line[i]))
		myl.append(newRow)
	myMat = mat(myl)
	print "****original matrix*****"
	printMat(myMat, thresh)
	U, Sigma, VT = la.svd(myMat)
	SigRecon = mat(zeros((numSv, numSv)))
	for k in range(numSv):
		SigRecon[k, k] = Sigma[k]
	reconMat = U[:, :numSv] * SigRecon * VT[:numSv, :]
	print "****reconstructed matrix using %d singular values*****" % numSv
	printMat(reconMat, thresh)












