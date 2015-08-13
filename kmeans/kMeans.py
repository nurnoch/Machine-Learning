#coding=utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

# 导入数据到一个列表中
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
    	curLine = line.strip().split('\t')  # num1	num2
    	fltLine = map(float, curLine)
    	dataMat.append(fltLine)
    return dataMat

# 计算欧氏距离
def distEclud(vecA, vecB):
	return sqrt(sum(power(vecA - vecB, 2)))

# 返回k个随机的簇中心
def randCent(dataSet, k):
	n = shape(dataSet)[1] 
	centroids = mat(zeros((k, n)))
	for j in xrange(n):
		minJ = min(dataSet[:, j]) # 当前列的最小值
		rangeJ = float(max(dataSet[:, j]) - minJ)
		centroids[:, j] = minJ + rangeJ * random.rand(k, 1) # 范围为[minJ, minJ + rangeJ] 
	return centroids

# K-means算法
def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m, 2))) # [clusterIndex, Dist]
	centroids = randCent(dataSet, k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = inf
			minIndex = -1
			for j in xrange(k): # 分别计算该点与k个质心的距离
				distJI = distEclud(dataSet[i, :], dataSet[j, :])
				if distJI < minDist:
					minDist = distJI
					minIndex = j
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
			clusterAssment[i, :] = minIndex, minDist**2

		print centroids
		for cent in xrange(k):
			ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
			centroids[cent, :] = mean(ptsInClust, axis = 0)
	return centroids, clusterAssment

# bisecting K-means	
def biKmeans(dataSet, k, distMeas = distEclud):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros(((m, 2))))  # 存储簇分配结果和平方误差
	centroid0 = mean(dataSet, axis = 0).tolist()[0]
	cenList = [centroid0]
	for j in xrange(m):
		clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
	while (len(cenList) < k):
		lowestSSE = inf
		for i in range(len(cenList)):
			ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
			centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
			sseSplit = sum(splitClustAss[:, 1])
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
			print "sseSplit, and sseNotSplit: ", sseSplit, sseNotSplit
			if (sseSplit + sseNotSplit) < lowestSSE:
				bestCentToSplit = i
				bestNewCents = centroidMat
				bestClustAss = splitClustAss.copy()
				lowestSSE = sseSplit + sseNotSplit
		bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(cenList)
		bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
		print "the bestCentToSplit is: ", bestCentToSplit
		print "the len of bestClustAss is: ", len(bestClustAss)
		cenList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
		cenList.append(bestNewCents[1, :].tolist()[0])
		clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
	return mat(cenList), clusterAssment

#### 示例：对地理坐标进行聚类

# 返回地球表面两点间的距离
def distSLC(vecA, vecB):
	a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
	b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
	return arccos(a + b) * 6371.0

# 聚类并且画图
def clusterClubs(numClust = 5):
	datList = []
	fr = open("places.txt")
	for line in fr.readlines():
		lineArr = line.split('\t')
		# 第4列和第5列分别表示：纬度和经度 	
		datList.append([float(lineArr[4]), float(lineArr[3])]) 
	datMat = mat(datList) 
	myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas = distEclud)

	fig = plt.figure()
	rect = [0.1, 0.1, 0.8, 0.8]
	scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
	axprops = dict(xticks = [], yticks=[])
	ax0 = fig.add_axes(rect, label = 'ax0', **axprops)
	imgP = plt.imread('Portland.png')
	ax0.imshow(imgP)
	ax1 = fig.add_axes(rect, label = 'ax1', frameon = False)
	for i in range(numClust):
		ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
		markerStyle = scatterMarkers[i % len(scatterMarkers)]
		ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], 
						marker = markerStyle, s = 90)
	ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], 
					marker = '+', s = 300)
	plt.show()


















