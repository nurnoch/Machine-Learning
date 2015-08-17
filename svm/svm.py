#coding=utf-8
from numpy import *

#### SMO算法的辅助函数

# 导入数据
def loadDataSet(fileName):
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat, labelMat

# 选取第二个alpha的下标
def selectJrand(i, m):
	j = i
	while(j == i):
		j = int(random.uniform(0, m))
	return j

# 调整大于H或小于L的alpha值
def clipAlpha(aj, H, L):
	if aj > H:
		aj = H
	if aj < L:
		aj = L
	return aj

# 简化版的SMO
# 5个参数分别为：数据集、类别标签、常数C、容错率、最大循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	# 转换成numpy矩阵
	dataMat = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	m, n = shape(dataMat)
	alphas = mat(zeros((m, 1))) # alpha列矩阵，初始化为0
	b = 0  # 参数b
	iter = 0
	while (iter < maxIter):
		alphaPairsChanged = 0 # 对alpha的优化是否work
		for i in range(m):
			# 计算预测值：w.x + b
			fXi = float(multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + b
			Ei = fXi - float(labelMat[i]) # 计算预测和实际的误差
			# 误差较大，就进行优化
			if((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
				j = selectJrand(i, m)
				fXj = float(multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T)) + b
				Ej = fXj - float(labelMat[j])
				alphaIold = alphas[i].copy()
				alphaJold = alphas[j].copy()
				# alphaJNew的取值范围
				if(labelMat[i] != labelMat[j]):
					L = max(0, alphas[j] - alphas[i])
					H = min(C, C + alphas[j] - alphas[i])
				else:
					L = max(0, alphas[j] + alphas[i] - C)
					H = min(C, alphas[j] + alphas[i])
				if L == H:
					print "L == H"
					continue
				# eta = 2K_ij - K_ii - K_jj	
				eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - dataMat[i, :] * dataMat[i, :].T - dataMat[j, :] * dataMat[j, :].T
				if eta >= 0:
					print "eta >= 0"
					continue
				# 约束方向未经剪辑的第二个alpha
				alphas[j] -= labelMat[j] * (Ei - Ej) / eta
				alphas[j] = clipAlpha(alphas[j], H, L) # 剪辑后
				if (abs(alphas[j] - alphaJold) < 0.00001):
					print "j not moving enough"
					continue
				# 更新第一个alpha
				alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
				# 更新b
				b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMat[i,:]*dataMat[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMat[j,:]*dataMat[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                	b = b1
                elif (0 < alphas[i]) and (C > alphas[j]):
                	b = b2
                else:
                	b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
        if (alphaPairsChanged == 0):
        	iter += 1
        else:
        	iter = 0
        print "iteration number: %d" % iter
        return b,alphas


