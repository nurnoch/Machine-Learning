#coding=utf-8
import sys
from numpy import mat, mean, power

def read_input(file):
	for line in file:
		yield line.rstrip()

input = read_input(sys.stdin)
mapperOut = [line.split('\t') for line in input]
cumNum = 0.0 # 个数
cumVal = 0.0 # 总的值
cumSumSq = 0.0 # 总的平方值
for instance in mapperOut:
	num = float(instance[0])
	cumNum += num
	cumVal += float(instance[1]) * num
	cumSumSq += float(instance[2]) * num

mean = cumVal / cumNum
varSum = (cumSumSq + mean * mean * cumNum - 2 * mean * cumVal) / cumNum

print "%d\t%f\t%f" % (cumNum, mean, varSum)
print >> sys.stderr, "report: still alive" 