#coding=utf-8
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

def main():
    # 创建训练集
    dataset = genfromtxt(open('train.csv', 'r'), delimiter = ',', dtype = 'f8')[1:]
    target = [x[0] for x in dataset] # 第一列为label
    train = [x[1:] for x in dataset]

    # 创建测试集
    test = genfromtxt(open('test.csv', 'r'), delimiter = ',', dtype = 'f8')[1:]

    # 创建并且训练一个随机森林模型
    rf = RandomForestClassifier(n_estimators = 100)
    rf.fit(train, target)
    predicted_results = [[index + 1, x] for index, x in enumerate(rf.predict(test))]

    # 利用随机森林对测试集进行预测，并将结果保存到输出文件中
    savetxt('result.csv', predicted_results, delimiter = ',', fmt = '%d,%d', 
            header = 'ImageId,Label', comments = '')

if __name__ == '__main__':
    main()
