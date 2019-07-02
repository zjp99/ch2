# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN
from numpy import *


def classify0(inX, dataSet, labels, k):#四个输入参数：输入向量，训练样本集，标签向量，最近邻居的数目
    #1.计算距离
    dataSetSize = dataSet.shape[0]#计算训练集维数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet#行重复四次，计算出差值数组并转化为差值矩阵
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5#计算出到所有点的欧氏距离
    #2.距离排序
    sortedDistIndicies = distances.argsort()#返回数组从小到大的索引值
    #选择距离最小的点
    classCount = {}
    for i in range(k):#从0-k不包含k
        voteIlabel = labels[sortedDistIndicies[i]]#取出前k个是哪个类型
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #计算每个分类的个数classCount = {“A”:2,"B":2}
    # 3.降序，将classCount字典分解为元组列表，并且按第二个元素进行降序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]#出现频数最高的那个对应的分类标签

def img2vector(filename):
    #创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    #打开文件
    fr = open(filename)
    #按行读取
    for i in range(32):
        #读一行数据
        lineStr = fr.readline()
        #每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    #返回转换后的1x1024向量
    return returnVect


def handwritingClassTest():
    #错误文件名列表
    error_dict={"错误文件名":"识别结果"}
   # empty= []
    #测试集的Labels
    hwLabels = []
    #返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')
    #返回文件夹下文件的个数
    m = len(trainingFileList)
    #初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, 1024))
    #从文件名中解析出训练集的类别
    for i in range(m):
        #获得文件的名字
        fileNameStr = trainingFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        #将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
    #构建kNN分类器
    neigh = kNN(n_neighbors = 3, algorithm = 'auto')
    #拟合模型, trainingMat为测试矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    #返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    #从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #获得分类的真实数字
        classNumber = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        #获得预测结果
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
       #classifierResult = neigh.predict(vectorUnderTest)
        print("序号为%s\t分类返回结果为%d\t真实结果为%d" % (fileNameStr,classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
            error_dict[fileNameStr]=str(classifierResult)
            #empty.append(fileNameStr)
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))
    print(error_dict)
    #print(empty)


if __name__ == '__main__':
    handwritingClassTest()
