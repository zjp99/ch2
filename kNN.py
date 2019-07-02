
from numpy import *
import operator
from os import listdir
import numpy as np

def classify0(inX, dataSet, labels, k):#四个输入参数：输入向量，训练眼样本集，标签向量，最近邻居的数目
    #计算距离
    dataSetSize = dataSet.shape[0]#计算训练集维数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet#行重复四次，计算出差值数组并转化为差值矩阵
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5#计算出到所有点的欧氏距离
    sortedDistIndicies = distances.argsort()#返回数组从小到大的索引值
    #选择距离最小的点
    classCount = {}
    for i in range(k):#从0-k不包含k
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #计算每个分类的个数classCount = {“A”:2,"B":2}
    # 降序，将classCount字典分解为元组列表，并且按第二个元素进行降序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]#出解现频数最高的那个对应的分类标签

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 获取文件的行数
    returnMat = zeros((numberOfLines, 3))  # 创建和数据集大小一样的矩阵
    classLabelVector = []  # 解析文件数据
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet= zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals



def datingClassTest():
    hoRatio = 0.10  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print( "分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print( "错误率:%f%%" % (errorCount / float(numTestVecs)))


if __name__ == '__main__':
    datingClassTest()