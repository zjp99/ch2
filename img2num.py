# -*- coding: utf-8 -*-#

import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN
from numpy import *

def picTo01(filename):
    """
    将图片转化为32*32像素的文件，用0 1表示
    """
    # 打开图片
    img = Image.open(filename).convert('RGBA')

    # 得到图片的像素值
    raw_data = img.load()

    # 将其降噪并转化为黑白两色
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if raw_data[x, y][0] < 90:
                raw_data[x, y] = (0, 0, 0, 255)

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if raw_data[x, y][1] < 136:
                raw_data[x, y] = (0, 0, 0, 255)
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if raw_data[x, y][2] > 0:
                raw_data[x, y] = (255, 255, 255, 255)
    # 设置为32*32的大小
    img = img.resize((32, 32), Image.LANCZOS)

    # 进行保存，方便查看
    imgName=filename.split('/',2)[2]
    # print(imgName)
    img.save('image/test/%s' %(imgName))

    # 得到像素数组，为(32,32,4)
    array = plt.array(img)

    # 按照公式将其转为01, 公式： 0.299 * R + 0.587 * G + 0.114 * B

    gray_array = np.zeros((32, 32))

    # 行数
    for x in range(array.shape[0]):
        # 列数
        for y in range(array.shape[1]):
            # 计算灰度，若为255则白色，数值越小越接近黑色
            gary = 0.299 * array[x][y][0] + 0.587 * array[x][y][1] + 0.114 * array[x][y][2]

            # 设置一个阙值，记为0
            if gary == 255:
                gray_array[x][y] = 0
            else:
                # 否则认为是黑色，记为1
                gray_array[x][y] = 1

    # 得到对应名称的txt文件
    name01 = imgName.split('.')[0]
    name01 = 'image/txt/'+name01 + '.txt'
    # 保存到文件中
    np.savetxt(name01, gray_array, fmt='%d', delimiter='')

def classify0(inX, dataSet, labels, k):#四个输入参数：输入向量，训练样本集，标签向量，最近邻居的数目
    #计算距离
    dataSetSize = dataSet.shape[0]#计算训练集维数
    #行重复四次，计算出差值数组并转化为差值矩阵
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5#计算出到所有点的欧氏距离
    sortedDistIndicies = distances.argsort()#返回数组从小到大的索引值
    #选择距离最小的点
    classCount = {}
    for i in range(k):#从0-k不包含k
        voteIlabel = labels[sortedDistIndicies[i]]
        #计算每个分类的个数classCount = {“A”:2,"B":2}
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 降序，将classCount字典分解为元组列表，并且按第二个元素进行降序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]#出解现频数最高的那个对应的分类标签

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
    neigh = kNN(n_neighbors =3, algorithm = 'auto')
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
        classifierResult = neigh.predict(vectorUnderTest)
        print("序号为%s\t分类返回结果为%d\t真实结果为%d" % (fileNameStr,classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
            error_dict[fileNameStr]=str(classifierResult)
            #empty.append(fileNameStr)
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, (1-errorCount/mTest) * 100))
    print(error_dict)


def imgTest():
    error_dict={"错误文件名":"识别结果"}
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
    neigh = kNN(n_neighbors = 3, algorithm = 'auto')
    neigh.fit(trainingMat, hwLabels)
    testFileList = listdir('image/txt')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(str(fileNameStr[0]))
        vectorUnderTest = img2vector('image/txt/%s' % (fileNameStr))
        classifierResult = neigh.predict(vectorUnderTest)
        print("序号为%s\t分类返回结果为%d\t真实结果为%s" % (fileNameStr,classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
            error_dict[fileNameStr]=str(classifierResult)
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))
    print(error_dict)

def readFile(file):
    testFileList = listdir(file)
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = file+testFileList[i]
        picTo01(fileNameStr)
    imgTest()
if __name__ == '__main__':
    #handwritingClassTest()
    readFile('image/picture/')