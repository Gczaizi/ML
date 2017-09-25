# coding: utf-8

from numpy import *
import operator


def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    
    # 计算距离
      
    # print(tile(inX, (dataSetSize, 1)))  # 复制 inX dataSetSize 次

    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # print(diffMat)
    sqDiffMat = diffMat ** 2
    # print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)  # 计算每个样本与输入的距离和
    # print(sqDistances)
    distances = sqDistances ** 0.5
    # print(distances)
    sortedDistIndicies = distances.argsort()    # 返回数组值从小到大的索引
    # print(sortedDistIndicies)
    classCount = {}
    
    # 选择距离最小的 k 个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # print(classCount.get(voteIlabel, 0))
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # 统计标签出现次数
        # print(classCount)
    
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # 获取出现最多的标签
    # print(sortedClassCount)
    
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOnlines = fr.readlines()
    numberOfLines = len(arrayOnlines)   # 获取文件行数
    returnMat = zeros((numberOfLines, 3))   # 返回矩阵
    classLabelVector = []
    index = 0

    # 解析文件数据到列表
    for line in arrayOnlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    
    return returnMat, classLabelVector


# 归一化函数
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.12  # 测试集比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)    # 测试集样本数量
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print(classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print(errorCount/float(numTestVecs))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of game time: "))
    ffMiles = float(input("flier miles: "))
    iceCream = float(input("icecream per year: "))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print(resultList[classifierResult - 1])