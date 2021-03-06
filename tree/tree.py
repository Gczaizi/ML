# coding: utf-8

from math import log
import operator
import treePlotter


def calcShannonEnt(dataSet):
    # 计算数据集的香农熵
    numEntries = len(dataSet)
    labelCounts = {}

    # 为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        else:
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
    # print(labelCounts)
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        # print(prob)
        shannonEnt -= prob * log(prob,2)
    # print(shannonEnt)

    return shannonEnt


def createDataSet():
    dataSet = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
        ]
    labels = ['no surfacing', 'flippers']

    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print(infoGain)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
            # print(bestInfoGain, bestFeature)

    return bestFeature


# 返回出现次数最多的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    # print(dataSet)
    classList = [example[-1] for example in dataSet]    # 包含数据集的所有类标签
    # print(classList)
    
    # 类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList): # classList[0] 出现了 len(classList) 次
        return classList[0]
    
    # 遍历完所有特征时返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    
    # 得到列表包含的所有属性值
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    # print(bestFeat, featValues)
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree


# 决策树分类器
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]

    # 将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)  # index 方法查找 featLabels 中第一个匹配 firstStr 的元素位置
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                # 到达叶子节点，返回节点分类标签
                classLabel = secondDict[key]
    
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    myDat, labels = createDataSet()
    # myTree = createTree(myDat, labels)
    # storeTree(myTree, 'classifierStorage.txt')    # 储存决策树到文件
    myTree = grabTree('classifierStorage.txt')
    # myDat, labels = createDataSet()
    # print(labels)
    # print(myTree)
    # myTree['no surfacing'][3] = 'maybe'
    # treePlotter.createPlot(myTree) # 画出决策树
    # print(treePlotter.getNumLeafs(myTree))
    # print(treePlotter.getTreeDepth(myTree))