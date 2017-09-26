# coding: utf-8

from numpy import *


def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0,1,0,1,0,1]

    return postingList, classVec


def createVocabList(dataSet):
    
    # 创建一个空集
    vocabSet = set([])
    for document in dataSet:

        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):

    # 创建一个其中所含元素都为 0 的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word: %s is not in my Vocabulary!" % word)
    
    return returnVec


def bagOfWords2Vec(vocabList, inputSet):

    # 创建一个其中所含元素都为 0 的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 侮辱性文档概率
    
    # 初始化概率
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:

            # 向量相加
            p1Num += trainMatrix[i] # 侮辱性文档词频加 1
            p1Denom += sum(trainMatrix[i])  # 侮辱性文档总词数累加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 对每个元素做除法
    p1Vect = log(p1Num / p1Denom)    # 每个单词在类别 1 下出现的概率
    p0Vect = log(p0Num / p0Denom)    # 每个单词在类别 0 下出现的概率

    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):

    # 元素相乘
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)

    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):

        # 导入并解析文本文档
        wordList = textParse(open(('/mnt/e/Study/data/email/spam/%d.txt') % i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open(('/mnt/e/Study/data/email/ham/%d.txt') % i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []

    # 随机创建测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0

    # 对测试集分类
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))

    return float(errorCount) / len(testSet)


if __name__ == '__main__':
    #listOPosts, listClasses = loadDataSet()
    #myVocabList = createVocabList(listOPosts)
    # print(setOfWords2Vec(myVocabList, listOPosts[0]))
    # print(myVocabList)
    #testingNB()
    b = 0.0
    for i in range(1000):
        a = spamTest()
        b += a
    print(b / 1000.0)