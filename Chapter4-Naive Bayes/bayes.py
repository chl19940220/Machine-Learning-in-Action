from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec

"""
将数据集整理成不含重复元素的词汇列表
Parameters：
    dataSet：初始数据集
Return：
    词汇列表
"""
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:print("the word: %s is not in my Vocabulary!" %word)
    return returnVec

"""
计算P(class=1)，每个单词在class=1以及class=0时的概率矩阵
Parameters:
    trainMatrix:词向量，对每一组语句，使用上面的setOfWords2Vec()方法计算这条语句中的词是否在整个词汇表中出现
    trainCategory:类别向量，由loadDataSet()方法中的classVec得到
"""
def trainNB0(trainMatrix, trainCategory):
    dataSetLength = len(trainCategory) #用于测试的数据的规模
    """首先计算class=1的概率"""
    p1 = sum(trainCategory) / float(dataSetLength)
    """分别计算每个词在class=1和class=0时的概率 p=出现次数/这个类别中词汇总数"""
    vocabularyNumber = len(trainMatrix[0])#词汇表中包含词的总数
    p0Vec=zeros(vocabularyNumber) #定义一个初始值全为0的向量，用来保存每个词在class=0时的概率
    p1Vec = zeros(vocabularyNumber) #定义一个初始值全为0的向量，用来保存每个词在class=1时的概率
    p0Num = 0.0; p1Num = 0.0 #统计每个类别词汇总数
    for i in range(dataSetLength):
        if trainCategory[i] == 1:
            p1Vec += trainMatrix[i]
            p1Num += sum(trainMatrix[i])
        else:
            p0Vec += trainMatrix[i]
            p0Num += sum(trainMatrix[i])
    return p1, p1Vec/p1Num, p0Vec/p0Num



"""
计算P(class=1)，每个单词在class=1以及class=0时的概率矩阵
为了避免出现其中一个概率值为0，导致最后乘机为0的情况，将所有词的出现次数初始化为1，并将分母初始化为2
为了避免程序下溢出，将返回值用对数处理
Parameters:
    trainMatrix:词向量，对每一组语句，使用上面的setOfWords2Vec()方法计算这条语句中的词是否在整个词汇表中出现
    trainCategory:类别向量，由loadDataSet()方法中的classVec得到
"""
def trainNB1(trainMatrix, trainCategory):
    dataSetLength = len(trainCategory) #用于测试的数据的规模
    """首先计算class=1的概率"""
    p1 = sum(trainCategory) / float(dataSetLength)
    """分别计算每个词在class=1和class=0时的概率 p=出现次数/这个类别中词汇总数"""
    vocabularyNumber = len(trainMatrix[0])#词汇表中包含词的总数
    """为了避免出现其中一个概率值为0，导致最后乘机为0的情况，将所有词的出现次数初始化为1，并将分母初始化为2"""
    p0Vec=ones(vocabularyNumber) #定义一个初始值全为0的向量，用来保存每个词在class=0时的概率
    p1Vec = ones(vocabularyNumber) #定义一个初始值全为0的向量，用来保存每个词在class=1时的概率
    p0Num = 2.0; p1Num = 2.0 #统计每个类别词汇总数
    for i in range(dataSetLength):
        if trainCategory[i] == 1:
            p1Vec += trainMatrix[i]
            p1Num += sum(trainMatrix[i])
        else:
            p0Vec += trainMatrix[i]
            p0Num += sum(trainMatrix[i])
    return p1, log(p1Vec/p1Num), log(p0Vec/p0Num)

"""
对词条进行分类
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1-pClass1)
    if p1>p0: return 1
    else: return 0

def testNB():
    postingList, classVec = loadDataSet()
    vocabularyList = createVocabList(postingList)
    print(vocabularyList)

    trainMatrix = []
    for postinDoc in postingList:
        trainMatrix.append(setOfWords2Vec(vocabularyList, postinDoc))

    p1, p2, p3 = trainNB1(trainMatrix, classVec)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(vocabularyList, testEntry)
    print(str(testEntry) + " classified as: " + str(classifyNB(thisDoc, p3,p2,p1)))
    testEntry = ['stupid', 'garbage', 'dalmation']
    thisDoc = setOfWords2Vec(vocabularyList, testEntry)
    print(str(testEntry) + " classified as: " + str(classifyNB(thisDoc, p3, p2, p1)))

testNB()



