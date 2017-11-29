from math import log
import operator

"""
创建数据集
return：
     数据集和标签
"""
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers'] #dataSet[i][0]为1，代表'no surfacing'条件为正
    return dataSet, labels


"""
计算给定数据集的香农熵
Parameters:
      dataSet:给定的数据集

Return：
      数据集的香农熵
"""
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}  # 定义一个dict
    for featVec in dataSet:
        currentLabel = featVec[-1]  # dataSet最后一列代表类别
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


"""
按照给定特征划分数据集
Parameters:
      dataSet：给定的数据集
      axis：选择的某个属性
      value：该属性的值

Return：
       dataSet中所有axis属性值等于value的所有数据集
"""
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis + 1:])  # 去掉axis属性
            retDataSet.append(reducedFeatVec)
    return retDataSet


"""
选择最好的属性进行划分
Parameters:
     dataSet：数据集合

Return：
      能使信息增益最大的属性的下标
"""
def chooseBestTeatureToSplit(dataSet):
    baseEntropy = calcShannonEnt(dataSet)  # 计算熵
    numFeatures = len(dataSet[0]) - 1  # 总的特征数，最后一列是类别，所以要减去1
    baseInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 对value去重，集合不能有重复元素
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 计算经验条件熵
        infoGain = baseEntropy - newEntropy
        if (infoGain > baseInfoGain):
            baseInfoGain = infoGain  # 找到能使信息增益最大的划分属性
            bestFeature = i
    return bestFeature


"""
返回出现次数最多的分类名称
"""
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

"""
创建一棵决策树
"""
def createDecisionTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0] #如果数据集都属于同一个标签，则直接返回这个标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList) #如果只剩下一个属性，则返回这个数据集中数量最多的那个标签
    bestFeat = chooseBestTeatureToSplit(dataSet) #在数据集中选择最好的属性进行划分
    bestFeatLabel = labels[bestFeat] #这个下标对应的属性名称
    myTree = {bestFeatLabel: {}} #使用dict类型来保存树的信息
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createDecisionTree(splitDataSet(dataSet, bestFeat, value), subLabels) #递归创建树
    return myTree

myDat, labels = createDataSet()
myTree = createDecisionTree(myDat, labels)
print(myTree)
