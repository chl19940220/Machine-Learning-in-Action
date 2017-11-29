import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext = centerPt, textcoords = 'axes fraction',
                            va='center', ha='center', bbox=nodeType,
                            arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('A Decision Node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('A Leaf Node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

"""
获取叶节点的数目
Parameters：
     之前步骤构造的决策树，结构：{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
Return：
    决策树的叶节点数
"""
def getNumLeafs(myTree):
    numLeafs = 0

    firstStr = next(iter(myTree))
    secondStr = myTree[firstStr]
    for key in secondStr.keys():
        if type(secondStr[key]).__name__ == 'dict': #如果是字典结构，递归求树的叶节点数
            numLeafs += getNumLeafs(secondStr[key])
        else:
            numLeafs += 1
    return numLeafs

"""
获取树的层数
Parameters：
     之前步骤构造的决策树，结构：{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
Return：
    决策树的层数
"""
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondStr = myTree[firstStr]
    for key in secondStr.keys():
        if type(secondStr[key]).__name__ == 'dict': #如果是字典结构，递归求树的叶节点数
            thisDepth = 1 + getTreeDepth(secondStr[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

"""
在父子节点之间插入文本信息
"""
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (cntrPt[0] + parentPt[0]) / 2 #文本信息的横坐标
    yMid = (cntrPt[1] + parentPt[1]) / 2 #文本信息的纵坐标
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]

myTree = retrieveTree(0)
#createPlot(myTree)


def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]  # 下一个字典
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

"""
使用pickle模块存储决策树
"""
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

storeTree(myTree,"decesiontree.txt")
print(grabTree("decesiontree.txt"))