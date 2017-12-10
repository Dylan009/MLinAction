# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     trees
   Description :
   Author :       Andrew
   date：          2017/12/10
-------------------------------------------------
   Change Activity:
                   2017/12/10:
-------------------------------------------------
"""
from math import log
import operator


def calcShannonEnt(dataSet):   # 计算给定数据集的香农熵
    numEntries = len(dataSet)  # 求长度
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 获得标签
        if currentLabel not in labelCounts.keys():  # 如果标签不在新定义的字典里创建该标签值
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 该类标签下含有数据的个数
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  # 同类标签出现的概率
        shannonEnt -= prob * log(prob, 2)   # 以2为底求对数
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

# Choosing the best feature to split on
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# voting portion
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classList.keys():
            classCount = 0
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount

# Tree-building code
def createTree(dataSet, labels):     # 创建树的函数代码
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 类别完全相同则停止划分
        return classList[0]
    if len(dataSet[0]) ==1:            # 遍历完所有特征值时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)   # 选择最好的数据集划分方式
    bestFeatLabel = labels[bestFeat]   # 得到对应的标签值
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])      # 清空labels[bestFeat],在下一次使用时清零
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # 递归调用创建决策树函数
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


myDat, labels = createDataSet()

if False:
    # 1.计算香农信息熵shannoEnt
    print(myDat)
    print(calcShannonEnt(myDat))

if False:
    # 2.更加复杂的情况
    myDat[0][-1] = 'maybe'
    print(myDat)
    print(calcShannonEnt(myDat))

if False:
    # 3.Dataset splitting on a given feature
    a = [1, 2, 3]
    b = [4, 5, 6]
    # noinspection PyTypeChecker
    a.append(b)  # a new a[]
    print(a)
    # If you do a.append(b), you have a list with four elements,
    # and the fourth element is a list.
    # However, if you do
    a.extend(b)  # a[] is the new a[] list
    print(a)
    # you now have one list with all the elements from new a and b

if False:
    # try out the splitDataSet()
    myDat, labels = createDataSet()
    print(myDat)
    print splitDataSet(myDat, 0, 1)
    print splitDataSet(myDat, 0, 0)

if False:
    # try out the chooseBestFearureToSplit()
    myDat, labels = createDataSet()
    print chooseBestFeatureToSplit(myDat), '\n', myDat

if True:
    myDat, labels = createDataSet()
    myTree = createTree(myDat, labels)
    print myTree
