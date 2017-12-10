# _*_ coding:utf-8 _*_

from numpy import *
import operator
from os import listdir

def creatDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# k-Nearest Neighbors algorithm
# 第一个kNN分类器  inX-测试数据 dataSet-样本数据  labels-标签 k-邻近的k个样本
# the first kNN Classifier, inX-testData,dataSet-trainingData, labels, k- k nearest neighbors
def classify0(inX, dadaSet, labels, k):
    dadaSetSize = dadaSet.shape[0]
    # Distance calculation 计算距离
    diffMat = tile(inX, (dadaSetSize, 1)) - dadaSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()
    classCount = {}

    # 选择距离最小的k个点
    # Voting with lowest k distances
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # Sort dictionary 排序
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# Text record to NumPy parsing code
# 将文本记录到转换numPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# Data-normalizing code 自动归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    norDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    norDataSet = dataSet - tile(minVals, (m, 1))
    norDataSet = norDataSet / tile(ranges, (m, 1))
    return norDataSet, ranges, minVals

# Classifier testing code for dating site
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    norMat, ranges, minVals = autoNorm(datingDataMat)
    m = norMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(norMat[i,:], norMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m], 3)
        print "The classifier came back with: %d, the real answer is: %d"\
              % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "The total error rate is: %f" % (errorCount / float(numTestVecs))

# Dating site predictor function
def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person: ",resultList[classifierResult - 1]

# converting images into test vectors
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

# Handwritten digits testing
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_',)[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "The classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0

    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))


# 1.function createDataSet()
if False:
    group, labels = creatDataSet()
    print group
    print labels

# 2.To predict the class wiht function classify0()
if False:
    group, labels = creatDataSet()
    print classify0([0, 0], group, labels, 3)

# 3. To test function file2matrix()
if False:
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    print datingDataMat
    print datingLabels

# 4. creating scatter plots with Matplotlib (run with 3 is True)
if False:
    import matplotlib.pyplot as plt

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    plt.show()

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
               15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()

# 5.To try out autoNorm (run with 3 is True)
if False:
    norMat, ranges, minVals = autoNorm(datingDataMat)
    print norMat
    print ranges

# 6.Classifier testing code for dating site
if False:
    print datingClassTest()

# 7.To see the program in action with running classifyPerson()
if False:
    classifyPerson()

# 8.converting images into test vectors
if False:
    testVector = img2vector('testDigits/0_13.txt')
    print testVector[0, 0:31]
    print  testVector[0, 32:63]

# 9.Handwritten digits testing
"""The dataset is a modified version of the “Optical Recognition 
of Handwritten Digits Data Set” by E. Alpaydin, C. Kaynak, Department
of Computer Engineering at Bogazici University, 80815 Istanbul Turkey,
retrieved from the UCI Machine Learning Repository (http://archive.ics.uci.edu/ml) 
"""
if False:
    handwritingClassTest()