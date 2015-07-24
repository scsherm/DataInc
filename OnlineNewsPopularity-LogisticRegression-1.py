
# coding: utf-8

# # __Predicting An Article's Popularity Based On Total Shares__
# Samuel Sherman 
# 
# July, 2015

# ##_Data Examination_

# In[1]:

import os.path
baseDir = os.path.join('data')
inputPath = os.path.join('study', 'OnlineNewsPopularity.csv')
fileName = os.path.join(baseDir, inputPath)
numPartitions = 2
rawData = sc.textFile(fileName, numPartitions)
header = rawData.first()
rawData = rawData.filter(lambda x: x != header)
dataFeats = unicode.decode(header).split()

print dataFeats
print len(dataFeats)-3
print dataFeats[60]


# The data represents the different attributes of online news articles and includes a count of shares for that article. There are a total of 61 colmuns in the dataset and the "shares" and "timedelta" features will be used as the dependent variable or label.  Only 58 of the remaining features will be used as predictors for the model. This will not include the "url", which appears to have little relevance.   

# ##_Data Partition_

# In[2]:

# Load appropriate packages
from pyspark.mllib.regression import LabeledPoint 
from pyspark.mllib.linalg import SparseVector
import numpy as np


# In[3]:

weights = [.8, .1, .1] # 80% for training and 20% for validation and testing
seed = 42

# Use randomSplit with weights and seed, partition
rawTrainData, rawValidationData, rawTestData = rawData.randomSplit(weights, seed)

# Cache the data
rawTrainData.cache()
rawValidationData.cache()
rawTestData.cache()
numTrain = rawTrainData.count()
numVal = rawValidationData.count()
numTest = rawTestData.count()

print numTrain, numVal, numTest, numTrain + numVal + numTest # Print distribution and total
print rawData.take(1) # Print the features of first element 


# ##_Data Encoding and Model_

# I now define three functions to encode the features as categorial. They will be represented in MLlib's SparseVector format. The first will split each element of the RDD by the comma delimiter and return a list of tuples for each feature indexed by its key. The second will index all the parsed data and return all the categories in the entire dataset as a dictionary, where the key is the (index, feature) and the value will be its index realtive to the entire dataset. This will be a dictionary containing all of the distinct categories for the entire dateset. Finally, the third function will encode the categorial features in SparseVector format. All features will be represented as categorial in this case, as all the occurences of any particular feature will still be represented. Therefore, any continuous variables will be appropriately defined.
# 

# In[4]:

def parseLine(line):
    p = line.split(', ')
    return [(i-2, p[i]) for i in range(2,len(p)-1)]  

parsedTrainFeat = rawTrainData.map(parseLine)
print parsedTrainFeat.take(1)


# In[5]:

def toOneHotDict(inputData):
    return inputData.zipWithIndex().collectAsMap()

def oneHotEncoder(rawFeats, OHEDict, numOHEFeats):
    v = []
    for i in rawFeats:
      if i in OHEDict: # To take out unforseen features, not observed in the model
          v += [OHEDict[i]]
    return SparseVector(numOHEFeats, sorted(v), np.ones(len(v)))


# In[6]:

newsOHEDict = toOneHotDict(parsedTrainFeat.flatMap(lambda k: k).distinct()) # Unique tuples
numNewsOHEFeats = len(newsOHEDict.keys()) 
print newsOHEDict[(50, u'0.7')]
print numNewsOHEFeats


# The next function will use the functions above to transform the features and dependent variable to the appropriate format for training the model. This will be in MLlib's LabeledPoint format, where the features are represented as the SparseVector. The label will be defined as a binary variable, where 1 is defined as a popular article and 0 is not. In this case, the "shares" will be divided by the given "timedelta". 
# 
# After, scraping a few articles from the mashable website it was determined that the "age", provided in the html code, is the total days that have passed since the release of the article. This was concluded because articles with today's date provided an age of 0, while an article where 24 hours have passed, provided an age of 1. Therefore, the variable being predicted is the shares per day. This will provide a more accurate measure of the popularity, since as time passes there is more opportunity for shares. 

# In[7]:

def parseOHELine(line, OHEDict, numOHEFeats):
    p = line.split(', ')
    label = float(p[60])
    delt = float(p[1])
    val = label/delt
    if val > 50:
                l = 1 # Popular articles 
    else:
                l = 0 # Low popularity
    return LabeledPoint(l, oneHotEncoder(parseLine(line),OHEDict,numOHEFeats))

# Train Data
OHETrainData = rawTrainData.map(lambda line: parseOHELine(line, newsOHEDict, numNewsOHEFeats))
OHETrainData.cache()
print OHETrainData.take(1)


# In[8]:

# Validation Data
OHEValData = rawValidationData.map(lambda line: parseOHELine(line, newsOHEDict, numNewsOHEFeats))
OHEValData.cache()
print OHEValData.take(1)


# In[9]:

# Test Data
OHETestData = rawTestData.map(lambda line: parseOHELine(line, newsOHEDict, numNewsOHEFeats))
OHETestData.cache()
print OHETestData.take(1)


# In[10]:

trainPop = OHETrainData.filter(lambda lp: lp.label == 1).count()
testPop = OHETestData.filter(lambda lp: lp.label == 1).count()
valPop = OHEValData.filter(lambda lp: lp.label == 1).count()
trainTot = OHETrainData.count()
testTot = OHETestData.count()
valTot = OHEValData.count()
print (trainPop+testPop+valPop)/float(trainTot+testTot+valTot)


# The shares per day for the dataset ranged from values of .002 to 5266, with the majority of the data representing lower shares and being left skewed. Therefore, it was determined that a value of 50 shares/day seems to be good threshold for classification. This represents about 7% of the data. Hence, the model will be predicting whether an article can reach 50 shares per day.  

# In[11]:

from pyspark.mllib.classification import LogisticRegressionWithSGD 

# Fixed hyperparameters
numIters = 50
stepSize = 10.
regParam = 1e-6
regType = 'l2'
includeIntercept = True

model0 = LogisticRegressionWithSGD.train(OHETrainData, numIters, regParam=regParam, regType=regType, intercept=includeIntercept)
sortedWeights = sorted(model0.weights)
print sortedWeights[:5], model0.intercept # Examine five weights and intercept of model


# ##_Log Loss, Error, and ROC Curve_

# In[15]:

from math import log

def LogLoss(p, y):
    epsilon = 10e-12 # To keep range between 0 and 1
    if p == 0:
      p = epsilon
    elif p == 1:
      p = p - epsilon
    if y == 1:
      ll = -log(p)
    elif y == 0: 
      ll = -log(1-p)
    return ll


# Next, the mean of the training data labels is determined as a measure of the distribution of 1's and 0's in the data. This will be used to calculate the log loss between each label, and consequently a baseline log loss of the training data for comparing with the model.

# In[16]:

PopFractionTrain = OHETrainData.map(lambda lp: lp.label).mean()# Fraction of training with class one
print classOneFracTrain

logLossTrBase = OHETrainData.map(lambda lp: LogLoss(PopFractionTrain, lp.label)).mean()
print 'Baseline Train Logloss = {0:.3f}\n'.format(logLossTrBase)


# The following function will calculate a prediction based on the dot product of the features provided and the weights of the model, plus the intercept. This will initially be evaluated on the training data for computing the log loss between its prediction and labels. 

# In[17]:

from math import exp 

def getPred(x, w, intercept):
    rawPrediction = x.dot(w) + intercept 
    return float((1+exp(-rawPrediction))**(-1))

trainingPredictions = OHETrainData.map(lambda lp: getPred(lp.features, model0.weights, model0.intercept))
print trainingPredictions.take(5)


# The evaluate function will use a given model to make predictions on a given dataset and compute the log loss. Here, the function will evaluate the training data, which will be compared to the baseline log loss (computed with the mean label). 

# In[19]:

def evaluate(model, data):
    p = data.map(lambda lp: getPred(lp.features, model.weights, model.intercept)).collect()
    y = data.map(lambda lp: lp.label).collect()
    logLoss = []
    for i in range(len(p)):
      logLoss.append(LogLoss(p[i], y[i]))
    return np.mean(logLoss) # Mean of log loss between each prediction and label
  
logLossTrLR0 = evaluate(model0, OHETrainData)
print ('OHE Features Train Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossTrBase, logLossTrLR0))

# Examine highest five predictions of training set
probsAndLabelsTrain = OHETrainData.map(lambda lp: (getPred(lp.features, model0.weights, model0.intercept), lp.label))
print probsAndLabelsTrain.filter(lambda x: x[1] ==1).takeOrdered(5, key = lambda x: -x[0])


# In[22]:

logLossValBase = OHEValData.map(lambda lp: LogLoss(PopFractionTrain, lp.label)).mean()
logLossValLR0 = evaluate(model0, OHEValData)
print ('OHE Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossValBase, logLossValLR0))

# Examine highest five predictions of validation set
probsAndLabels = OHEValData.map(lambda lp: (getPred(lp.features, model0.weights, model0.intercept), lp.label))
print probsAndLabels.filter(lambda x: x[1] ==1).takeOrdered(5, key = lambda x: -x[0])


# In[24]:

import matplotlib.pyplot as plt


def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0):
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

labelsAndScores = OHEValData.map(lambda lp:
                                            (lp.label, getPred(lp.features, model0.weights, model0.intercept)))
labelsAndWeights = labelsAndScores.collect()
labelsAndWeights.sort(key=lambda (k, v): v, reverse=True)
labelsByWeight = np.array([k for (k, v) in labelsAndWeights])

length = labelsByWeight.size
truePositives = labelsByWeight.cumsum()
numPositive = truePositives[-1]
falsePositives = np.arange(1.0, length + 1, 1.) - truePositives

truePositiveRate = truePositives / numPositive
falsePositiveRate = falsePositives / (length - numPositive)

# Generate layout and plot data
fig, ax = preparePlot(np.arange(0., 1.1, 0.1), np.arange(0., 1.1, 0.1))
ax.set_xlim(-.05, 1.05), ax.set_ylim(-.05, 1.05)
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
plt.plot(falsePositiveRate, truePositiveRate, color='#8cbfd0', linestyle='-', linewidth=3.)
plt.plot((0., 1.), (0., 1.), linestyle='--', color='#d6ebf2', linewidth=2.)  # Baseline model 
pass


# ##_SVM Model_

# In[25]:

from pyspark.mllib.classification import SVMWithSGD 

# fixed hyperparameters
numIters = 50
stepSize = 10.
regParam = 1e-6
regType = 'l2'
includeIntercept = True

model1 = SVMWithSGD.train(OHETrainData, numIters, regParam=regParam, regType=regType, intercept=includeIntercept)
sortedWeights = sorted(model1.weights)
print sortedWeights[:5], model1.intercept # Examine last five weights and intercept of model


# In[26]:

trainingPredictions1 = OHETrainData.map(lambda lp: getPred(lp.features, model1.weights, model1.intercept))

print trainingPredictions1.take(5)


# In[27]:

logLossTrSVM1 = evaluate(model1, OHETrainData)
print ('OHE Features Train Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossTrBase, logLossTrSVM1))


# In[28]:

logLossValSVM1 = evaluate(model1, OHEValData)
print ('OHE Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossValBase, logLossValSVM1))


probsAndLabels1 = OHEValData.map(lambda lp: (getPred(lp.features, model1.weights, model1.intercept), lp.label))
print probsAndLabels1.filter(lambda x: x[1] ==1).takeOrdered(5, key = lambda x: -x[0])


# In[29]:

labelsAndScores1 = OHEValData.map(lambda lp:
                                            (lp.label, getPred(lp.features, model1.weights, model1.intercept)))
labelsAndWeights = labelsAndScores1.collect()
labelsAndWeights.sort(key=lambda (k, v): v, reverse=True)
labelsByWeight = np.array([k for (k, v) in labelsAndWeights])

length = labelsByWeight.size
truePositives = labelsByWeight.cumsum()
numPositive = truePositives[-1]
falsePositives = np.arange(1.0, length + 1, 1.) - truePositives

truePositiveRate = truePositives / numPositive
falsePositiveRate = falsePositives / (length - numPositive)

# Generate layout and plot data
fig, ax = preparePlot(np.arange(0., 1.1, 0.1), np.arange(0., 1.1, 0.1))
ax.set_xlim(-.05, 1.05), ax.set_ylim(-.05, 1.05)
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
plt.plot(falsePositiveRate, truePositiveRate, color='#8cbfd0', linestyle='-', linewidth=3.)
plt.plot((0., 1.), (0., 1.), linestyle='--', color='#d6ebf2', linewidth=2.)  # Baseline model 
pass


# In[22]:

labelsAndPreds = OHEValData.map(lambda lp: (model0.predict(lp.features), lp.label))
print labelsAndPreds.take(3)


# In[94]:

n1 = labelsAndPreds.filter(lambda x: x[0]==1 & x[1]==0).count()
n2 = labelsAndPreds.filter(lambda x: x[0]==0 & x[1]==1).count()
tot = labelsAndPreds.count()
print (n1+n2)/tot
labelPredsAndFeats = OHEValData.map(lambda lp: (lp.label, model0.predict(lp.features), lp.features))
goodFeatsVal = OHETrainData.filter(lambda lp: lp.label == 1)
print goodFeatsVal.count()
print goodFeatsVal.take(3)


# In[76]:

labelsAndPreds = OHEValData.map(lambda lp: (lp.label, model0.predict(lp.features)))
labelPredsAndFeats = OHEValData.map(lambda lp: (lp.label, model0.predict(lp.features), lp.features))
goodFeats = OHETrainData.filter(lambda lp: lp.label == 0)
print goodFeats.count()
rmseValLR1 = calcRMSE(labelsAndPreds)
#print('Root mean squared error, Model 1 = ' + str(rmseValLR1))
#print rmseValLR1
#print labelsAndPreds.take(50)


# In[95]:

numPoints = rawData.count()
print numPoints
samplePoints = rawData.take(5)
print samplePoints

