import pandas
import numpy
import random
import math
from statsmodels import api as sm

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
        self.feature = None
        self.constraint = None
        self.featuresLeftToTry = None
        self.gini = None

def createTestForLogit():
	# should be no errors here, perfect regression set up
	inSample = pandas.DataFrame({'G' : pandas.Series(range(100)) , 'A' : pandas.Series([0]*50+[2]*50), 'NUMFM' : pandas.Series(numpy.random.randint(0, 10, 100)), 'LVD' : pandas.Series([0]*50+[1]*50)})
	outOfSample = pandas.DataFrame({'G' : pandas.Series(range(25,75)) , 'A' : pandas.Series([0]*25+[2]*25), 'NUMFM' : pandas.Series(numpy.random.randint(0, 10, 50)), 'LVD' : pandas.Series([0]*25+[1]*25)})
	return inSample, outOfSample

def createTestForGini():
	# in this example, A should be split off first, then NUMFM, and we should be 100% correct for these four out of sample observations
	hundredZeroes = [0]*100
	fourZeroes = [0]*4
	inSample = pandas.DataFrame({'G' : pandas.Series(hundredZeroes) , 'A' : pandas.Series([0]*50+[1]*50), 'NUMFM' : pandas.Series([0]*41+[1]*9+[0]*41+[1]*9), 'LVD' : pandas.Series([0]*40+[1]*10+[1]*40+[0]*10)})
	outOfSample = pandas.DataFrame({'G' : pandas.Series(fourZeroes) , 'A' : pandas.Series([0,0,1,1]), 'NUMFM' : pandas.Series([0,1,0,1]), 'LVD' : pandas.Series([0,1,1,0])})
	return inSample, outOfSample

def createTestDataForNearestNeighbors():
	# nearest neighbors should be 100% correct for these three out of sample observations. k must be three for this to work. 
	sixZeroes = [0,0,0,0,0,0]
	threeZeroes = [0,0,0]
	inSample = pandas.DataFrame({'G' : pandas.Series(sixZeroes) , 'A' : pandas.Series([0,0,1,1,.5,.5]), 'NUMFM' : pandas.Series([0,1,0,1,.5,.5]), 'LVD' : pandas.Series([0,0,1,1,0,1])})
	outOfSample = pandas.DataFrame({'G' : pandas.Series(threeZeroes) , 'A' : pandas.Series([0.25,0.75,0.25]), 'NUMFM' : pandas.Series([0.25,0.25,0.75]), 'LVD' : pandas.Series([0,1,0])})
	return inSample, outOfSample
      
# This function creates random data to analyze and model      
def createData():
	numSamples = 20
	gData = numpy.random.randint(0, 2, numSamples)
	aData = numpy.random.randint(0, 100, numSamples)
	numfmData = numpy.random.randint(0, 6, numSamples)
	lvdData = numpy.random.randint(0, 2, numSamples)
	preFrameData = {'G' : pandas.Series(gData) , 'A' : pandas.Series(aData), 'NUMFM' : pandas.Series(numfmData), 'LVD' : pandas.Series(lvdData)}
	framedData = pandas.DataFrame(preFrameData)
	return framedData
	
def createStaticData():
	gData = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0,1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1,0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
	aData = [66, 83, 46, 21,  4, 66, 36, 49, 78, 59, 47, 56, 28,  6, 42, 69, 76,85, 58,  7, 76, 46, 65, 16, 79, 79,  7, 96, 39, 60, 73, 25, 53, 52,32, 35, 18, 53, 22, 54,  4, 46, 89, 41, 39, 80, 78, 28, 85, 86, 13,67, 20,  3,  4, 86, 95, 25, 87, 28]
	numfmData = 3, 5, 0, 1, 5, 5, 4, 5, 4, 1, 4, 3, 4, 4, 2, 2, 1, 1, 1, 3, 4, 1, 2,4, 1, 0, 0, 1, 2, 5, 3, 1, 5, 1, 2, 2, 4, 1, 2, 2, 4, 1, 5, 0, 0, 1,5, 3, 2, 1, 5, 0, 3, 1, 5, 2, 5, 0, 5, 5
	lvdData = [1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1,1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1,1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0]
	preFrameData = {'G' : pandas.Series(gData) , 'A' : pandas.Series(aData), 'NUMFM' : pandas.Series(numfmData), 'LVD' : pandas.Series(lvdData)}
	inSampleData = pandas.DataFrame(preFrameData)
	gDataOOS = [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1,
       1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0,
       0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0,
       0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1,
       1, 1, 0, 0, 1, 1, 1, 0]
	aDataOOS = [31,  3, 81, 87, 64, 51, 45,  5, 98, 25,  8, 11, 97, 10, 20, 56, 40,
       18, 46, 35, 32, 53, 99, 68, 97, 37, 82,  6, 63, 76, 29, 11, 95, 59,
       57, 89, 56, 30, 10, 85, 72, 45, 81, 15, 62, 91, 71, 51, 56, 52, 70,
       17, 58, 29, 11, 83, 22, 42, 78,  4, 18, 28, 62, 64, 30, 22, 36, 31,
       56, 20, 69, 42,  5, 90, 94, 49, 85, 48, 68, 69, 44, 42, 76, 15, 61,
       96, 20, 43, 11, 92,  0, 71, 81, 99,  3, 81, 31, 93, 41, 79]
	numfmDataOOS = [4, 2, 4, 3, 2, 0, 3, 2, 2, 2, 2, 5, 2, 4, 2, 0, 5, 3, 4, 1, 0, 0, 1,3, 1, 1, 3, 5, 3, 1, 5, 4, 0, 4, 2, 0, 0, 3, 1, 4, 3, 1, 4, 4, 0, 2,1, 2, 1, 1, 2, 4, 5, 3, 2, 4, 1, 5, 1, 4, 4, 4, 2, 0, 1, 3, 1, 1, 5,5, 0, 5, 0, 2, 2, 5, 1, 4, 5, 4, 2, 1, 4, 0, 0, 2, 4, 1, 2, 3, 4, 0,3, 2, 1, 5, 5, 0, 2, 2]
	lvdDataOOS = [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1,
       0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
       0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1,
       1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1,
       1, 1, 1, 0, 0, 1, 0, 1]
	outOfSampleData = pandas.DataFrame({'G' : pandas.Series(gDataOOS) , 'A' : pandas.Series(aDataOOS), 'NUMFM' : pandas.Series(numfmDataOOS), 'LVD' : pandas.Series(lvdDataOOS)})
	return inSampleData, outOfSampleData	

# This function returns an in sample data set and an out of sample data set from an original, whole data set
# The parameter percentInSample tells us how much of the original data set to pour into the new in sample data set
def getInAndOutSample(data, percentInSample):
	inSampleLength = int(math.floor(percentInSample*len(data)))
	inSampleIndices = random.sample(xrange(len(data)), inSampleLength)
	inSampleData = data.ix[inSampleIndices]
	outOfSampleData = data.drop(inSampleIndices)
	return inSampleData, outOfSampleData

# This function calculates the euclidean distance between two observations
def getEuclideanDistance(features, currentOutOfSampleDataPoint, currentInSampleDataPoint):
	sumOfSquares = 0
	for feature in features:
		sumOfSquares += (currentOutOfSampleDataPoint[feature] - currentInSampleDataPoint[feature]) ** 2
	return math.sqrt(sumOfSquares)

# This function runs a nearest neighbors model and returns an error rate. We loop through all of the out of sample observations,
# and for each one, calculate the euclidean distance to each in sample observation. We then take the k closest observations and average their
# target variable to reach out prediction. 
def runNearestNeighbors(inSampleData, outOfSampleData, k):
	numIncorrectPredictions = 0.0
	features = inSampleData.columns.drop('LVD')
	for outOfSampleIndex in outOfSampleData.index: # loop through out of sample observations
		euclideanDistancesAndData = []
		currentOutOfSampleDataPoint = outOfSampleData.ix[outOfSampleIndex]
		for inSampleIndex in inSampleData.index: # loop through in sample observations
			currentInSampleDataPoint = inSampleData.ix[inSampleIndex]
			euclideanDistance = getEuclideanDistance(features, currentOutOfSampleDataPoint, currentInSampleDataPoint) # calculate the distance between the current in and out of sample observations
			euclideanDistancesAndData.append({'EuclideanDistance' : euclideanDistance , 'InSampleDataPoint' : currentInSampleDataPoint}) # append the above distance calculation
		euclideanDistancesAndData = sorted(euclideanDistancesAndData, key=lambda k : k['EuclideanDistance']) # sort the dictionary of in sample observations and their distances to the current out of sample observation
		targetVars = [ distanceAndDataPoint['InSampleDataPoint']['LVD'] for distanceAndDataPoint in euclideanDistancesAndData[:k]] # find the closest k in sample observations
		averageTarget = numpy.mean(targetVars)
		# If the average target is greater than 0.5, then we predict a value of 1 to be most likely, which corresponds to alive. Otherwise we predict a value of 0. 
		prediction = 1 if averageTarget > .5 else 0
		if prediction != currentOutOfSampleDataPoint['LVD']: # check if our prediction is correct and increment numIncorrectPredictions if it is not
			numIncorrectPredictions += 1
	return numIncorrectPredictions / len(outOfSampleData)
	
def runNearestNeighborsKernelSmoothing(inSampleData, outOfSampleData):
	# TO DO - still need to write this function
	return 1
		
def runLogisticRegression(inSampleData, outOfSampleData):
	features = inSampleData.columns.drop('LVD')
	logit = sm.Logit(inSampleData['LVD'], inSampleData[features]) 
	result = logit.fit()
	outOfSamplePredictions = result.predict(outOfSampleData[features])
	transformedOutOfSamplePredictions = [ 1 if prediction > 0.5 else 0 for prediction in outOfSamplePredictions]
	outOfSampleData = outOfSampleData.reset_index()
	incorrectPredictionTracker = [ 1 if transformedOutOfSamplePredictions[i] != outOfSampleData['LVD'][i] else 0 for i in range(len(transformedOutOfSamplePredictions))] 
	return sum(incorrectPredictionTracker)/float(len(incorrectPredictionTracker))

# This function calculations the gini impurity for the current node
def getNodeGini(decisionTree):
	percentAlive = sum(decisionTree.data['LVD']) / float(len(decisionTree.data['LVD']))
	percentDead = 1 - percentAlive
	return 1 - (percentAlive ** 2) - (percentDead ** 2)

# This function calculates the gini impurity for the next hypothetical level of the decision tree for a given feature and constraint. 
# If we find the new gini to be low enough as compared to the gini impurity of the current node, then we may be able to reassign the lowestGini variable. 
def getNextStepGini(lowestGini, decisionTree, feature, giniImprovementRequirement, contraint):
	leftData = decisionTree.data[decisionTree.data[feature] < contraint]
	rightData = decisionTree.data[decisionTree.data[feature] > contraint]
	if len(leftData)!= 0 and len(rightData) != 0: # here we check that both branches would have at least one observation
		leftRatioAlive = sum(leftData['LVD']) / float(len(leftData['LVD'])) # the percent of alive passengers in the left branch
		rightRatioAlive = sum(rightData['LVD']) / float(len(rightData['LVD'])) # the percent of alive passengers in the right branch
		ratioOfSamplesToLeft = len(leftData) / float(len(decisionTree.data)) # the percent of observations in the left branch
		ratioOfSampleToRight = 1 - ratioOfSamplesToLeft
		leftGini = 1 - (leftRatioAlive ** 2) - ((1-leftRatioAlive) ** 2)
		rightGini = 1 - (rightRatioAlive ** 2) - ((1-rightRatioAlive) ** 2)
		averageGini = leftGini * ratioOfSamplesToLeft + rightGini * ratioOfSampleToRight # this is the calculation of the gini impurity for the next hypothetical level
		if decisionTree.gini - averageGini > giniImprovementRequirement and averageGini < lowestGini['Gini']:
			lowestGini = { 'Gini' : averageGini, 'LeftGini': leftGini, 'RightGini' : rightGini,  'Feature' : feature, 'Constraint' : contraint}
	return lowestGini	

# This function tries to find a lower gini impurity by cycling through a few different constraint values for the given feature
def tryNewFeature(lowestGini, decisionTree, feature, giniImprovementRequirement):
	numSteps = 10 # the number of constraints to cycle through
	minValue = decisionTree.data[feature].min()
	maxValue = decisionTree.data[feature].max()
	constraintsToTry = [minValue + (1.0 / numSteps)*i*(maxValue - minValue) for i in range(numSteps)] # calculate constraints to cycle through
	for contraint in constraintsToTry:
		lowestGini = getNextStepGini(lowestGini, decisionTree, feature, giniImprovementRequirement, contraint)	
	return lowestGini

# This recursive function builds out our decision tree from the in sample data implanted in a tree root. This function just alters the variable decisionTree, it does not
# return anything. 
def buildOutDecisionTree(decisionTree, giniImprovementRequirement):
	thisNodesGiniImpurity = getNodeGini(decisionTree) # Here we calculate the gini impurity of the current node
	decisionTree.gini = thisNodesGiniImpurity
	lowestGini = { 'Gini' : float('inf'), 'LeftGini' : None, 'RightGini' : None, 'Feature' : None, 'Constraint' : None} # This dictionary tracks the best gini impurity we have found so far
	for feature in decisionTree.featuresLeftToTry: # filter through all of the features we have not yet used and search each one for a way to beat the current lowest gini impurity
		lowestGini = tryNewFeature(lowestGini, decisionTree, feature, giniImprovementRequirement)
	if lowestGini['Gini'] < float('inf'): # if we found at least one gini impurity smaller enough than the gini impurity of the current node, then create left and right branches
		decisionTree.feature = lowestGini['Feature']
		decisionTree.constraint = lowestGini['Constraint']
		decisionTree.left = Tree()
		decisionTree.right = Tree()
		decisionTree.left.featuresLeftToTry = decisionTree.right.featuresLeftToTry = decisionTree.featuresLeftToTry.drop(lowestGini['Feature'])
		decisionTree.left.data = decisionTree.data[decisionTree.data[lowestGini['Feature']] < lowestGini['Constraint']]
		decisionTree.right.data = decisionTree.data[decisionTree.data[lowestGini['Feature']] > lowestGini['Constraint']]	
		decisionTree.left.gini = lowestGini['LeftGini']
		decisionTree.right.gini =  lowestGini['RightGini']		
		buildOutDecisionTree(decisionTree.left, giniImprovementRequirement) # recursively build out the left and right branches
		buildOutDecisionTree(decisionTree.right, giniImprovementRequirement)


# This functions predicts the target variable for a single observation and decision tree and returns if our prediction was correct or not
def predictTargetForObservation(decisionTree, outOfSampleObservation):
	if decisionTree.left == None or decisionTree.right == None: # then we are at a leaf and need to check our prediction now
		predictedTarget = round(numpy.mean(decisionTree.data['LVD']))
	elif outOfSampleObservation[decisionTree.feature] > decisionTree.constraint: # in this case we need to recurse through the right branch of the current node
		return predictTargetForObservation(decisionTree.right, outOfSampleObservation)
	elif outOfSampleObservation[decisionTree.feature] <= decisionTree.constraint: # in this case we need to recurse through the left branch of the current node
		return predictTargetForObservation(decisionTree.left, outOfSampleObservation)
	else: # We should never get here so I wanted to print an error in case we do
		print 'We ran into an unexpected edge case!'
		return 1/0
	observationIncorrectlyPredicted = 	int(predictedTarget != outOfSampleObservation['LVD'])
	return observationIncorrectlyPredicted

# This function runs the out of sample data through the decision tree and returns our error rate for predicting the target variable
def evaluateOutOfSampleDataThroughDecisionTree(decisionTree, outOfSampleData):
	numIncorrectPredictions = 0
	outOfSampleData = outOfSampleData.reset_index()
	for rowIndex in range(len(outOfSampleData)):	# here we loop through all observations in the out of sample data
		observationIncorrectlyPredicted = predictTargetForObservation(decisionTree, outOfSampleData.ix[rowIndex])
		numIncorrectPredictions += observationIncorrectlyPredicted # increment numIncorrectPredictions everytime we make another incorrect prediction
	return float(numIncorrectPredictions) / len(outOfSampleData)
		
# This is the main function to get an error rate for the decision tree learning model. This model uses the gini impurity metric
# and uses the simple stopping criteria of comparing the gini gain to the parameter giniImprovementRequirement and making sure it is larger. 	
def runDecisionTreeGiniImpurity(inSampleData, outOfSampleData, giniImprovementRequirement):
	decisionTree = Tree() # Here we create the root of our decision tree and start filling in some basic data fields
	decisionTree.data = inSampleData
	decisionTree.featuresLeftToTry = inSampleData.columns.drop('LVD')
	buildOutDecisionTree(decisionTree, giniImprovementRequirement) # Here we start the recursion process and build out our decision tree based on the in sample data
	errorRate = evaluateOutOfSampleDataThroughDecisionTree(decisionTree, outOfSampleData) # Here we run our out of sample data through the decision tree and calculate the error rate
	return errorRate
	
# This function returns the error rate for a single cross validation run. This function
# feeds into other functions nearly immediately depending on which model is specified by the user. 
def getErrorRate(modelType, inSampleData, outOfSampleData):
	if modelType == 'NearestNeighbors':
		k = 3
		return runNearestNeighbors(inSampleData, outOfSampleData, k)
	elif modelType == 'NearestNeighborsKernelSmoothing':
		return runNearestNeighborsKernelSmoothing(inSampleData, outOfSampleData) # TO DO - still need to write this function
	elif modelType == 'LogisticRegression':
		return runLogisticRegression(inSampleData, outOfSampleData)
	elif modelType == 'DecisionTreeGiniImpurity':
		giniImprovementRequirement = .1
		return runDecisionTreeGiniImpurity(inSampleData, outOfSampleData, giniImprovementRequirement)
	else:
		print 'Did not input a valid model type! \n'
		return 1/0

def main():
	# This is our main function
	percentInSample = 0.75 # The percent of data used as in sample data for each run
	numberOfCrossValidations = 10 # The number of randomized cross validations that we will run
	modelType = 'LogisticRegression' # Choiced of model type are DecisionTreeGiniImpurity, NearestNeighborsKernelSmoothing, NearestNeighbors, and LogisticRegression
	errorRates = []
	
	#myData = createData() # Creates fake data to use before we get real data
	#for crossValidation in xrange(numberOfCrossValidations):
	#	inSampleData, outOfSampleData = getInAndOutSample(myData, percentInSample) # Pick our in and out of sample data sets from the overall data set
	#	errorRate = getErrorRate(modelType, inSampleData, outOfSampleData) 
	#	errorRates.append(errorRate)
	
	inSampleData, outOfSampleData = createStaticData()
	errorRate = getErrorRate(modelType, inSampleData, outOfSampleData) 
	errorRates.append(errorRate)
	
	### NOTES: Error rate is 0.55 for LogisticRegression, 0.57 for NearestNeighbors, and 0.47 for DecisionTreeGiniImpurity
	
	averageErrorRate = numpy.mean(errorRates)
	print 'The average error rate is: ', averageErrorRate
	#return myData
