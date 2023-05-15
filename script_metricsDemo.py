''' Script to show how to use PerformanceMetricsLib '''

__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

from PerformanceMetricsLib import *


## SETUP
# defining various parameters for postprocessing and performance measuring
class PerfParams:
    samplFreq = 1 #in Hz, e.g. 1 sample=1sec
    toleranceFP_befEvent = 1 #in sec
    toleranceFP_aftEvent = 2 #in sec
    movingWinLen = 5 #in sec
    movingWinPercentage = 0.5
    distanceBetween2events = 3 #in sec
    bayesProbThresh =1.5 #bayes probability threshold (probably has to be tuned)

perfMetrics = EventsAndDurationPerformances(PerfParams)

############################################################################################
############################################################################################
## DIFFERENT TEST EXAMPLES

trueLabels=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0])
predictions=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0])
predProbab=np.array([0.9,0.8,0.7,0.8,0.7,0.8,0.9,0.6,0.5,0.6,0.7,0.8,0.9,0.9,0.9,0.6,0.7,0.8,0.8,0.8])

# trueLabels=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0])
# predictions=np.array([0,0,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0])
# predProbab=np.array([0.9,0.8,0.7,0.8,0.7,0.8,0.9,0.6,0.5,0.6,0.7,0.8,0.9,0.9,0.9,0.6,0.7,0.8,0.8,0.8])

# trueLabels=np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0])
# predictions=np.array([0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0])
# predProbab=np.array([0.9,0.8,0.7,0.8,0.7,0.8,0.9,0.8,0.7,0.8,0.6,0.5,0.6,0.7,0.8,0.9,0.9,0.9,0.6,0.7,0.8,0.8,0.8])


# #testing detecting seizure at the end
# trueLabels=np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1])
# predictions=np.array([0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1])
# predProbab=np.array([0.9,0.8,0.7,0.8,0.7,0.8,0.9,0.8,0.7,0.8,0.6,0.5,0.6,0.7,0.8,0.9,0.9])

trueLabels=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0])
predictions=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0])
predProbab=np.array([0.9,0.8,0.7,0.8,0.7,0.8,0.9,0.6,0.5,0.6,0.7,0.8,0.9,0.9,0.9,0.6,0.7,0.8,0.8,0.8,0.9,0.8,0.7,0.8,0.7,0.8,0.9,0.6,0.5,0.6,0.7,0.8,0.9,0.9,0.9,0.6,0.7,0.8,0.8,0.8])

############################################################################################
############################################################################################
## MEASURING PERFORMANCE

#all 9 performance metrics - on raw predictions, without smoothing
# firs three are on the level of events (sensitivity, precision, F1score)
# next three are on the level of duration (sensitivity, precision, F1score)
# then mean or F1E and F1D, geoMean of F1E and F1DE
# last is number of false positives that would be per day (linear interpolation from numFP in current sample)
performancesNoSmooth= perfMetrics.performance_all9(predictions, trueLabels)
print(performancesNoSmooth)

# performance after 2 types of postprocessing (moving average and bayes smoothing)
(performanceMetrics, smoothedPredictions) = perfMetrics.calculatePerformanceAfterVariousSmoothing(predictions, trueLabels,predProbab)
print(performanceMetrics)


############################################################################################
## VISUALIZATION OF POSTPROCESSED LABELS
allPredictSmoothing=np.vstack((trueLabels, predictions, smoothedPredictions['MovAvrg'], smoothedPredictions['MovAvrg&Merge'], smoothedPredictions['Bayes'], smoothedPredictions['Bayes&Merge']))
print(allPredictSmoothing)
perfMetrics.plotInterp_PredAndConf(trueLabels,predictions, predProbab, smoothedPredictions, 'PredictionVisualization')