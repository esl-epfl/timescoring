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
    # TODO: add parmeter for minimal percentage of overlap to count as a match, and max len of FP before split on two

perfMetrics = EventsAndDurationPerformances(PerfParams)

############################################################################################
############################################################################################
## DIFFERENT TEST EXAMPLES
# TODO: add expected outcome in terms of performance metrics for each
# seizures with typical distribution, some overlaping some not - example 1
trueLabels=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0, 0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
predictions=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1])
predProbab=np.ones(len(predictions))

# # seizures with typical distribution, some overlaping some not - example 2
# trueLabels=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0, 0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# predictions=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1])
# predProbab=np.ones(len(predictions))
#
# # long predicted seizures spaning several true seizures
# trueLabels=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0, 0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# predictions=np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# predProbab=np.ones(len(predictions))
#
# # long true seizure and many shorted predicted seizures
# trueLabels=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0, 0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# predictions=np.array([0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0])
# predProbab=np.ones(len(predictions))
#
# # combination of everything
# trueLabels=np.array([0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# predictions=np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0])
# predProbab=np.ones(len(predictions))
#
# # no true labels
# trueLabels=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# predictions=np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0])
# predProbab=np.ones(len(predictions))
#
# #simple example with different probabilities of labels
# trueLabels=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0])
# predictions=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0])
# predProbab=np.array([0.9,0.8,0.7,0.8,0.7,0.8,0.9,0.6,0.5,0.6,0.7,0.8,0.9,0.9,0.9,0.6,0.7,0.8,0.8,0.8,0.9,0.8,0.7,0.8,0.7,0.8,0.9,0.6,0.5,0.6,0.7,0.8,0.9,0.9,0.9,0.6,0.7,0.8,0.8,0.8])

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
# TODO: Think of and implement visualization of matching - probably in event matching function and have parameter if ploting is on or off

# performance after 2 types of postprocessing (moving average and bayes smoothing)
(performanceMetrics, smoothedPredictions) = perfMetrics.calculatePerformanceAfterVariousSmoothing(predictions, trueLabels,predProbab)
print(performanceMetrics)

############################################################################################
## VISUALIZATION OF POSTPROCESSED LABELS
allPredictSmoothing=np.vstack((trueLabels, predictions, smoothedPredictions['MovAvrg'], smoothedPredictions['MovAvrg&Merge'], smoothedPredictions['Bayes'], smoothedPredictions['Bayes&Merge']))
print(allPredictSmoothing)
perfMetrics.plotInterp_PredAndConf(trueLabels,predictions, predProbab, smoothedPredictions, 'PredictionVisualization')