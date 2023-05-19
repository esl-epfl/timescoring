

# Library for measuring performance based event and duration level of predictions 

## Motivation

For temporal and sequential data (e.g. in biomedical applications), standard performance evaluation metrics, such as sensitivity and specificity, may not always be the most appropriate and can even be misleading. Evaluation metrics must ultimately reflect the needs of users and also be sufficiently sensitive to guide algorithm development.

For example, for epilepsy monitoring, neurologists ask for assesments on the level of seizure episodes (events), rather than duration or sample-by-sample based metrics. Similarly,  another performance measure with a strong practical impact in epilepsy monitoring, is the false alarm rate (FAR), or the number of false positives per hour/day. Clinicians and patients see this measure as more meaningful than many more commonly used metrics, and are very demanding in terms of performance, requiring it to be as low as possible for potential wearable applications (e.g., less than 1 FP/day). This also necessitates exceptionally high constraints on the required precision (usually much higher than 99\%). 

For this reason, here we provide code that measures performance on the level of events and duration, as well as FAR. 


## Metrics

In more details, we measures performance on the level of: 

- events (e.g. epilepsy episodes), not caring about the exact length overlap between true and predicted event, it classifies a match if there is any overlap between predicted and true event.
- duration (or sample-by-sample), classical performance metric that cares about each sample classification
- combination of both (mean and geometric mean of F1 scores of event and duration based metrics)
- number of false positives per day (useful for biomedical applications such as epilepsy monitoring)

In picture below are illustrate several use cases, how errors are counted, and what is the final performance measure.

![Illustration of duration and episode-based performance metrics.](PerformanceMetricsIllustration.png)

## Code 

All code for performance metric is in *PerformanceMetricsLib.py* whereas in *script_metricsDemo.py* is demo how to use it.  
*PerformanceMetricsLib.py* is easily imported with: 
```
from PerformanceMetricsLib import *
```

Class *EventsAndDurationPerformances* is defined that contains several parameters and functions. 

### Parameters

Parameters that need to be defined and passed to *EventsAndDurationPerformances* class are: 
- samplFreq - sampling frequency of labels and predictions, needed to calculate FP per day
- toleranceFP_befEvent  - how much time [s] before event it is still ok to predict event without classifying it as false positive (FP)
- toleranceFP_aftEvent -  how much time [s] after event it is still ok to predict event without classifying it as false positive (FP)
- percOverlapNeeded - minimal overlap true event has to be overlapped with predicted event to be finally counted as matched [percentage of length if true event]
- maxLenFP - max tolerable length [s] of false positive before it has to be split and counted as more FP
- eventStableLenToTest - window length [s] in which it postprocesses and smooths labels
- eventStablePercToTest - what percentage of labels needs to be 1 in eventStableLenToTest window to say that it is 1
- distanceBetween2events - if events are closer then distanceBetween2events [s] then it merges them to one (puts all labels inbetween to 1 too)
- bayesProbThresh - threshold for Bayes postprocessing of predicted labels (based on their confidences)

### Functions

Functions defined in *EventsAndDurationPerformances* class are: 

> calculateStartsAndStops(self, labels)
- Function that detects starts and stop of event (or groups of labels 1) and returns them in form [start, stop].

> calc_TPAndFP
- For a pair of ref and hyp event decides if it is false or true prediction. 
-   It returns: 
    - tp - percentage of overlap if true event
    - fp - if this predicted event contributes to FP before (-1), after (1), or on both sides (2) of true event
    - fp_bef and fp_aft - lengths of FP (if exists) before and after true event (as multiplier of maxLenFP)   

> performance_events
- Function that detects events in a stream of true labels and predictions.
- Detects overlaps and returns [sensitivity, precision , F1 score and number of false positives/day].  

> performance_duration
- Calculates performance metrics on the  sample by sample basis.
- Returns [sensitivity, precision and F1 score].

> performance_all9
- Function that returns 9 different performance measures of prediction on epilepsy
	- on the level of events (sensitivity, precision and F1 score)
	- on the level of event duration, or each sample (sens, prec, F1 score)
	- combination of F1 scores for events and duration ( mean or gmean)
	- number of false positives per day
- Returns them in this order:  ['Sensitivity events', 'Precision events', 'F1score events', 'Sensitivity duration', 'Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
	
> smoothenLabels_movingAverage
- Returns labels after two steps of postprocessing: 
	- moving window with voting (if more then threshold of labels are 1 final label is 1 otherwise 0)
	- merging events that are too close
	
> smoothenLabels_Bayes
- Returns labels bayes postprocessing. Calculates cummulative probability of event and non event over the window of size eventStableLenToTestIndx. if log (cong_pos /cong_ned )> bayesProbThresh then label is 1. 

> calculatePerformanceAfterVariousSmoothing
- Function that calculates performance for epilepsy. 
- It evaluates on raw predictions but also performs different postprocessing and evaluated performance after postprocessing:
	- first smoothing is just moving average with specific window size and percentage of labels that have to be 1 to give final label 1
	- then merging of too close event is performed in step2
	- another option for postprocessing and smoothing of labels is bayes postprocessing
- Returns dictionary with 9 values for each postprocessing option:['Sensitivity events', 'Precision events', 'F1score events', 'Sensitivity duration','Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
- Also returns postprocessed (smoothed) labels. 

> plotInterp_PredAndConf
- Function that plots in time true labels, raw predictions as well as postprocessed predictions.

## Example of usage

```
# defining various parameters for postprocessing and performance measuring
class PerfParams:
    #parameters for scoring performance
    samplFreq = 1 #in Hz, e.g. 1 sample=1sec
    toleranceFP_befEvent = 1 #in sec
    toleranceFP_aftEvent = 2 #in sec
    percOverlapNeeded= 0 #in percentage of true event
    maxLenFP= 1.5*1000000# in sec
    #parameters for postprocessing
    movingWinLen = 5 #in sec
    movingWinPercentage = 0.5
    distanceBetween2events = 3 #in sec
    bayesProbThresh =1.5 #bayes probability threshold (probably has to be tuned)
perfMetrics = EventsAndDurationPerformances(PerfParams)

# example of labels and predicsitons
# with maxLenTP= inf
# with percOverlapNeeded=0 -  totalTP=2, totalFP=4, totalFN=1
# with percOverlapNeeded=0.5 -  totalTP=1, totalFP=4, totalFN=2
trueLabels=np.array([0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
predictions=np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0])
predProbab=np.ones(len(predictions))

# MEASURING PERFORMANCE
# all 9 performance metrics - on raw predictions, without postprocessing
performancesNoSmooth= perfMetrics.performance_all9(predictions, trueLabels)

# POSTPROCESSING 
# performance after 2 types of postprocessing (moving average and bayes smoothing)
(performanceMetrics, smoothedPredictions) = perfMetrics.calculatePerformanceAfterVariousSmoothing(predictions, trueLabels,predProbab)

# visualizing postprocessed labels 
allPredictSmoothing=np.vstack((trueLabels, predictions, smoothedPredictions['MovAvrg'], smoothedPredictions['MovAvrg&Merge'], smoothedPredictions['Bayes'], smoothedPredictions['Bayes&Merge']))
perfMetrics.plotInterp_PredAndConf(trueLabels,predictions, predProbab, smoothedPredictions, 'PredictionVisualization')
```