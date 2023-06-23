# Library for measuring performance of time series classification

![PyPI](https://img.shields.io/pypi/v/timescoring?style=flat-square)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/esl-epfl/epilepsy_performance_metrics/python-app.yml?label=unittest&style=flat-square)

## Motivation

For temporal and sequential data (e.g. in biomedical applications), standard performance evaluation metrics, such as sensitivity and specificity, may not always be the most appropriate and can even be misleading. Evaluation metrics must ultimately reflect the needs of users and also be sufficiently sensitive to guide algorithm development.

For example, for epilepsy monitoring, neurologists ask for assesments on the level of seizure episodes (events), rather than duration or sample-by-sample based metrics. Similarly,  another performance measure with a strong practical impact in epilepsy monitoring, is the false alarm rate (FAR), or the number of false positives per hour/day. Clinicians and patients see this measure as more meaningful than some established metrics in the ML community, and are very demanding in terms of performance, requiring it to be as low as possible for potential wearable applications (e.g., less than 1 FP/day). This also necessitates exceptionally high constraints on the required precision (usually much higher than 99\%).

For this reason, here we provide code that measures performance on the level of events and on a sample-by-sample basis.

## Metrics

In more details, we measures performance on the level of:

- Sample : Performance metric that threats every label sample independently.
- Events (e.g. epileptic seizure) : Classifies each event in both reference and hypothesis based on overlap of both.

Both methods are illustrated in the following figures :

![Illustration of sample based scoring.](https://user-images.githubusercontent.com/747240/248309097-b7f76fde-c87a-41df-812d-9821375b640e.png)
![Illustration of event based scoring.](https://user-images.githubusercontent.com/747240/248308898-64b4ae39-d02f-4f06-9b10-f07aaf6110d1.png)

## Installation

The package can be installed through pip using the following command :

`pip install timescoring`

## Code

The `timescoring` package provides three classes :

- `annotation.Annotation` : store annotations
- `scoring.SampleScoring(ref, hyp)` : Compute sample based scoring
- `scoring.EventScoring(ref, hyp)` : Compute event based scoring

In addition it also provides functions to visualize the output of the scoring algorithm (see `visualization.py`).

### Parameters

Sample based scoring allows to set the sampling frequency of the labels. It defaults to 1 Hz.

Event based scoring allows to define certain parameters which are provided as an instance of `scoring.EventScoring.Parameters` :

- `toleranceStart` (float): Allow some tolerance on the start of an event without counting a false detection. Defaults to 30  # [seconds].
- `toleranceEnd` (float): Allow some tolerance on the end of an event without counting a false detection. Defaults to 60  # [seconds].
- `minOverlap` (float): Minimum relative overlap between ref and hyp for a detection. Defaults to 0 which corresponds to any overlap  # [relative].
- `maxEventDuration` (float): Automatically split events longer than a given duration. Defaults to 5*60  # [seconds].
- `minDurationBetweenEvents` (float): Automatically merge events that are separated by less than the given duration. Defaults to 90 # [seconds].

### Scores

Scores are provided as attributes of the scoring class. The following metrics can be accesses :

- `sensitivity`
- `precision`
- `f1` : F1-score
- `fpRate` : False alarm rate per 24h

## Example of usage

```python
# Loading Annotations #


from timescoring.annotations import Annotation

# Annotation objects can be instantiated from a binary mask

fs = 1
mask = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]

labels = Annotation(mask, fs)

print('Annotation objects contain a representation as a mask and as a list of events:')
print(labels.mask)
print(labels.events)


# Annotation object can also be instantiated from a list of events
fs = 1
numSamples = 10  # In this case the duration of the recording in samples should be provided
events = [(1, 3), (6, 9)]

labels = Annotation(events, fs, numSamples)


# Computing performance score #

from timescoring import scoring
from timescoring import visualization

fs = 1
duration = 66 * 60
ref = Annotation([(8 * 60, 12 * 60), (30 * 60, 35 * 60), (48 * 60, 50 * 60)], fs, duration)
hyp = Annotation([(8 * 60, 12 * 60), (28 * 60, 32 * 60), (50.5 * 60, 51 * 60), (60 * 60, 62 * 60)], fs, duration)
scores = scoring.SampleScoring(ref, hyp)
figSamples = visualization.plotSampleScoring(ref, hyp)

# Scores can also be computed per event
param = scoring.EventScoring.Parameters(
    toleranceStart=30,
    toleranceEnd=60,
    minOverlap=0,
    maxEventDuration=5 * 60,
    minDurationBetweenEvents=90)
scores = scoring.EventScoring(ref, hyp, param)
figEvents = visualization.plotEventScoring(ref, hyp, param)

print("# Event scoring\n" +
      "- Sensitivity : {:.2f} \n".format(scores.sensitivity) +
      "- Precision   : {:.2f} \n".format(scores.precision) +
      "- F1-score    : {:.2f} \n".format(scores.f1) +
      "- FP/24h      : {:.2f} \n".format(scores.fpRate))
```

A presentation explaining these metrics is available [here](https://drive.google.com/file/d/1-k6i2jVpU7bzqnV6zQPUKlfPkO7qaXau/view?usp=sharing).
