''' Unit testing
'''

__author__ = "Jonathan Dan"
__email__ = "jonathan.dan at epfl.ch"

import unittest

import numpy as np

from src.timescoring.annotations import Annotation
from src.timescoring import scoring


class TestAnnotation(unittest.TestCase):
    def assertListOfTupleEqual(expected, actual, message):
        difference = set(expected) ^ set(actual)
        assert not difference, 'List of events check failed : {}'.format(message)

    def assertMask(expected, actual, message):
        difference = np.sum(expected ^ actual)
        assert not difference, 'Mask check failed : {}'.format(message)

    def checkMaskEvents(mask, events, fs, numSamples, message):
        labels = Annotation(events, fs, numSamples)
        TestAnnotation.assertMask(mask, labels.mask, message)

        labels = Annotation(mask, fs)
        TestAnnotation.assertListOfTupleEqual(events, labels.events, message)

    def test_mask_to_eventList(self):
        fs = 10
        numSamples = 10

        # Simple events
        mask = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
        events = [(0.1, 0.3), (0.6, 0.9)]
        TestAnnotation.checkMaskEvents(mask, events, fs, numSamples, 'Simple events')

        # Event == File duration
        mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        events = [(0.0, 1.0)]
        TestAnnotation.checkMaskEvents(mask, events, fs, numSamples, 'Event = file duration')

        # Event at start
        mask = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]
        events = [(0.0, 0.4), (0.6, 0.7)]
        TestAnnotation.checkMaskEvents(mask, events, fs, numSamples, 'event at start')

        # Event at end
        mask = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
        events = [(0.1, 0.3), (0.7, 1.)]
        TestAnnotation.checkMaskEvents(mask, events, fs, numSamples, 'event at end')

        # Event at start and end
        mask = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1]
        events = [(0.0, 0.3), (0.7, 0.8), (0.9, 1.0)]
        TestAnnotation.checkMaskEvents(mask, events, fs, numSamples, 'event at start and end')


class TestSampleScoring(unittest.TestCase):
    def test_sample_scoring(self):
        fs = 1

        # Simple events
        ref = Annotation([1, 1, 1, 0, 0, 0, 1, 1, 1, 0], fs)
        hyp = Annotation([0, 1, 1, 0, 1, 1, 0, 0, 1, 0], fs)
        scores = scoring.SampleScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, 3 / 6, 'sensitivity simple test')
        np.testing.assert_equal(scores.precision, 3 / 5, 'precision simple test')
        np.testing.assert_equal(scores.fpRate, 2 * 3600 * 24 / 10, 'FP / day simple test')

        # No detections
        ref = Annotation([1, 1, 1, 0, 0, 0, 1, 1, 1, 0], fs)
        hyp = Annotation([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], fs)
        scores = scoring.SampleScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, 0, 'sensitivity no detections')
        np.testing.assert_equal(scores.precision, np.nan, 'precision no detections')
        np.testing.assert_equal(scores.fpRate, 0, 'FP / day no detections')

        # No events
        ref = Annotation([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], fs)
        hyp = Annotation([0, 1, 1, 0, 1, 1, 0, 0, 1, 0], fs)
        scores = scoring.SampleScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, np.nan, 'sensitivity no events')
        np.testing.assert_equal(scores.precision, 0, 'precision no events')
        np.testing.assert_equal(scores.fpRate, 5 * 3600 * 24 / 10, 'FP / day no events')

        # Resampling
        fs = 256
        ref = Annotation([(1.2, 5.4), (7.1, 8.7)], fs, fs * 10)
        hyp = Annotation([(3, 5), (6.8, 9.2)], fs, fs * 10)
        scores = scoring.SampleScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, 4 / 6, 'sensitivity resampling')
        np.testing.assert_equal(scores.precision, 1, 'precision resampling')
        np.testing.assert_equal(scores.fpRate, 0, 'FP / day resampling')


class TestEventScoring(unittest.TestCase):
    def test_long_event_splitting(self):
        fs = 10
        numSamples = 10
        events = [(0.0, 0.5), (0.6, 0.9)]
        maxEventDuration = 0.2

        labels = Annotation(events, fs, numSamples)
        shortLabels = scoring.EventScoring._splitLongEvents(labels, maxEventDuration)

        # Check maximum duration
        for event in shortLabels.events:
            self.assertLessEqual(event[1] - event[0], maxEventDuration + 1e-10,  # Computre precision tolerance
                                 "Long event splitting resulted in long event")

        # Check mask remains unchanged
        TestAnnotation.assertMask(labels.mask, shortLabels.mask,
                                  "Event mask changed when splitting events")

    def test_merge_neighbouring_events(self):
        fs = 10
        labels = Annotation([(1, 2), (2, 3), (3.5, 4), (4, 5), (18, 20)], fs, 600)
        minDurationBetweenEvents = 0.2

        mergedEvents = scoring.EventScoring._mergeNeighbouringEvents(labels, minDurationBetweenEvents)
        expectedMergedEvents = Annotation([(1, 3), (3.5, 5), (18, 20)], fs, 600)
        TestAnnotation.assertListOfTupleEqual(expectedMergedEvents.events, mergedEvents.events, 'Test merge events')

    def test_extending_events(self):
        fs = 1
        numSamples = 10
        events = [(0.0, 2.0), (6.0, 8.0)]
        before = 0.5
        after = 1.0

        target = [(0.0, 3.0), (5.5, 9.0)]

        labels = Annotation(events, fs, numSamples)
        extendedLabels = scoring.EventScoring._extendEvents(labels, before, after)

        TestAnnotation.assertListOfTupleEqual(target, extendedLabels.events,
                                              "Extended events mismatch")

    def test_event_scoring(self):
        fs = 10
        numSamples = 60 * 60 * fs  # 1 hour

        # Simple events
        ref = Annotation([(40, 60)], fs, numSamples)
        hyp = Annotation([(10, 20), (42, 65)], fs, numSamples)
        param = scoring.EventScoring.Parameters(toleranceStart=0,
                                                toleranceEnd=10,
                                                minDurationBetweenEvents=0)
        scores = scoring.EventScoring(ref, hyp, param)
        np.testing.assert_equal(scores.sensitivity, 1, 'sensitivity no detections')
        np.testing.assert_equal(scores.precision, 0.5, 'precision no detections')
        np.testing.assert_equal(scores.fpRate, 1 * 24, 'FP / day no detections')

        # Tolerance before events
        # REF      <----->
        # HYP    <------->
        ref = Annotation([(40, 60)], fs, numSamples)
        hyp = Annotation([(39, 60)], fs, numSamples)
        param = scoring.EventScoring.Parameters(toleranceStart=1)
        scores = scoring.EventScoring(ref, hyp, param)
        np.testing.assert_equal(scores.sensitivity, 1, 'sensitivity no detections')
        np.testing.assert_equal(scores.precision, 1, 'precision no detections')
        np.testing.assert_equal(scores.fpRate, 0, 'FP / day no detections')

        # Split long events
        # REF <----->
        # HYP   <-------------------------->
        # SPLIT  <----------------><-------->
        ref = Annotation([(40, 60)], fs, numSamples)
        hyp = Annotation([(42, 65 + 6 * 60)], fs, numSamples)
        param = scoring.EventScoring.Parameters(maxEventDuration=5 * 60)
        scores = scoring.EventScoring(ref, hyp, param)
        np.testing.assert_equal(scores.sensitivity, 1, 'sensitivity no detections')
        np.testing.assert_equal(scores.precision, 1 / 3, 'precision no detections')
        np.testing.assert_equal(scores.fpRate, 2 * 24, 'FP / day no detections')

        # No detections
        ref = Annotation([(40, 60)], fs, numSamples)
        hyp = Annotation([], fs, numSamples)
        scores = scoring.EventScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, 0, 'sensitivity no detections')
        np.testing.assert_equal(scores.precision, np.nan, 'precision no detections')
        np.testing.assert_equal(scores.fpRate, 0, 'FP / day no detections')

        # No events
        ref = Annotation([], fs, numSamples)
        hyp = Annotation([(40, 60)], fs, numSamples)
        scores = scoring.EventScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, np.nan, 'sensitivity no events')
        np.testing.assert_equal(scores.precision, 0, 'precision no events')
        np.testing.assert_equal(scores.fpRate, 1 * 24, 'FP / day no events')

        #
        # Seizure with typical distribution, some overlapping some not
        ref = Annotation([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         fs)
        hyp = Annotation([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                          1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                         fs)
        param = scoring.EventScoring.Parameters(
            toleranceStart=0,
            toleranceEnd=0,
            minOverlap=0,
            maxEventDuration=5 * 60,
            minDurationBetweenEvents=0)
        scores = scoring.EventScoring(ref, hyp, param)
        np.testing.assert_equal(scores.sensitivity, 2 / 3, 'sensitivity typical distribution')
        np.testing.assert_equal(scores.precision, 0.4, 'precision typical distribution')

        param.minOverlap = 0.5
        scores = scoring.EventScoring(ref, hyp, param)
        np.testing.assert_equal(scores.sensitivity, 1 / 3, 'sensitivity typical distribution')
        np.testing.assert_equal(scores.precision, 1 / 4, 'precision typical distribution')

        #
        # Long hypothesis seizure spaning several true seizures
        message = 'Long predicted seizure spaning several true seizures'
        ref = Annotation([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                          0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         fs)
        hyp = Annotation([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         fs)

        param.minOverlap = 0.0
        scores = scoring.EventScoring(ref, hyp, param)
        np.testing.assert_equal(scores.sensitivity, 1, 'sensitivity : ' + message)
        np.testing.assert_equal(scores.precision, 5 / 7, 'precision : ' + message)

        #
        # Long true seizure and many short predicted seizures​
        message = 'Long true seizure and many short predicted seizures​'
        ref = Annotation([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         fs)
        hyp = Annotation([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                          0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                         fs)

        param.minOverlap = 0.0
        scores = scoring.EventScoring(ref, hyp, param)
        np.testing.assert_equal(scores.sensitivity, 0.5, 'sensitivity : ' + message)
        np.testing.assert_equal(scores.precision, 0.25, 'precision : ' + message)

        # Typial distribution (one missed extended REF would hide FP)
        fs = 1
        ref = Annotation([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         fs)
        hyp = Annotation([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                          1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                         fs)
        message = 'typical - extended REF would hide FP'
        param.minOverlap = 0.6
        scores = scoring.EventScoring(ref, hyp, param)
        np.testing.assert_equal(scores.sensitivity, 1 / 3, 'sensitivity : ' + message)
        np.testing.assert_equal(scores.precision, 0.25, 'precision : ' + message)


if __name__ == '__main__':
    unittest.main()
