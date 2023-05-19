import unittest

import numpy as np

from annotations import Annotation
import scoring 

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
        mask = [0,1,1,0,0,0,1,1,1,0]
        events = [(0.1, 0.3), (0.6, 0.9)]
        TestAnnotation.checkMaskEvents(mask, events, fs, numSamples, 'Simple events')


        # Event == File duration
        mask = [1,1,1,1,1,1,1,1,1,1]
        events = [(0.0, 1.0)]
        TestAnnotation.checkMaskEvents(mask, events, fs, numSamples, 'Event = file duration')

        # Event at start
        mask = [1,1,1,1,0,0,1,0,0,0]
        events = [(0.0, 0.4), (0.6, 0.7)]
        TestAnnotation.checkMaskEvents(mask, events, fs, numSamples, 'event at start')

        # Event at end
        mask = [0,1,1,0,0,0,0,1,1,1]
        events = [(0.1, 0.3), (0.7, 1.)]
        TestAnnotation.checkMaskEvents(mask, events, fs, numSamples, 'event at end')

        # Event at start and end
        mask = [1,1,1,0,0,0,0,1,0,1]
        events = [(0.0, 0.3), (0.7, 0.8), (0.9, 1.0)]
        TestAnnotation.checkMaskEvents(mask, events, fs, numSamples, 'event at start and end')
        

class TestWindowScoring(unittest.TestCase):
    def test_window_scoring(self): 
        fs = 10
        
        # Simple events
        ref = Annotation([1,1,1,0,0,0,1,1,1,0], fs)
        hyp = Annotation([0,1,1,0,1,1,0,0,1,0], fs)
        scores = scoring.WindowScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, 3/6, 'sensitivity simple test')
        np.testing.assert_equal(scores.precision, 3/5, 'precision simple test') 
        np.testing.assert_equal(scores.fpRate, 2*3600*24, 'FP/day simple test')

        # No detections
        ref = Annotation([1,1,1,0,0,0,1,1,1,0], fs)
        hyp = Annotation([0,0,0,0,0,0,0,0,0,0], fs)
        scores = scoring.WindowScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, 0, 'sensitivity no detections')
        np.testing.assert_equal(scores.precision, np.nan, 'precision no detections') 
        np.testing.assert_equal(scores.fpRate, 0, 'FP/day no detections')
        
        # No events
        ref = Annotation([0,0,0,0,0,0,0,0,0,0], fs)
        hyp = Annotation([0,1,1,0,1,1,0,0,1,0], fs)
        scores = scoring.WindowScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, np.nan, 'sensitivity no events')
        np.testing.assert_equal(scores.precision, 0, 'precision no events') 
        np.testing.assert_equal(scores.fpRate, 5*3600*24, 'FP/day no events')


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
            self.assertLessEqual(event[1]-event[0], maxEventDuration + 1e-10,  # Computre precision tolerance
                            "Long event splitting resulted in long event")
        
        # Check mask remains unchanged
        TestAnnotation.assertMask(labels.mask, shortLabels.mask, 
                                  "Event mask changed when splitting events")
        
        
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
        scores = scoring.EventScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, 1, 'sensitivity no detections')
        np.testing.assert_equal(scores.precision, 0.5, 'precision no detections') 
        np.testing.assert_equal(scores.fpRate, 1*24, 'FP/day no detections')
        
        # Split long events
        # REF <----->
        # HYP   <-------------------------->
        #SPLIT  <----------------><-------->
        ref = Annotation([(40, 60)], fs, numSamples)
        hyp = Annotation([(42, 65 + 6*60)], fs, numSamples)
        scores = scoring.EventScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, 1, 'sensitivity no detections')
        np.testing.assert_equal(scores.precision, 1/3, 'precision no detections') 
        np.testing.assert_equal(scores.fpRate, 2*24, 'FP/day no detections')

        # No detections
        ref = Annotation([(40, 60)], fs, numSamples)
        hyp = Annotation([], fs, numSamples)
        scores = scoring.EventScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, 0, 'sensitivity no detections')
        np.testing.assert_equal(scores.precision, np.nan, 'precision no detections') 
        np.testing.assert_equal(scores.fpRate, 0, 'FP/day no detections')
        
        # No events
        ref = Annotation([], fs, numSamples)
        hyp = Annotation([(40, 60)], fs, numSamples)
        scores = scoring.EventScoring(ref, hyp)
        np.testing.assert_equal(scores.sensitivity, np.nan, 'sensitivity no events')
        np.testing.assert_equal(scores.precision, 0, 'precision no events') 
        np.testing.assert_equal(scores.fpRate, 1*24, 'FP/day no events')


if __name__ == '__main__':
    unittest.main()
