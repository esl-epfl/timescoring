import unittest

import numpy as np

from annotations import Annotation
import scoring 

class TestAnnotation(unittest.TestCase):
    def test_mask_to_eventList(self):
        def assertListOfTupleEqual(expected, actual, message):
            difference = set(expected) ^ set(actual)
            assert not difference, 'List of events check failed : {}'.format(message)
    
        def assertMask(expected, actual, message):
            difference = np.sum(expected - actual)
            assert not difference, 'Mask check failed : {}'.format(message)
            
        def checkMaskEvents(mask, events, fs, numSamples, message):
            labels = Annotation(events, fs, numSamples)
            assertMask(mask, labels.mask, message)
            
            labels = Annotation(mask, fs)
            assertListOfTupleEqual(events, labels.events, message)
        
        fs = 10
        numSamples = 10
        
        # Simple events
        mask = [0,1,1,0,0,0,1,1,1,0]
        events = [(0.1, 0.3), (0.6, 0.9)]
        checkMaskEvents(mask, events, fs, numSamples, 'Simple events')


        # Event == File duration
        mask = [1,1,1,1,1,1,1,1,1,1]
        events = [(0.0, 1.0)]
        checkMaskEvents(mask, events, fs, numSamples, 'Event = file duration')

        # Event at start
        mask = [1,1,1,1,0,0,1,0,0,0]
        events = [(0.0, 0.4), (0.6, 0.7)]
        checkMaskEvents(mask, events, fs, numSamples, 'event at start')

        # Event at end
        mask = [0,1,1,0,0,0,0,1,1,1]
        events = [(0.1, 0.3), (0.7, 1.)]
        checkMaskEvents(mask, events, fs, numSamples, 'event at end')

        # Event at start and end
        mask = [1,1,1,0,0,0,0,1,0,1]
        events = [(0.0, 0.3), (0.7, 0.8), (0.9, 1.0)]
        checkMaskEvents(mask, events, fs, numSamples, 'event at start and end')
        

class TestWindowScoring(unittest.TestCase):
    def test_Window_scoring(self): 
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


if __name__ == '__main__':
    unittest.main()
