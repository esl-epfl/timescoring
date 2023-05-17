import unittest

import numpy as np

from annotations import Annotation

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
if __name__ == '__main__':
    unittest.main()
