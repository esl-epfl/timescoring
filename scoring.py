''' Scoring functions between a reference annotation (ground-truth) and hypotheses (e.g. ML output).
'''

__author__ = "Jonathan Dan, Una Pale"
__email__ = "jonathan.dan at epfl.ch"

import numpy as np

from annotations import Annotation

class _Scoring:
    """" Base class for different scoring methods. The class provides the common
    attributes and computation of common scores based on these attributes.
    """
    fs : int
    numSamples : int
    
    refTrue : int
    tp : int
    fp : int

    sensitivity : float
    precision : float
    f1 : float
    fpRate : float
    
    def computeScores(self):
        """ Compute performance metrics."""
        # Sensitivity
        if self.refTrue > 0:
            self.sensitivity = self.tp / self.refTrue
        else:
            self.sensitivity = np.nan  # no ref event
        
        # Precision
        if self.tp + self.fp > 0:
            self.precision = self.tp / (self.tp + self.fp)
        else:
            self.precision = np.nan  # no hyp event
            
        # F1 Score
        if np.isnan(self.sensitivity) or np.isnan(self.precision):
            self.f1 = np.nan
        elif (self.sensitivity + self.precision) == 0:  # No overlap ref & hyp
            self.f1 = 0
        else:
            self.f1 = 2 * self.sensitivity * self.precision / (self.sensitivity + self.precision)
            
        # FP Rate
        self.fpRate = self.fp / (self.numSamples / self.fs / 3600 / 24)  # FP per day
    
    
class WindowScoring(_Scoring):
    """Calculates performance metrics on the sample by sample basis"""
    
    def __init__(self, ref : Annotation, hyp : Annotation):
        """Computes a scoring on a sample by sample basis.

        Args:
            ref (Annotation): Reference annotations (ground-truth)
            hyp (Annotation): Hypotheses annotations (output of a ML pipeline)
        """
        self.fs = ref.fs
        self.numSamples = len(ref.mask)
        
        self.refTrue = np.sum(ref.mask)
        
        self.tp = np.sum(hyp.mask[ref.mask])
        self.fp = np.sum(hyp.mask) - self.tp
        
        self.computeScores()
        
        
class EventScoring(_Scoring):
    """Calculates performance metrics on an event basis"""
    class Parameters:
        """Parameters for event scoring"""
        
        def __init__(self, toleranceStart : float = 1,
                     toleranceEnd : float = 10,
                     minOverlap : float = 0.66,
                     maxEventDuration : float = 5*60):
            """Parameters for event scoring

            Args:
                toleranceStart (float): Allow some tolerance on the start of an event 
                    without counting a false detection. Defaults to 1  # [seconds].
                toleranceEnd (float): Allow some tolerance on the end of an event 
                    without counting a false detection. Defaults to 10  # [seconds].
                minOverlap (float): Minimum relative overlap between ref and hyp for 
                    a detection. Defaults to 0.66  # [relative].
                maxEventDuration (float): Automatically split events longer than a 
                    given duration. Defaults to 5*60  # [seconds].
            """   
            self.toleranceStart = toleranceStart
            self.toleranceEnd = toleranceEnd
            self.minOverlap = minOverlap
            self.maxEventDuration = maxEventDuration      

    
    def __init__(self, ref : Annotation, hyp : Annotation, param : Parameters = Parameters()):
        """Computes a scoring on an event basis.

        Args:
            ref (Annotation): Reference annotations (ground-truth)
            hyp (Annotation): Hypotheses annotations (output of a ML pipeline)
            param(EventScoring.Parameters, optional):  Parameters for event scoring.
                Defaults to default values.
        """
        # Split long events to param.maxEventDuration
        ref = EventScoring._splitLongEvents(ref, param.maxEventDuration)
        hyp = EventScoring._splitLongEvents(hyp, param.maxEventDuration)
        
        self.fs = ref.fs
        self.numSamples = len(ref.mask)
        
        self.refTrue = len(ref.events)
        
        # Count True detections
        self.tp = 0
        detectionMask = np.zeros_like(ref.mask)
        for event in ref.events:
            if (np.sum(hyp.mask[int(event[0]*hyp.fs):int(event[1]*hyp.fs)])/hyp.fs)/(event[1]-event[0]) > param.minOverlap:
                self.tp +=1
                detectionMask[int(event[0]*ref.fs):int(event[1]*ref.fs)] = 1
                
        # Count False detections
        self.fp = 0
        extendedDetections = EventScoring._extendEvents(Annotation(detectionMask, ref.fs), param.toleranceStart, param.toleranceEnd)
        for event in hyp.events:
            if np.any(~extendedDetections.mask[int(event[0]*extendedDetections.fs):int(event[1]*extendedDetections.fs)]):
                self.fp +=1
        
        self.computeScores()
        
        
    def _splitLongEvents(events : Annotation, maxEventDuration : float) -> Annotation:
        """Split events longer than maxEventDuration in shorter events.
        Args:
            events (Annotation): Annotation object containing events to split
            maxEventDuration (float): maximum duration of an event [seconds]

        Returns:
            Annotation: Returns a new Annotation instance with all events split to
                a maximum duration of maxEventDuration.
        """
        
        shorterEvents = events.events.copy()
        
        for i, event in enumerate(shorterEvents):
            if event[1] - event[0] > maxEventDuration:
                shorterEvents[i] = (event[0], event[0] + maxEventDuration)
                shorterEvents.insert(i + 1, (event[0] + maxEventDuration, event[1]))
                
        return Annotation(shorterEvents, events.fs, len(events.mask))
    
    
    def _extendEvents(events : Annotation, before : float, after : float) -> Annotation:
        """_summary_

        Args:
            events (Annotation): Annotation object containing events to extend
            before (float): Time to extend before each event [seconds]
            after (float):  Time to extend after each event [seconds]

        Returns:
            Annotation: Returns a new Annotation instance with all events extended
        """
        
        extendedEvents = events.events.copy()
        fileDuration = len(events.mask) / events.fs
        
        for i, event in enumerate(extendedEvents):
            extendedEvents[i] = (max(0, event[0] - before), (min(fileDuration, event[1] + after)))
            
        return Annotation(extendedEvents, events.fs, len(events.mask))
        