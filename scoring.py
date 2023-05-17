
import numpy as np

from annotations import Annotation

class _Scoring:
    """" Base class for different scoring methods. The class provides the common
    attributes and computation of common scores based on these attributes.
    """
    sefs : int
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
        elif (self.fp + self.sensitivity) == 0:  # No overlap ref & hyp
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