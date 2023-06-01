import matplotlib.pyplot as plt
import numpy as np

from annotations import Annotation
import scoring

def plotWindowScoring(ref : Annotation, hyp : Annotation) -> plt.figure:
    """Build an overview plot showing the outcome of window scoring.

    Args:
        ref (Annotation): Reference annotations (ground-truth)
        hyp (Annotation): Hypotheses annotations (output of a ML pipeline)

    Returns:
        plt.figure: Output matplotlib figure
    """
    
    score = scoring.WindowScoring(ref, hyp)
    time = np.arange(len(ref.mask)) / ref.fs
    
    # Compute event masks
    tp = ref.mask & hyp.mask
    fp = ~ref.mask & hyp.mask
    fn = ref.mask & ~hyp.mask

    fig = plt.figure(figsize=(16, 3))
    
    # Plot background shading
    plt.fill_between(time, 0, 1, where=tp, 
                     alpha=0.2, color='tab:green',
                     transform=plt.gca().get_xaxis_transform())
    plt.fill_between(time, 0, 1, where=fn, 
                     alpha=0.2, color='tab:purple',
                     transform=plt.gca().get_xaxis_transform())
    plt.fill_between(time, 0, 1, where=fp,
                     alpha=0.2, color='tab:red',
                     transform=plt.gca().get_xaxis_transform())
    
    # Plot Labels
    plt.plot(time, ref.mask*0.4 + 0.6, 'k')
    plt.plot(time, hyp.mask*0.4 + 0.1, 'k')
    
    # Plot Colored dots for detections
    lineFn, = plt.plot(time[fn], fn[fn], 'o', color='tab:purple')
    lineTp, = plt.plot(time[tp], tp[tp], 'o', color='tab:green')
    plt.plot(time[tp], tp[tp]*0.5, 'o', color='tab:green')
    lineFp, = plt.plot(time[fp], fp[fp]*0.5, 'o', color='tab:red')
        
    # Text  
    plt.title('Window Scoring')

    plt.yticks([0.3, 0.8], ['HYP', 'REF'])
    plt.xlabel('time [s]')
    
    plt.legend([lineTp, lineFn, lineFp],
               ['TP : {}'.format(np.sum(tp)), 
                'FN : {}'.format(np.sum(fn)),
                'FP : {}'.format(np.sum(fp))], loc=(1.02, 0.65))
    
    textstr = "• Sensitivity : {:.2f}\n".format(score.sensitivity)
    textstr+= "• Precision   : {:.2f}\n".format(score.precision)
    textstr+= "• F1-score    : {:.2f}".format(score.f1)
    fig.text(1.02, 0.05, textstr, fontsize=12, transform=plt.gca().transAxes)
    
    # Adjust spacing
    plt.margins(x=0)  # No margin on X data
    plt.tight_layout()
    fig.subplots_adjust(right=0.86)  # Allow space for scoring text
        
    return fig


def plotEventScoring(ref : Annotation, hyp : Annotation, param : scoring.EventScoring.Parameters = scoring.EventScoring.Parameters()) -> plt.figure:
    """Build an overview plot showing the outcome of event scoring.

    Args:
        ref (Annotation): Reference annotations (ground-truth)
        hyp (Annotation): Hypotheses annotations (output of a ML pipeline)
        param(EventScoring.Parameters, optional):  Parameters for event scoring.
            Defaults to default values.

    Returns:
        plt.figure: Output matplotlib figure
    """
    def _plotEvent(x, y, color):
            plt.axvspan(x[0], x[1], alpha=0.2, color=color)
            if x[1] - x[0] > 0:
                plt.plot(x, y, color=color, linewidth=5, solid_capstyle='butt')
            else:
                plt.scatter(x[0], y[0], color=color)
    
    
    ref = scoring.EventScoring._splitLongEvents(ref, param.maxEventDuration)
    hyp = scoring.EventScoring._splitLongEvents(hyp, param.maxEventDuration)
    score = scoring.EventScoring(ref, hyp, param)
    time = np.arange(len(ref.mask)) / ref.fs

    fig = plt.figure(figsize=(16, 3))
    
    # Plot Labels
    plt.plot(time, ref.mask*0.4 + 0.6, 'k')
    plt.plot(time, hyp.mask*0.4 + 0.1, 'k')
    
    # Initialize lines for legend
    lineTp, = plt.plot([], [], color='tab:green', linewidth=5)
    lineFn, = plt.plot([], [], color='tab:purple', linewidth=5)
    lineFp, = plt.plot([], [], color='tab:red', linewidth=5)
    
    # Plot TP & FN
    detectionMask = np.zeros_like(ref.mask)
    for event in ref.events:
        # TP
        if (np.sum(hyp.mask[int(event[0]*hyp.fs):int(event[1]*hyp.fs)])/hyp.fs)/(event[1]-event[0]) > param.minOverlap:
            _plotEvent([event[0], event[1]-(1/hyp.fs)], [1, 1], 'tab:green')
            detectionMask[int(event[0]*ref.fs):int(event[1]*ref.fs)] = 1
        else:
            _plotEvent([event[0], event[1]-(1/hyp.fs)], [1, 1], 'tab:purple')
    
    # Plot FP 
    extendedDetections = scoring.EventScoring._extendEvents(Annotation(detectionMask, ref.fs), param.toleranceStart, param.toleranceEnd)
    for event in hyp.events:
        fpFlag = False
        if np.any(~extendedDetections.mask[int(event[0]*extendedDetections.fs):int(event[1]*extendedDetections.fs)]):
            _plotEvent([event[0], event[1]-(1/hyp.fs)], [0.5, 0.5], 'tab:red')
            fpFlag = True
        if np.any(extendedDetections.mask[int(event[0]*extendedDetections.fs):int(event[1]*extendedDetections.fs)]):
            if fpFlag :
                lineStyle = (0, (2, 2))
            else:
                lineStyle = 'solid'
            plt.plot([event[0], event[1]-(1/hyp.fs)], [0.5, 0.5], color='tab:green', linewidth=5, solid_capstyle='butt', linestyle=lineStyle)

    # Text  
    plt.title('Event Scoring')

    plt.yticks([0.3, 0.8], ['HYP', 'REF'])
    plt.xlabel('time [s]')
    
    plt.legend([lineTp, lineFn, lineFp],
               ['TP : {}'.format(np.sum(score.tp)), 
                'FN : {}'.format(np.sum(score.refTrue - score.tp)),
                'FP : {}'.format(np.sum(score.fp))], loc=(1.02, 0.65))
    
    textstr = "• Sensitivity : {:.2f}\n".format(score.sensitivity)
    textstr+= "• Precision   : {:.2f}\n".format(score.precision)
    textstr+= "• F1-score    : {:.2f}".format(score.f1)
    fig.text(1.02, 0.05, textstr, fontsize=12, transform=plt.gca().transAxes)
    
    # Adjust spacing
    plt.margins(x=0)  # No margin on X data
    plt.tight_layout()
    fig.subplots_adjust(right=0.86)  # Allow space for scoring text
        
    return fig


if __name__ == "__main__":
    fs = 1
    ref = Annotation([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,1,1,1,1,1,1,1,0,0,0, 0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0],
                        fs)
    hyp = Annotation([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
                        1,1,1,1,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,
                        0,0,0,0,0,0,0,0,0,1,1,1,1],
                        fs)

    fig = plotWindowScoring(ref, hyp)
    fig = plotEventScoring(ref, hyp)
    plt.show()
