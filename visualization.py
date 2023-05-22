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
    plt.plot(ref.mask*0.4 + 0.6, 'k')
    plt.plot(hyp.mask*0.4 + 0.1, 'k')
    
    # Plot Colored dots for detections
    lineFn, = plt.plot(time[fn], fn[fn], 'o', color='tab:purple')
    lineTp, = plt.plot(time[tp], tp[tp], 'o', color='tab:green')
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
    plt.show()
