import matplotlib.pyplot as plt
import numpy as np

from .annotations import Annotation
from . import scoring


def plotSampleScoring(ref: Annotation, hyp: Annotation, fs: int = 1) -> plt.figure:
    """Build an overview plot showing the outcome of sample scoring.

    Args:
        ref (Annotation): Reference annotations (ground - truth)
        hyp (Annotation): Hypotheses annotations (output of a ML pipeline)
        fs (int): Sampling frequency of the labels. Default 1 Hz.

    Returns:
        plt.figure: Output matplotlib figure
    """

    score = scoring.SampleScoring(ref, hyp, fs)
    time = np.arange(len(score.tpMask)) / fs
    # Resample Data
    ref = Annotation(ref.events, fs, round(len(ref.mask) / ref.fs * fs))
    hyp = Annotation(hyp.events, fs, round(len(hyp.mask) / hyp.fs * fs))

    fig = plt.figure(figsize=(16, 3))

    # Plot background shading
    plt.fill_between(time, 0, 1, where=score.tpMask,
                     alpha=0.2, color='tab:green',
                     transform=plt.gca().get_xaxis_transform())
    plt.fill_between(time, 0, 1, where=score.fnMask,
                     alpha=0.2, color='tab:purple',
                     transform=plt.gca().get_xaxis_transform())
    plt.fill_between(time, 0, 1, where=score.fpMask,
                     alpha=0.2, color='tab:red',
                     transform=plt.gca().get_xaxis_transform())

    # Plot Labels
    plt.plot(time, ref.mask * 0.4 + 0.6, 'k')
    plt.plot(time, hyp.mask * 0.4 + 0.1, 'k')

    # Plot Colored dots for detections
    lineFn, = plt.plot(time[score.fnMask], score.fnMask[score.fnMask], 'o', color='tab:purple')
    lineTp, = plt.plot(time[score.tpMask], score.tpMask[score.tpMask], 'o', color='tab:green')
    plt.plot(time[score.tpMask], score.tpMask[score.tpMask] * 0.5, 'o', color='tab:green')
    lineFp, = plt.plot(time[score.fpMask], score.fpMask[score.fpMask] * 0.5, 'o', color='tab:red')

    # Text
    plt.title('Sample based Scoring')

    plt.yticks([0.3, 0.8], ['HYP', 'REF'])
    _scale_time_xaxis(fig)

    _buildLegend(lineTp, lineFn, lineFp, score, fig)

    return fig


def plotEventScoring(ref: Annotation, hyp: Annotation,
                     param: scoring.EventScoring.Parameters = scoring.EventScoring.Parameters()) -> plt.figure:
    """Build an overview plot showing the outcome of event scoring.

    Args:
        ref (Annotation): Reference annotations (ground - truth)
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

    score = scoring.EventScoring(ref, hyp, param)
    time = np.arange(len(ref.mask)) / ref.fs

    fig = plt.figure(figsize=(16, 3))

    # Plot Labels
    plt.plot(time, ref.mask * 0.4 + 0.6, 'k')
    plt.plot(time, hyp.mask * 0.4 + 0.1, 'k')

    # Initialize lines for legend
    lineTp, = plt.plot([], [], color='tab:green', linewidth=5)
    lineFn, = plt.plot([], [], color='tab:purple', linewidth=5)
    lineFp, = plt.plot([], [], color='tab:red', linewidth=5)

    # Plot REF TP & FN
    for event in ref.events:
        # TP
        if np.any(score.tpMask[round(event[0] * score.fs):round(event[1] * score.fs)]):
            _plotEvent([event[0], event[1] - (1 / ref.fs)], [1, 1], 'tab:green')
            score.tpMask[round(event[0] * score.fs):round(event[1] * score.fs)] = 1
        else:
            _plotEvent([event[0], event[1] - (1 / ref.fs)], [1, 1], 'tab:purple')

    # Plot HYP TP & FP
    for event in hyp.events:
        # FP
        if np.all(~score.tpMask[round(event[0] * score.fs):round(event[1] * score.fs)]):
            _plotEvent([event[0], event[1] - (1 / ref.fs)], [0.5, 0.5], 'tab:red')
        # TP
        elif np.all(score.tpMask[round(event[0] * score.fs):round(event[1] * score.fs)]):
            plt.plot([event[0], event[1] - (1 / ref.fs)], [0.5, 0.5],
                     color='tab:green', linewidth=5, solid_capstyle='butt', linestyle='solid')
        # Mix TP, FP
        else:
            _plotEvent([event[0], event[1] - (1 / ref.fs)], [0.5, 0.5], 'tab:red')
            plt.plot([event[0], event[1] - (1 / ref.fs)], [0.5, 0.5],
                     color='tab:green', linewidth=5, solid_capstyle='butt', linestyle=(0, (2, 2)))

    # Text
    plt.title('Event Scoring')

    plt.yticks([0.3, 0.8], ['HYP', 'REF'])
    _scale_time_xaxis(fig)

    _buildLegend(lineTp, lineFn, lineFp, score, fig)

    return fig


def _scale_time_xaxis(fig: plt.figure):
    """Scale x axis of a figure where initial values are in seconds.

    The function leaves the xaxis as is if the number of seconds to display is < 5 * 60
    If it is larger than 5 minutes, xaxis is formatted as m:s
    If it is larger than 5 hours, xaxis is formatted as h:m:s

    Args:
        fig (plt.figure): figure to handle
    """

    def s2m(x, _):
        return f'{int(x / 60)}:{int(x%60)}'

    def s2h(x, _):
        return f'{int(x / 3600)}:{int((x / 60)%60)}:{int(x%60)}'

    maxTime = fig.gca().get_xlim()[1]
    if maxTime > 5 * 60 * 60:
        fig.gca().xaxis.set_major_formatter(s2h)
        fig.gca().set_xlabel('time [h:m:s]')
    elif maxTime > 5 * 60:
        fig.gca().xaxis.set_major_formatter(s2m)
        fig.gca().set_xlabel('time [m:s]')
    else:
        fig.gca().set_xlabel('time [s]')


def _buildLegend(lineTp, lineFn, lineFp, score, fig):
    """Build legend and adjust spacing for scoring text"""
    plt.legend([lineTp, lineFn, lineFp],
               ['TP: {}'.format(np.sum(score.tp)),
                'FN: {}'.format(np.sum(score.refTrue - score.tp)),
                'FP: {}'.format(np.sum(score.fp))], loc=(1.02, 0.65))

    textstr = "• Sensitivity: {:.2f}\n".format(score.sensitivity)
    textstr += "• Precision  : {:.2f}\n".format(score.precision)
    textstr += "• F1 - score   : {:.2f}".format(score.f1)
    fig.text(1.02, 0.05, textstr, fontsize=12, transform=plt.gca().transAxes)

    # Adjust spacing
    plt.margins(x=0)  # No margin on X data
    plt.tight_layout()
    fig.subplots_adjust(right=0.86)  # Allow space for scoring text


if __name__ == "__main__":
    fs = 1
    duration = 66 * 60
    ref = Annotation([(8 * 60, 12 * 60), (30 * 60, 35 * 60), (48 * 60, 50 * 60)], fs, duration)
    hyp = Annotation([(8 * 60, 12 * 60), (28 * 60, 32 * 60), (50.5 * 60, 51 * 60), (60 * 60, 62 * 60)], fs, duration)

    fig = plotSampleScoring(ref, hyp)
    param = scoring.EventScoring.Parameters(
        toleranceStart=30,
        toleranceEnd=60,
        minOverlap=0,
        maxEventDuration=5 * 60,
        minDurationBetweenEvents=90)
    fig = plotEventScoring(ref, hyp, param)
    plt.show()
