from __future__ import absolute_import

from .Algorithm import Algorithm
from .AlgorithmException import AlgorithmException
import numpy as np
from numpy.lib import stride_tricks as npls
import scipy.signal as sps
from obspy.core import Stream, Stats


class FilterAlgorithm(Algorithm):
    """
        Filter Algorithm that filters and downsamples data
    """

    def __init__(self, window=None, decimation=1, sample_period=None,
                 location=None, inchannels=None, outchannels=None):
        
        Algorithm.__init__(self, inchannels=inchannels,
            outchannels=outchannels)
        
        if window is None:
            # default to standard INTERMAGNET one-minute filter coefficients
            self.numtaps = 91
            self.window = sps.get_window(window=('gaussian', 15.8734),
                                                    Nx=self.numtaps)
        else:
            self.numtaps = len(window)
            self.window = window
        
        self.decimation = decimation
        self.sample_period = sample_period
        self.location = location
        self.inchannels = inchannels
        self.outchannels = outchannels
        
        # always normalize filter window
        self.window = self.window / np.sum(self.window)

    def create_trace(self, channel, stats, data):
        """Utility to create a new trace object.

        This may be necessary for more sophisticated metadata modifications, but
        for now it simply passes inputs back to parent Algorithm class.

        Parameters
        ----------
        channel : str
            channel name.
        stats : obspy.core.Stats
            channel metadata to clone.
        data : numpy.array
            channel data.

        Returns
        -------
        obspy.core.Trace
            trace containing data and metadata.
        """

        trace = super(FilterAlgorithm, self).create_trace(channel, stats,
            data)
        return trace

    def process(self, stream):
        """Run algorithm for a stream.
        Processes all traces in the stream.
        Parameters
        ----------
        stream : obspy.core.Stream
            stream of data to process
        Returns
        -------
        out : obspy.core.Stream
            stream containing 1 trace per original trace.
        """

        out = Stream()

        tr_i = 0
        for trace in stream:
            tr = trace.copy()
            if self.sample_period is None:
                # pad inputs if sample_period wasn't specified
                tr.trim(
                    starttime=(tr.stats.starttime -
                        self.numtaps // 2 * tr.stats.delta),
                    endtime=(tr.stats.endtime +
                        self.numtaps // 2 * tr.stats.delta),
                    pad=True)
            data = tr.data
            step = self.decimation
            
            filtered = self.firfilter(data, self.window, step)

            stats = Stats(tr.stats)
            stats.starttime = tr.stats.starttime + \
                    self.numtaps // 2 * stats.delta
            stats.delta = stats.delta * step
            stats.npts = filtered.shape[0]
            
            # user may need to change location code
            if not self.location is None:
                stats.location = self.location

            # user may need to change output channel codes
            if not self.outchannels is None:
                stats.channel = self.outchannels[tr_i]
                tr_i += 1

            trace_out = self.create_trace(
                stats.channel, stats, filtered)

            out += trace_out

        return out

    @staticmethod
    def firfilter(data, window, step, allowed_bad=0.1):
        """Run fir filter for a numpy array.
        Processes all traces in the stream.
        Parameters
        ----------
        data: numpy.ndarray
            array of data to process
        window: numpy.ndarray
            array of filter coefficients
        step: int
            ratio of output sample period to input sample period
            should always be an integer
        allowed_bad: float
            ratio of bad samples to total window size
        Returns
        -------
        filtered_out : numpy.ndarray
            stream containing filtered output
        """

        if step == 0 or not step % 1 == 0:
            raise AlgorithmException(
              'decimation step must be non-zero integer')
        else:
            step = int(step)
        
        numtaps = len(window)

        # build view into data, with numtaps  chunks separated into
        # overlapping 'rows'
        shape = data.shape[:-1] + \
                (data.shape[-1] - numtaps + 1, numtaps)
        strides = data.strides + (data.strides[-1],)
        as_s = npls.as_strided(data, shape=shape, strides=strides,
                               writeable=False)

        # build masked array for invalid entries, also 'decimate' by step
        as_masked = np.ma.masked_invalid(as_s[::step], copy=True)
        # sums of the total 'weights' of the filter corresponding to
        # valid samples
        as_weight_sums = np.dot(window, (~as_masked.mask).T)
        # mark the output locations as 'bad' that have missing input weights
        # that sum to greater than the allowed_bad threshhold
        as_invalid_masked = np.ma.masked_less(as_weight_sums, 1 - allowed_bad)

        # apply filter, using masked version of dot (in 3.5 and above, there
        # seems to be a move toward np.matmul and/or @ operator as opposed to
        # np.dot/np.ma.dot - haven't tested to see if the shape of first and
        # second argument need to be changed)
        filtered = np.ma.dot(window, as_masked.T)
        # re-normalize, especially important for partially filled windows
        filtered = np.divide(filtered, as_weight_sums)
        # use the earlier marked output locations to mask the output data
        # array
        filtered.mask = as_invalid_masked.mask
        # convert masked array back to regular array, with nan as fill value
        # (otherwise the type returned is not always the same, and can cause
        # problems with factories, merge, etc.)
        filtered_out = np.ma.filled(filtered, np.nan)

        return filtered_out

    def get_input_interval(self, start, end, observatory=None, channels=None):
        """Get Input Interval

        start : UTCDateTime
            start time of requested output.
        end : UTCDateTime
            end time of requested output.
        observatory : string
            observatory code.
        channels : string
            input channels.

        Returns
        -------
        input_start : UTCDateTime
            start of input required to generate requested output
        input_end : UTCDateTime
            end of input required to generate requested output.
        """

        half = self.numtaps // 2
        start = start - half * self.sample_period
        end = end + half * self.sample_period

        return (start, end)

    @classmethod
    def add_arguments(cls, parser):
        """Add command line arguments to argparse parser.
        Parameters
        ----------
        parser: ArgumentParser
            command line argument parser
        """
        parser.add_argument('--filter-interval',
            help='Allowed sampling intervals for filtered output',
            choices=['daily', 'day', 'hourly', 'hour', 'minute', 'second', 'tenhertz'])

    def configure(self, arguments):
        """Configure algorithm using comand line arguments.
        Parameters
        ----------
        arguments: Namespace
            parsed command line arguments
        """
        Algorithm.configure(self, arguments)
        self.inchannels = self._inchannels
        self.outchannels = self._outchannels

        if arguments.interval in ['tenhertz']:
            self.sample_period = 0.1
        elif arguments.interval in ['second']:
            self.sample_period = 1.0
        elif arguments.interval in ['minute']:
            self.sample_period = 60.0
        elif arguments.interval in ['hour', 'hourly'] :
            self.sample_period = 3600.0
        elif arguments.interval in ['day','daily']:
            self.sample_period = 86400.0
        
        if arguments.filter_interval is None:
            self.decimation = 1
        elif arguments.filter_interval in ['tenhertz']:
            self.decimation = 0.1 / self.sample_period
        elif arguments.filter_interval in ['second']:
            self.decimation = 1.0 / self.sample_period
        elif arguments.filter_interval in ['minute']:
            self.decimation = 60.0 / self.sample_period
        elif arguments.filter_interval in ['hour', 'hourly']:
            self.decimation = 3600.0 / self.sample_period
        elif arguments.filter_interval in ['day', 'daily']:
            self.decimation = 86400.0 / self.sample_period

        self.location = arguments.outlocationcode
