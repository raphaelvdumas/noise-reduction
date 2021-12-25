"""
FABADA is a non-parametric noise reduction technique based on Bayesian
inference that iteratively evaluates possibles moothed  models  of
the  data introduced,  obtaining  an  estimation  of the  underlying
signal that is statistically  compatible  with the  noisy  measurements.

based on P.M. Sanchez-Alarcon, Y. Ascasibar, 2022
"Fully Adaptive Bayesian Algorithm for Data Analysis. FABADA"

Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.
"""

from __future__ import print_function, division
import numpy
import pyaudio
from time import sleep
import np_rw_buffer
import numpy as np
from typing import Union
import scipy.stats as stats
from numba import jit
from multiprocessing.pool import Pool

@jit
def fabawrite(vox,buffer):
    buffer.write(fabada1x(vox))

def fabada1x(
        data: Union[np.array, list],
        max_iter: int = 3000,
        **kwargs
) -> np.array:
    """
        FABADA for any kind of data (1D or 2D). Performs noise reduction in input.
        :param data: Noisy measurements, either 1 dimension (M) or 2 dimensions (MxN)
        :param data_variance: Estimated variance of the input, either MxN array, list
                          or float assuming all point have same variance.
    :param max_iter: 3000 (default). Maximum of iterations to converge in solution.
    :param verbose: False (default) or True. Spits some informations about process.
        :param **kwargs: Future Work.
    :return bayes: denoised estimation of the data with same size as input.
    """
    data_variance = numpy.nanvar(data)
    data = np.array(data / 1.0)
    data_variance = np.array(data_variance / 1.0)

    if not kwargs:
        kwargs = {}
        kwargs["debug"] = False

    if data_variance.size != data.size:
        data_variance = data_variance * np.ones_like(data)

    # INITIALIZING ALGORITHM ITERATION ZERO

    posterior_mean = data
    posterior_variance = data_variance
    evidence = Evidence(0, np.sqrt(data_variance), 0, data_variance)
    initial_evidence = evidence
    chi2_pdf, chi2_data, iteration = 0, data.size, 0
    chi2_pdf_derivative, chi2_data_min = 0, data.size
    bayesian_weight = 0
    bayesian_model = 0

    converged = False

    try:
        while not converged:

            chi2_pdf_previous = chi2_pdf
            chi2_pdf_derivative_previous = chi2_pdf_derivative
            evidence_previous = np.mean(evidence)

            iteration += 1  # Check number of iterations done

            # GENERATES PRIORS
            prior_mean = running_mean(posterior_mean)
            prior_variance = posterior_variance

            # APPLIY BAYES' THEOREM
            posterior_variance = 1 / (1 / prior_variance + 1 / data_variance)
            posterior_mean = bayes_theorem(prior_mean, prior_variance, data_variance, posterior_variance, data)

            # EVALUATE EVIDENCE
            evidence = Evidence(prior_mean, data, prior_variance, data_variance)
            evidence_derivative = evidence_derivative1(evidence, evidence_previous)

            # EVALUATE CHI2
            chi2_data = np.sum((data - posterior_mean) ** 2 / data_variance)
            chi2_pdf = stats.chi2.pdf(chi2_data, df=data.size)
            chi2_pdf_derivative = chi2_pdf - chi2_pdf_previous
            chi2_pdf_snd_derivative = chi2_pdf_derivative - chi2_pdf_derivative_previous

            # COMBINE MODELS FOR THE ESTIMATION
            model_weight = evidence * chi2_data
            bayesian_weight += model_weight
            bayesian_model += multiply(model_weight, posterior_mean)

            if iteration == 1:
                chi2_data_min = chi2_data
            # CHECK CONVERGENCE
            if (
                    (chi2_data > data.size and chi2_pdf_snd_derivative >= 0)
                    and (evidence_derivative < 0)
                    or (iteration > max_iter)
            ):
                converged = True

                # COMBINE ITERATION ZERO
                model_weight = initial_evidence * chi2_data_min
                bayesian_weight += model_weight
                bayesian_model += multiply(model_weight, data)

    except:
        #print("Unexpected error:", sys.exc_info()[0])
        raise

    # bayes = divide(bayesian_model,bayesian_weight)
    return divide(bayesian_model, bayesian_weight).astype(numpy.int16)


@jit(nopython=True)
def divide(x, y):
    return x / y


@jit(nopython=True)
def multiply(x, y):
    return x * y


@jit(nopython=True)
def evidence_derivative1(evidence, evidence_previous):
    return np.mean(evidence) - evidence_previous


@jit(nopython=True)
def bayes_theorem(prior_mean, prior_variance, data_variance, posterior_variance, data):
    return (prior_mean / prior_variance + data / data_variance) * posterior_variance


def running_mean(dat):
    mean = np.array(dat)
    dim = len(mean.shape)

    if dim == 1:
        mean[:-1] += dat[1:]
        mean[1:] += dat[:-1]
        mean[1:-1] /= 3
        mean[0] /= 2
        mean[-1] /= 2
    elif dim == 2:
        mean[:-1, :] += dat[1:, :]
        mean[1:, :] += dat[:-1, :]
        mean[:, :-1] += dat[:, 1:]
        mean[:, 1:] += dat[:, :-1]
        mean[1:-1, 1:-1] /= 5
        mean[0, 1:-1] /= 4
        mean[-1, 1:-1] /= 4
        mean[1:-1, 0] /= 4
        mean[1:-1, -1] /= 4
        mean[0, 0] /= 3
        mean[-1, -1] /= 3
        mean[0, -1] /= 3
        mean[-1, 0] /= 3
    else:
        print("Warning: Size of array not supported")
    return mean


@jit(nopython=True)
def Evidence(mu1, mu2, var1, var2):
    return np.exp(-((mu1 - mu2) ** 2) / (2 * (var1 + var2))) / np.sqrt(
        2 * np.pi * (var1 + var2)
    )


@jit(nopython=True)
def PSNR(recover, signal, L=255):
    MSE = np.sum((recover - signal) ** 2) / (recover.size)
    return 10 * np.log10((L) ** 2 / MSE)


class StreamSampler(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.micindex = 1
        self.speakerindex = 1
        self.micstream = self.open_mic_stream()
        self.speakerstream = self.open_speaker_stream()
        self.errorcount = 0
        self.buffer = np_rw_buffer.AudioFramingBuffer(8192  * 4, dtype=numpy.int16)

    def stop(self):
        self.micstream.close()
        self.speakerstream.close()

    def open_mic_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxInputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        print("Found an input: device %d - %s" % (i, devinfo["name"]))
                        device_index = i
                        self.micindex = device_index

        if device_index == None:
            print("No preferred input found; using default input device.")

        stream = self.pa.open(format=pyaudio.paInt16,
                              channels=2,
                              rate=48000,
                              input=True,
                              input_device_index=self.micindex,  # device_index,
                              frames_per_buffer=8192,
                              stream_callback=self.non_blocking_stream_read,
                              )

        return stream

    def open_speaker_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxOutputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        print("Found an output: device %d - %s" % (i, devinfo["name"]))
                        device_index = i
                        self.speakerindex = device_index

        if device_index == None:
            print("No preferred input found; using default input device.")

        stream = self.pa.open(format=pyaudio.paInt16,
                              channels=2,
                              rate=48000,
                              output=True,
                              output_device_index=self.speakerindex,
                              frames_per_buffer=8192,
                              stream_callback=self.non_blocking_stream_write,
                              )
        return stream

    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
            try:
                self.buffer.write(numpy.frombuffer(in_data, dtype=numpy.int16))
                return in_data, pyaudio.paContinue # self.buffer.write(inputbuffer), pyaudio.paContinue
            except:
                #self.buffer.clear()
                return in_data, pyaudio.paContinue

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
            try:
                return fabada1x(self.buffer.read(16384)), pyaudio.paContinue
            except:
                #self.buffer.clear()
                return in_data, pyaudio.paContinue

    def stream_start(self):
        self.micstream.start_stream()
        self.speakerstream.start_stream()
        while self.micstream.is_active():
            sleep(0.001)
        return

    def listen(self):
        self.stream_start()
    # root window

    # slider current value


if __name__ == "__main__":
    SS = StreamSampler()
    SS.listen()
