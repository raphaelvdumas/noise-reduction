#!/usr/bin/env python3
from scipy.fftpack import fft, ifft
import scipy.io.wavfile as wav
import scipy.signal as sg
import numpy as np

def halfwave_rectification(array):
    """
    Function that computes the half wave rectification with a threshold of 0.
        Input :
            array : 1D np.array, Temporal frame
        Output :
            halfwave : 1D np.array, Half wave temporal rectification
    """
    halfwave = np.zeros(array.size)
    halfwave[np.argwhere(array > 0)] = 1
    return halfwave


class NoisyAudio:

    def __init__(self, WAV_FILE):
        super(NoisyAudio, self).__init__()
        self.WAV_FILE = WAV_FILE
        self.FS, self.x = wav.read(self.WAV_FILE + '.wav')
        self.SHIFT, self.NFFT = 0.5, 2**10
        self.FRAME = int(0.02*self.FS)
        self.OFFSET = int(self.SHIFT*self.FRAME)
        self.WINDOW = sg.hann(self.FRAME)
        self.EW = np.sum(self.WINDOW)

        self.channels = range(self.x.shape[1]) if self.x.shape != (self.x.size,)  else range(1)
        length = self.x.shape[0] if len(self.channels) > 1 else self.x.size
        self.frames = range((length - self.FRAME) // self.OFFSET + 1)



    def generate_wav(self, FILE_NAME, FS, data):
        wav.write(self.WAV_FILE + FILE_NAME + '.wav', FS, data)

class NoiseEstimation(NoisyAudio):

    def __init__(self, TYPE):
        super(NoiseEstimation, self).__init__()

    def welchs_periodogram(self, T_NOISE):
        Sbb = np.zeros(self.NFFT)
        N_NOISE = int(T_NOISE*self.FS)
        NOISE_FRAMES = (N_NOISE - self.FRAME) // self.OFFSET + 1
        for frame in range(NOISE_FRAMES):
            i_min, i_max = frame*self.OFFSET, frame*self.OFFSET + self.FRAME
            x_framed = self.x[i_min:i_max, 0]*self.WINDOW
            X = fft(x_framed, self.NFFT)
            Sbb = frame * Sbb / (frame + 1) + np.abs(X)**2 / (frame + 1)
        return Sbb

class Wiener(NoisyAudio):
    """
    Class made for wiener filtering.

    """

    def __init__(self, WAV_FILE, T_NOISE):
        super(Wiener, self).__init__(WAV_FILE)
        self.Sbb = NoiseEstimation.welchs_periodogram(self, T_NOISE)
        self.s_est = np.zeros(self.x.shape)

    @staticmethod
    def a_posteriori_gain(SNR):
        G = (SNR - 1)/SNR
        return G

    @staticmethod
    def a_priori_gain(SNR):
        G = SNR/(SNR + 1)
        return G

    def wiener(self):
        for channel in self.channels:
            for frame in self.frames:
                i_min, i_max = frame*self.OFFSET, frame*self.OFFSET + self.FRAME
                x_framed = self.x[i_min:i_max, channel]*self.WINDOW
                X = fft(x_framed, self.NFFT)
                SNR_post = (np.abs(X)**2/self.EW)/self.Sbb
                G = Wiener.a_priori_gain(SNR_post)
                S = X * G
                temp_s_est = np.real(ifft(S)) * self.SHIFT
                self.s_est[i_min:i_max, channel] += temp_s_est[:self.FRAME]

        self.generate_wav('_wiener', self.FS, self.s_est/self.s_est.max())

    def wiener_two_step(self):
        beta = 0.98
        S = np.zeros((2, self.NFFT), dtype='cfloat')
        for channel in self.channels:
            for frame in self.frames:
                i_min, i_max = frame*self.OFFSET, frame*self.OFFSET + self.FRAME
                x_framed = self.x[i_min:i_max, channel]*self.WINDOW
                # Zero padding x_framed
                X_framed = fft(x_framed, self.NFFT)
                ############# Wiener Filter ########################################
                # Computation of spectral gain G using SNR a posteriori
                SNR_post = np.abs(X_framed)**2/self.EW/self.Sbb
                G = Wiener.a_priori_gain(SNR_post)
                S[0, :] = G * X_framed
                ############# Directed Decision ####################################
                # Computation of spectral gain G_dd using output S of Wiener Filter
                SNR_dd_prio = beta*np.abs(S[-1, :])**2/self.Sbb + (1 - beta)*halfwave_rectification(SNR_post - 1)
                G_dd = Wiener.a_priori_gain(SNR_dd_prio)
                S_dd = G_dd * X_framed
                ############# Two Step Noise Reduction #############################
                # Computation of spectral gain G_tsnr using output S_dd of Directed Decision
                SNR_tsnr_prio = np.abs(S_dd)**2/self.Sbb
                G_tsnr = Wiener.a_priori_gain(SNR_tsnr_prio)
                S_tsnr = G_tsnr * X_framed
                ############# Temporal estimated Signal ############################
                # Estimated signal at frame normalized by the shift value
                temp_s_est_tsnr = np.real(ifft(S_tsnr))*self.SHIFT
                self.s_est[i_min:i_max, channel] += temp_s_est_tsnr[:self.FRAME] # Truncating zero padding
                ############# Update ###############################################
                # Rolling matrix to update old values
                S = np.roll(S, 1, axis=0)

        self.generate_wav('_wiener_two_step', self.FS, self.s_est/self.s_est.max())
