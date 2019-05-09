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


class Wiener:
    """
    Class made for wiener filtering.

    """

    CHANNELS, MAXIMUM, FILE_NAME = 0, 0, ''

    def __init__(self, WAV_FILE, T_NOISE):
        self.WAV_FILE, self.T_NOISE = WAV_FILE, T_NOISE
        self.FS, self.x = wav.read(self.WAV_FILE + '.wav')

        Wiener.CHANNELS = self.x.shape[1] if self.x.shape != (self.x.size,)  else 1

        self.NFFT, self.SHIFT =  2**10, 0.25
        self.FRAME = int(0.02*self.FS)

        self.OFFSET = int(self.SHIFT*self.FRAME)
        length = self.x.shape[0] if Wiener.CHANNELS > 1 else self.x.size
        self.FRAMES = (length - self.FRAME) // self.OFFSET + 1

        self.WINDOW = sg.hann(self.FRAME)
        self.EW = np.sum(self.WINDOW)
        self.noise_estimation(self.T_NOISE)
        self.s_est = np.zeros(self.x.shape)

    def noise_estimation(self, T_NOISE):
        self.Sbb = np.zeros(self.NFFT)
        N_NOISE = int(self.T_NOISE*self.FS)
        NOISE_FRAMES = (N_NOISE - self.FRAME) // self.OFFSET + 1
        for frame in range(NOISE_FRAMES):
            X = self._get_framed(0, frame)
            self.Sbb = frame * self.Sbb / (frame + 1) + np.abs(X)**2 / (frame + 1)

    def _get_framed(self, channel, frame):
        i_min, i_max = frame*self.OFFSET, frame*self.OFFSET + self.FRAME
        x_framed = self.x[i_min:i_max, channel]*self.WINDOW
        X = fft(x_framed, self.NFFT)
        return X

    def _get_wiener_output(self, X):
        SNR_post = (np.abs(X)**2/self.EW)/self.Sbb
        G = Wiener._a_priori_gain(SNR_post)
        S = X * G
        return S

    def _get_estimation(self, channel, frame, S):
        i_min, i_max = frame*self.OFFSET, frame*self.OFFSET + self.FRAME
        temp_s_est = np.real(ifft(S)) * self.SHIFT
        self.s_est[i_min:i_max, channel] += temp_s_est[:self.FRAME]

    def wiener(self):
        for channel in range(Wiener.CHANNELS):
            for frame in range(self.FRAMES):
                X = self._get_framed(channel, frame)
                S = self._get_wiener_output(X)
                self._get_estimation(channel, frame, S)
        Wiener.FILE_NAME, Wiener.MAXIMUM = '_wiener', self.s_est.max()
        self._generate_wav(self.s_est)

    def wiener_two_step(self):
        beta = 0.98
        S = np.zeros((2, self.NFFT), dtype='cfloat')
        for channel in range(Wiener.CHANNELS):
            for frame in range(self.FRAMES):

                i_min, i_max = frame*self.OFFSET, frame*self.OFFSET + self.FRAME
                x_framed = self.x[i_min:i_max, channel]*self.WINDOW

                # Zero padding x_framed
                X_framed = fft(x_framed, self.NFFT)

                ############# Wiener Filter ########################################
                # Computation of spectral gain G using SNR a posteriori
                SNR_post = np.abs(X_framed)**2/self.EW/self.Sbb
                G = Wiener._a_priori_gain(SNR_post)
                S[0, :] = G * X_framed

                ############# Directed Decision ####################################
                # Computation of spectral gain G_dd using output S of Wiener Filter
                SNR_dd_prio = beta*np.abs(S[-1, :])**2/self.Sbb + (1 - beta)*halfwave_rectification(SNR_post - 1)
                G_dd = Wiener._a_priori_gain(SNR_dd_prio)
                S_dd = G_dd * X_framed

                ############# Two Step Noise Reduction #############################
                # Computation of spectral gain G_tsnr using output S_dd of Directed Decision
                SNR_tsnr_prio = np.abs(S_dd)**2/self.Sbb
                G_tsnr = Wiener._a_priori_gain(SNR_tsnr_prio)
                S_tsnr = G_tsnr * X_framed

                ############# Temporal estimated Signal ############################
                # Estimated signal at frame normalized by the shift value
                temp_s_est_tsnr = np.real(ifft(S_tsnr))*self.SHIFT
                self.s_est[i_min:i_max, channel] += temp_s_est_tsnr[:self.FRAME] # Truncating zero padding

                ############# Update ###############################################
                # Rolling matrix to update old values
                S = np.roll(S, 1, axis=0)
        Wiener.FILE_NAME, Wiener.MAXIMUM = '_wiener_two_step', self.s_est.max()
        self._generate_wav(self.s_est)


    def _generate_wav(self, data):
        wav.write(self.WAV_FILE + Wiener.FILE_NAME + '.wav', self.FS, data/Wiener.MAXIMUM)

    @staticmethod
    def _a_posteriori_gain(SNR):
        G = (SNR - 1)/SNR
        return G

    @staticmethod
    def _a_priori_gain(SNR):
        G = SNR/(SNR + 1)
        return G
