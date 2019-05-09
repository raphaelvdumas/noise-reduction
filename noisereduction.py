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
    Class made for wiener filtering based on the article "Improved Signal-to-Noise Ratio Estimation for Speech
    Enhancement".
    Created by Raphael Dumas.

    Reference :
        Cyril Plapous, Claude Marro, Pascal Scalart. Improved Signal-to-Noise Ratio Estimation for Speech
        Enhancement. IEEE Transactions on Audio, Speech and Language Processing, Institute of Electrical
        and Electronics Engineers, 2006.
    """

    CHANNELS, MAXIMUM, FILE_NAME = 0, 0, ''

    def __init__(self, WAV_FILE, T_NOISE):
        """
        Input :
            WAV_FILE : wave file to be denoised
            T_NOISE : float, Time in seconds /!\ Only works if stationnary noise is at the beginning of wavfile /!\
        """
        self.WAV_FILE, self.T_NOISE = WAV_FILE, T_NOISE
        self.wav2data()

        # Constants are defined here
        self.NFFT, self.SHIFT =  2**10, 0.25
        self.FRAME = int(0.02*self.FS) # Frame of 20 ms

        # Computes the offset and number of frames for overlapp - add method.
        self.OFFSET = int(self.SHIFT*self.FRAME)
        if Wiener.CHANNELS > 1:
            self.FRAMES = (self.x.shape[0] - self.FRAME) // self.OFFSET + 1
        else :
            self.FRAMES = (self.x.size - self.FRAME) // self.OFFSET + 1
        # Hanning window and its energy Ew
        self.WINDOW = sg.hann(self.FRAME)
        self.EW = np.sum(self.WINDOW)

        # Evaluating noise psd with n_noise
        self.N_NOISE = int(self.T_NOISE*self.FS)
        self.Sbb = self.welchs_periodogram()

    def wav2data(self):
        self.FS, self.x = wav.read(self.WAV_FILE + '.wav')
        Wiener.CHANNELS = self.x.shape[1]

    def wav(self, data):
        wav.write(self.WAV_FILE + Wiener.FILE_NAME + '.wav', self.FS, data/Wiener.MAXIMUM)

    @staticmethod
    def a_posteriori_gain(SNR):
        """
        Function that computes the a posteriori gain G of Wiener filtering.
            Input :
                SNR : 1D np.array, Signal to Noise Ratio
            Output :
                G : 1D np.array, gain G of Wiener filtering
        """
        G = (SNR - 1)/SNR
        return G

    @staticmethod
    def a_priori_gain(SNR):
        """
        Function that computes the a priori gain G of Wiener filtering.
            Input :
                SNR : 1D np.array, Signal to Noise Ratio
            Output :
                G : 1D np.array, gain G of Wiener filtering
        """
        G = SNR/(SNR + 1)
        return G

    def welchs_periodogram(self):
        """
        Estimation of the Power Spectral Density (Sbb) of the stationnary noise
        with Welch's periodogram given prior knowledge of n_noise points where
        speech is absent.
            Output :
                Sbb : 1D np.array, Power Spectral Density of stationnary noise
        """
        # Initialising Sbb
        Sbb = np.zeros(self.NFFT)
        # Number of frames used for the noise
        NOISE_FRAMES = (self.N_NOISE - self.FRAME) // self.OFFSET + 1
        for frame in range(NOISE_FRAMES):
            i_min, i_max = frame*self.OFFSET, frame*self.OFFSET + self.FRAME
            x_framed = fft(self.x[i_min:i_max, 0]*self.WINDOW, self.NFFT)
            Sbbtmp = np.abs(x_framed)**2
            Sbb = frame * Sbb / (frame + 1) + Sbbtmp / (frame + 1)
        return Sbb

    def get_wiener(self):
        """
        Function that returns the estimated speech signal using overlapp - add method
        by applying a Wiener Filter on each frame to the noised input signal.
            Output :
                s_est : 1D np.array, Estimated speech signal
        """
        Wiener.FILE_NAME = '_wiener'
        # Initialising estimated signal s_est
        s_est = np.zeros(self.x.shape)
        for channel in range(Wiener.CHANNELS):
            for frame in range(self.FRAMES):
                ############# Initialising Frame ###################################
                # Temporal framing with a Hanning window
                i_min, i_max = frame*self.OFFSET, frame*self.OFFSET + self.FRAME
                x_framed = self.x[i_min:i_max, channel]*self.WINDOW

                # Zero padding x_framed
                X_framed = fft(x_framed, self.NFFT)

                ############# Wiener Filter ########################################
                # Apply a priori wiener gains G to X_framed to get output S
                SNR_post = (np.abs(X_framed)**2/self.EW)/self.Sbb
                G = Wiener.a_priori_gain(SNR_post)
                S = X_framed * G

                ############# Temporal estimated Signal ############################
                # Estimated signals at each frame normalized by the shift value
                temp_s_est = np.real(ifft(S)) * self.SHIFT
                s_est[i_min:i_max, channel] += temp_s_est[:self.FRAME]  # Truncating zero padding
        Wiener.MAXIMUM = s_est.max()
        self.wav(s_est)
        #return s_est

    def get_wiener_two_step(self):
        """
        Function that returns the estimated speech signals using overlapp - add method
        by applying a Two Step Noise Reduction on each frame (s_est_tsnr) to the noised input signal (x).
            Output :
                s_est_tsnr, s_est_hrnr : 1D np.array, 1D np.array
        """
        Wiener.FILE_NAME = '_wiener_two_step'
        # Typical constant used to determine SNR_dd_prio
        beta = 0.98

        # Initialising output estimated signal
        s_est_tsnr = np.zeros(self.x.shape)

        # Initialising matrix to store previous values.
        # For readability purposes, -1 represents past frame values and 0 represents actual frame values.
        S = np.zeros((2, self.NFFT), dtype='cfloat')
        for channel in range(Wiener.CHANNELS):
            for frame in range(self.FRAMES):
                ############# Initialising Frame ###################################
                # Temporal framing with a Hanning window
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
                s_est_tsnr[i_min:i_max, channel] += temp_s_est_tsnr[:self.FRAME] # Truncating zero padding

                ############# Update ###############################################
                # Rolling matrix to update old values
                S = np.roll(S, 1, axis=0)
        Wiener.MAXIMUM = s_est_tsnr.max()
        self.wav(s_est_tsnr)
