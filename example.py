#!/usr/bin/env python3
import noisereduction as nr
import os

WAV_FILE = os.getcwd() + '/example/noisefunkguitare'
noise_begin, noise_end = 0, 1

noised_audio = nr.Wiener(WAV_FILE, noise_begin, noise_end)
noised_audio.wiener()
noised_audio.wiener_two_step()
