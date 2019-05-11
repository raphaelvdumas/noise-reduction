#!/usr/bin/env python3
import noisereduction as nr
import os

WAV_FILE = os.getcwd() + '/example/noisefunkguitare'
T_NOISE = 0.8

noised_audio = nr.Wiener(WAV_FILE, T_NOISE)
noised_audio.wiener()
noised_audio.wiener_two_step()
