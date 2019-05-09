#!/usr/bin/env python3
import noisereduction as nr
import os

WAV_FILE = os.path.join(os.getcwd(), 'example/noisefunkguitare')
T_NOISE = 1

noised_audio = nr.Wiener(WAV_FILE, T_NOISE)
noised_audio.get_wiener()
