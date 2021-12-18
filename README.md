# Improve audio signals with noise reduction techniques

A simple example with the [Wiener filter](https://en.wikipedia.org/wiki/Wiener_filter) :
```python
#!/usr/bin/env python3
import noisereduction

audio = 'noised_audio' # name of wav file in current directory
noise_begin, noise_end = 0, 1 # window in second where only noise is present 

noised_audio = noisereduction.Wiener(audio, noise_begin, noise_end)
noised_audio.wiener() # Generates a cleaned wav file output of audio using Wiener filter
```
For a more advanced noise reduction technique, simply type :
```python
noised_audio.wiener_two_step() # Generates a cleaned wav file output of audio
```
Demonstration of the output wav files created can be found in the example directory using a noised guitar stereo audio signal.

**Work in progress :**
For now, this works pretty well with stationnary noise, but in the future the main goal will be to implement much more noise reduction techniques like causal wiener filtering in real time, Kalmann filtering, wavelets ...
And implement adaptive filtering algorithms treating the non-stationnary noise case. A long way to go !

*That's why feedback and help would be greatly appreciated !*

## Installation
noise-reduction runs with Python 3.7 and depends on [scipy](https://www.scipy.org/) and [numpy](https://www.numpy.org/) exclusively.

To install the libraries, clone this repository and in that directory execute:
```sh
python3 -m pip install -r requirements.txt
```
