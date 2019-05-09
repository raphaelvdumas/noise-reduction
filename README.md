# noise-reduction
**Made for reducing noise in audio signals**

```python
#!/usr/bin/env python3
import noisereduction as nr

audio = 'noised_audio' # wav file
t_noise = 1 # time of noise at the beginning of audio

noised_audio = nr.Wiener(audio, t_noise)
noised_audio.get_wiener() # Generates a cleaned wav file output of audio using Wiener filter
```

**Work in progress :** *Feedback and help would be greatly appreciated !*

# Installation
noise-reduction runs with Python 3.7 and depends on [scipy](https://www.scipy.org/) and [numpy](https://www.numpy.org/) exclusively.

To install the libraries, clone this repository and in that directory execute:
```sh
python3 -m pip install -r requirements.txt
```
s
# About this project
Created by [Raphaël Dumas](https://github.com/DumasRaphael) for a project first initiated at [Le Mans University](http://www.univ-lemans.fr/fr/index.html).
