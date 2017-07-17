from scipy.fftpack import fft
from scipy.io import wavfile  # get the api
import matplotlib.pyplot as plt
import numpy as np
import argparse

## Command line arguments
parser = argparse.ArgumentParser(
    description='Perform fft on audio file')

##   The following are 'convenience options' which could also be specified in the filter string
parser.add_argument('-f', '--dataFile', default="", help="name of data file (default=%(default)s)")
parser.add_argument('-t', '--test', default="no", help="run test sine wave (default=%(default)s)")

args = parser.parse_args()

audio_file = args.dataFile

if args.test != 'no':
    sampling_rate = 44100
    duration = 10
    y = np.sin(x * 2 * np.pi * 2000.)

else:
    sampling_rate, data = wavfile.read(audio_file)  # load the data
    duration = len(data) / sampling_rate
    a = data.T[0]  # this is a two channel soundtrack, I get the first track
    y = [(ele / 2 ** 8.) * 2 - 1 for ele in data]  # this is 8-bit track, b is now normalized on [-1,1)

N = sampling_rate * duration
T = 1. / sampling_rate

x = np.linspace(0.0, N*T, N)
yf = fft(y)  # calculate fourier transform (complex numbers list)


xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.xlim(0,3000.)
ax.set_yscale('log')
plt.show()
