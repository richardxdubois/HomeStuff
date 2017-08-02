from scipy.fftpack import fft
from scipy.io import wavfile  # get the api
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.backends.backend_pdf import PdfPages

## Command line arguments
parser = argparse.ArgumentParser(
    description='Perform fft on audio file')

##   The following are 'convenience options' which could also be specified in the filter string
parser.add_argument('-f', '--dataFile', default="", help="name of data file (default=%(default)s)")
parser.add_argument('-t', '--test', default="no", help="run test sine wave (default=%(default)s)")
parser.add_argument('-o', '--output', default="out.pdf", help="output file name (default=%(default)s)")
parser.add_argument('-i', '--infile', default="", help="input file name for list of wav files (default=%(default)s)")

args = parser.parse_args()

audio_file = args.dataFile
fn = audio_file.split('/')[-1]

fg = {}
axes = {}

if args.test != 'no':
    sampling_rate = 44100
    duration = 10
    y = np.sin(x * 2 * np.pi * 2000.)

else:

    with PdfPages(args.output) as pdf:

        fig = plt.figure('all')
        ax = fig.add_subplot(111)
        ax.set_ylabel('log')
        ax.axvline(260., color='r')
        ax.axvline(440., color='r')
        plt.xlabel('Frequency (Hz)')
        plt.suptitle(' Audio FFT ')

        plot_num = 0
        with open(args.infile) as f:
            for line in f:
                (who, avfile) = line.split()


                sampling_rate, data = wavfile.read(avfile)  # load the data
                duration = len(data) / sampling_rate
                a = data.T[0]  # this is a two channel soundtrack, I get the first track
                y = [(ele / 2 ** 8.) * 2 - 1 for ele in data]  # this is 8-bit track, b is now normalized on [-1,1)

                N = sampling_rate * duration
                T = 1. / sampling_rate

                x = np.linspace(0.0, N*T, N)
                yf = fft(y)  # calculate fourier transform (complex numbers list)

                xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

                f, Pxx_den = signal.periodogram(y,sampling_rate)

                ind_sort = np.argsort(Pxx_den)[::-1][:1000]

                max_freq = f[ind_sort[0]]
                second_freq = f[ind_sort[1]]
                third_freq =  f[ind_sort[2]]

                r12 = Pxx_den[ind_sort[1]]/Pxx_den[ind_sort[0]]
                r13 = Pxx_den[ind_sort[2]]/Pxx_den[ind_sort[0]]

                print max_freq, second_freq, third_freq, r12, r13

                plot_num += 1
                fg[who] = plt.figure(plot_num)
                axes[who] = fg[who].add_subplot(111)
                axes[who].set_yscale('log')
                axes[who].axvline(260., color='r')
                axes[who].axvline(440., color='r')
                plt.xlabel('Frequency (Hz)')
                plt.suptitle(' Audio FFT: ' + who)

                axes[who].hist(f,weights=Pxx_den, bins=np.arange(0.,1000.,20), histtype='step',label=who)

                plt.figure("all")
                ax.hist(f,weights=Pxx_den, bins=np.arange(0.,1000.,20), histtype='step',label=who)



        plt.legend()
        pdf.savefig()
        plt.close()