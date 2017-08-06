import matplotlib.pyplot as plt
import datetime
import numpy as np
import argparse
from matplotlib.backends.backend_pdf import PdfPages

def parse_file(infile):

    timeList = []
    powerList = []

    with open(infile) as f:
        for line in f:
            (mtu, tstamp, powerStr, cost, volts) = line.split(',')
            if mtu == "mtu":
                continue

            power = -1. * float(powerStr)

            datestamp = datetime.datetime.strptime(tstamp, '%m/%d/%Y %H:%M:%S')
            timeList.append(datestamp)
            powerList.append(power)

    return timeList, np.array(powerList)


## Command line arguments
parser = argparse.ArgumentParser(
    description='Analyse TED solar data from 2017 eclipse')

##   The following are 'convenience options' which could also be specified in the filter string
parser.add_argument('-f', '--dataFile', default="", help="name of data file (default=%(default)s)")
parser.add_argument('-m', '--tmin', default="", help="start timestamp (default=%(default)s)")
parser.add_argument('-x', '--tmax', default="", help="end timestamp (default=%(default)s)")
parser.add_argument('-o', '--output', default="power.pdf", help="output file name (default=%(default)s)")
parser.add_argument('-i', '--infile', default="", help="input file name for list of power files (default=%(default)s)")

args = parser.parse_args()

with PdfPages(args.output) as pdf:

    powerDict = {}

    with open(args.infile) as f:

        for line in f:
            (period, fspec) = line.split()

            (t, p) = parse_file(fspec)
            powerDict[period] = p

    signal = powerDict['eclipse'] - (powerDict['before'] + powerDict['after'])/2.

    plt.figure(0)
    plt.plot(t,signal)
    plt.xticks(rotation=30)
    plt.tight_layout()

    pdf.savefig()
    plt.close()