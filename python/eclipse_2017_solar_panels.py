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
            (tdate, ttime) = tstamp.split()
            if mtu == "mtu":
                continue

            power = -1. * float(powerStr)

            datestamp = datetime.datetime.strptime(ttime, '%H:%M:%S')
            timeList.append(datestamp)
            powerList.append(power)

    return timeList, np.array(powerList)

def sum_power(t,p, min, max):

    sums = 0.
    for i in range(len(t)):
        if t[i] >= min and t[i] <= max:
            sums += p[i]

    return sums

## Command line arguments
parser = argparse.ArgumentParser(
    description='Analyse TED solar data from 2017 eclipse')

##   The following are 'convenience options' which could also be specified in the filter string
parser.add_argument('-f', '--dataFile', default="", help="name of data file (default=%(default)s)")
parser.add_argument('-m', '--tmin', default="09:01:00", help="start timestamp (default=%(default)s)")
parser.add_argument('-x', '--tmax', default="11:21:00", help="end timestamp (default=%(default)s)")
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

    normal_shape = (powerDict['before'] + powerDict['after'])/2.
    signal = powerDict['eclipse'] - normal_shape

    startStamp = datetime.datetime.strptime(args.tmin, '%H:%M:%S')
    endStamp = datetime.datetime.strptime(args.tmax, '%H:%M:%S')

    tdiff = (endStamp-startStamp).seconds/3600.
    sigSum = sum_power(t,signal, startStamp, endStamp)/60.
    normalPower = sum_power(t,normal_shape, startStamp, endStamp)
    normalSum = normalPower/60.

    frac_left = sigSum/normalSum

    print 'Fraction of power left in ', args.tmin, ' and ', args.tmax, ' : ', frac_left, ' for ', sigSum, normalSum, ' kW-hrs'

    plt.figure(0)
    plt.plot(t,signal,label='signal')
    plt.plot(t,normal_shape,label='normal')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.legend()

    pdf.savefig()
    plt.close()