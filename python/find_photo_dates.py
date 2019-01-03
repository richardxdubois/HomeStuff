import numpy as np
import argparse
import os
import collections
import datetime
from operator import itemgetter

"""
Gather up Picasa-managed photos to make a date-sorted list of filenames with their creation dates. Picasa keeps the 
originals in a .picasaoriginals directory and rewrites the create dates in the main directory when a Save is done. 

This script makes a map of creation dates from the saved folder, and uses those instead of the saved files in the main 
directory.
"""

# Command line arguments
parser = argparse.ArgumentParser(
    description='Create csv file of trip photos with original dates')

parser.add_argument('-i', '--input', default=None, help="directory of Picasa photos")
parser.add_argument('-O', '--output', default="out.csv", help="output file name (default=%(default)s)")

args = parser.parse_args()

pix_dir = args.input.replace('\\', '')
pix_file_list = os.listdir(pix_dir)
pix_originals_dir = pix_dir + "/.picasaoriginals/"
output_csv = args.output

# make map of date stamps for the originals
pix_originals_list = os.listdir(pix_originals_dir)

orig_map = {}
print("files in " + pix_originals_dir + ": ", len(pix_originals_list))

for origs in pix_originals_list:
    if origs[0] == ".":
        continue

    fpath = pix_originals_dir + "/" + origs

    # use os.stat.st_birthtime to get creation date on a mac
    t_orig = os.stat(fpath).st_birthtime
    orig_map[origs] = datetime.datetime.fromtimestamp(t_orig).strftime('%Y-%m-%d %H:%M:%S')

print("files in " + pix_dir + ": ", len(pix_file_list))

full_list = collections.OrderedDict()

# make map of date stamps for the main directory (saved and untouched). Only use JPG's and ignore directories

for not_sorted in pix_file_list:
    if not_sorted[0] == ".":
        continue
    if "JPG" not in not_sorted.split(".")[1].upper():
        continue

    try:
        full_list[not_sorted] = orig_map[not_sorted]
    except KeyError:
        fpath = pix_dir + "/" + not_sorted
        t_not_sorted = os.stat(fpath).st_birthtime
        full_list[not_sorted] = datetime.datetime.fromtimestamp(t_not_sorted).strftime('%Y-%m-%d %H:%M:%S')

full_list_sorted = collections.OrderedDict(sorted(full_list.items(), key=itemgetter(1)))

# write out the csv file: image name and creation date.

out_csv = open(args.output, "w")

for pix in full_list_sorted:
    t = full_list_sorted[pix]

    out_str = pix.split(".")[0] + " , " + t + "\n"
    out_csv.write(out_str)

out_csv.close()
