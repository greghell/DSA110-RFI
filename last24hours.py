## /home/user/data/webPLOTS/rfi/last24hours/last24hours.py

import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np, matplotlib.pyplot as plt, os, pylab, glob
import scipy.signal
from scipy.stats.stats import pearsonr
from scipy import stats
from astropy.time import Time
from slack_sdk import WebClient
import os
import requests
import time
import pycurl
from io import BytesIO
import math
import ephem
import sys
sys.path.insert(0,'/mnt/data/dsa110/T3/gregtest/rfianalysis/snap_plots/')
from sat_analysis import *
from rfi_plotting import *
from opensky_api import OpenSkyApi
from mpl_toolkits.basemap import Basemap
from IPython import display
from datetime import datetime, timedelta
import mpu
import pytz
import csv
import dsautils.dsa_store as ds
mpl.rcParams['timezone'] = 'US/Pacific';

pat = '/home/user/vikram/PLOTS/DATA/';
nsnap = 17;
dirs = glob.glob(pat+'59*');
dirs.sort();
dirs = dirs[-96:]; # = 24 hours
ndirs = len(dirs);

times = [];
for k in range(ndirs):
    times.append(Time(float(dirs[k].split('/')[-1]),format='mjd').to_value('datetime'));

starttime = times[0];

times = mdates.date2num(times);

trigghist = get_triggers(starttime);

xfreq = 1530.-np.arange(1024)*250./1024.;

today = datetime.now().strftime('%Y-%m-%d %H:%M:%S');

alldata, mask, alldataI, ants = proc_snap_data(dirs,nsnap);


# plot incoherent spectrum


ax = plot_incoh_spec(xfreq,alldata,today);
fname = '/home/user/data/webPLOTS/rfi/last24hours/incohspec.png';
plt.savefig(fname);


# plot : incoherent sum spectrogram


ax = plot_incoh_specgram(xfreq,times,alldataI,today);
fname = '/home/user/data/webPLOTS/rfi/last24hours/incohspecgram.png';
plt.savefig(fname);


# plot : power vs satellites passes


ax, pass_glo, pass_gal, pass_bei, pass_gps, tst = sat_sep_plot(dirs,alldataI,trigghist,today);
fname = '/home/user/data/webPLOTS/rfi/last24hours/sat_analysis.png';
plt.savefig(fname);


## sky plot of all satellites tracks


ax = plot_sat_map(pass_glo, pass_gal, pass_bei, pass_gps, today);
fname = '/home/user/data/webPLOTS/rfi/last24hours/current_sat_skyplot.png';
plt.savefig(fname);


## POWER vs NUMBER OF PLANES WITHIN 10 km


ax, dat, numplanes = plot_air_traffic(dirs,today);
fname = '/home/user/data/webPLOTS/rfi/last24hours/air_traffic_map.png';
plt.savefig(fname);

ax = air_traffic_ovro(alldataI,tst,numplanes,dat,today);
fname = '/home/user/data/webPLOTS/rfi/last24hours/air_traffic_near_ovro.png';
plt.savefig(fname);


## first plot : occupancy vs freq


ax = antenna_occupancy(nsnap,xfreq,mask,ants,today);
fname = '/home/user/data/webPLOTS/rfi/last24hours/occupancy.png';
plt.savefig(fname);

# 2nd plot : power vs freq

ax = antenna_power(nsnap,xfreq,alldata,ants,today);
fname = '/home/user/data/webPLOTS/rfi/last24hours/power.png';
plt.savefig(fname);


# plot median mask


ax = median_flag_specgram(xfreq,times,mask,today);
fname = '/home/user/data/webPLOTS/rfi/last24hours/medmask.png';
plt.savefig(fname);

