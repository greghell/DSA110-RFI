#token = 'xoxb-508911196752-2019052043654-uADCWGbhAN5mmdJbSvKub1Ho';
# 0 9 * * * /home/user/anaconda3/envs/casa/bin/python /mnt/data/dsa110/T3/gregtest/rfianalysis/snap_plots/genplots.py

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

client = WebClient(token='xoxb-508911196752-2019052043654-uADCWGbhAN5mmdJbSvKub1Ho');
client.chat_postMessage(channel="#rfimitigation",text="daily report at "+datetime.now().strftime('%Y-%m-%d %H:%M:%S'));

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

today = datetime.now().strftime('%Y-%m-%d');

alldata, mask, alldataI, ants = proc_snap_data(dirs,nsnap);


# plot incoherent spectrum


ax = plot_incoh_spec(xfreq,alldata,today);
fname = '/mnt/data/dsa110/T3/gregtest/rfianalysis/snap_plots/incohspec_' + today + '.png';
plt.savefig(fname);
client.files_upload(channels='#rfimitigation',file=fname,initial_comment=fname.split('/')[-1]);


# plot : incoherent sum spectrogram


ax = plot_incoh_specgram(xfreq,times,alldataI,today);
fname = '/mnt/data/dsa110/T3/gregtest/rfianalysis/snap_plots/incohspecgram_' + today + '.png';
plt.savefig(fname);
client.files_upload(channels='#rfimitigation',file=fname,initial_comment=fname.split('/')[-1]);


# plot : power vs satellites passes


ax, pass_glo, pass_gal, pass_bei, pass_gps, tst = sat_sep_plot(dirs,alldataI,trigghist,today);
fname = '/mnt/data/dsa110/T3/gregtest/rfianalysis/snap_plots/sat_analysis_' + today + '.png';
plt.savefig(fname);
client.files_upload(channels='#rfimitigation',file=fname,initial_comment=fname.split('/')[-1]);


## sky plot of all satellites tracks


ax = plot_sat_map(pass_glo, pass_gal, pass_bei, pass_gps, today);
fname = '/mnt/data/dsa110/T3/gregtest/rfianalysis/snap_plots/sat_skyplot_' + today + '.png';
plt.savefig(fname);
client.files_upload(channels='#rfimitigation',file=fname,initial_comment=fname.split('/')[-1]);


## POWER vs NUMBER OF PLANES WITHIN 10 km


ax, dat, numplanes = plot_air_traffic(dirs,today);
fname = '/mnt/data/dsa110/T3/gregtest/rfianalysis/snap_plots/air_traffic_map_' + today + '.png';
plt.savefig(fname);
client.files_upload(channels='#rfimitigation',file=fname,initial_comment=fname.split('/')[-1]);


ax = air_traffic_ovro(alldataI,tst,numplanes,dat,today);
fname = '/mnt/data/dsa110/T3/gregtest/rfianalysis/snap_plots/air_traffic_near_ovro_' + today + '.png';
plt.savefig(fname);
client.files_upload(channels='#rfimitigation',file=fname,initial_comment=fname.split('/')[-1]);


## power vs freq

ax = antenna_power(nsnap,xfreq,alldata,ants,today);
fname = '/mnt/data/dsa110/T3/gregtest/rfianalysis/snap_plots/power_' + today + '.png';
plt.savefig(fname);
client.files_upload(channels='#rfimitigation',file=fname,initial_comment=fname.split('/')[-1]);


## first plot : occupancy vs freq


ax = antenna_occupancy(nsnap,xfreq,mask,ants,today);
fname = '/mnt/data/dsa110/T3/gregtest/rfianalysis/snap_plots/occupancy_' + today + '.png';
plt.savefig(fname);
client.files_upload(channels='#rfimitigation',file=fname,initial_comment=fname.split('/')[-1]);


# plot median mask


ax = median_flag_specgram(xfreq,times,mask,today);
fname = '/mnt/data/dsa110/T3/gregtest/rfianalysis/snap_plots/medmask_' + today + '.png';
plt.savefig(fname);
client.files_upload(channels='#rfimitigation',file=fname,initial_comment=fname.split('/')[-1]);

