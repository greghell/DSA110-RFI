## set of functions to plot and analyze RFI through SNAP data

## VERIFY TST AND TIMES ARE SAME THINGS


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
from opensky_api import OpenSkyApi
from mpl_toolkits.basemap import Basemap
from IPython import display
from datetime import datetime, timedelta, timezone
import mpu
import pytz
import csv
import requests
import dsautils.dsa_store as ds
import pandas as pd
mpl.rcParams['timezone'] = 'US/Pacific';


## flagger


def clip(spec,filtsize,thres):
    flag = np.zeros(np.shape(spec))
    bl = scipy.signal.medfilt(spec,kernel_size=filtsize);
    speccorrec = spec - bl;
    specstd = stats.median_absolute_deviation(speccorrec);
    flag[np.argwhere(speccorrec > thres*specstd)] = 1.;
    flag[np.argwhere(speccorrec < -thres*specstd)] = 1.;
    return flag;


## get triggers in the past


def get_triggers(starttime):
    ## extract triggers time stamps
    triggdirs = glob.glob('/mnt/data/dsa110/T2/*/');
    triggdirs.sort(key=os.path.getctime);
    triggfiles = ['/mnt/data/dsa110/T2/cluster_output.csv'];

    for k in range(len(triggdirs)):
        if datetime.fromtimestamp(os.path.getctime(triggdirs[-k])) > starttime:
            if os.path.isfile(triggdirs[-k]+'cluster_output.csv'):
                triggfiles.append(triggdirs[-k]+'cluster_output.csv');

    mjds = [];
    for fil in triggfiles:
        csv_data = pd.read_csv(fil);
        try:
            mjds.append(Time(np.asarray(csv_data['mjds']),format='mjd').to_value('datetime'));
        except:
            pass;
    mjds = [item for sublist in mjds for item in sublist];
    mjds = mdates.date2num(mjds);
    trigghist = np.histogram(mjds,1000);
    return trigghist;


## read SNAP data and flag RFI


def proc_snap_data(dirs,nsnap):
# nsnap can be estimated
    ndirs = len(dirs);
    mask = np.zeros((nsnap*3,2,ndirs,1024));
    alldata = np.zeros((nsnap*3,2,ndirs,1024));

    for k in range(ndirs):
        if(len(os.listdir(dirs[k])) != 0):
            data = [];
            ants = [];
            for snap in np.arange(1,nsnap+1):

                fl = dirs[k] + '/data/snap'+str(snap)+'.npz';
                d = np.load(fl,fix_imports=True)

                ants.append(str(d['a1']))
                ants.append(str(d['a2']))
                ants.append(str(d['a3']))

                for i in range(6):
                    data.append(d['specs'][i])

            for i in range(len(ants)):
                y1 = np.abs(data[2*i]);
                y2 = np.abs(data[2*i+1]);
                alldata[i,0,k,:] = y1;
                alldata[i,1,k,:] = y2;
                mask[i,0,k,:] = clip(y1,21,4);
                mask[i,1,k,:] = clip(y2,21,4);

    alldataI = np.sum(alldata,axis=1);
    alldataI = np.sum(alldataI,axis=0);
    
    return alldata, mask, alldataI, ants;


## plot incoherent spectrum


def plot_incoh_spec(xfreq,alldata,today):
    fig, ax = plt.subplots(figsize=(16,5));
    plt.plot(xfreq,np.mean(np.mean(alldata[:,0,:,:],axis=0),axis=0),label='Pol X');
    plt.plot(xfreq,np.mean(np.mean(alldata[:,1,:,:],axis=0),axis=0),label='Pol Y');
    plt.legend();
    plt.grid();
    plt.xlabel('frequency [MHz]');
    plt.ylabel('power [linear]');
    plt.title(today + ' -- incoherent spectrum');
    plt.tight_layout();
    return ax;


## incoherent sum spectrogram


def plot_incoh_specgram(xfreq,times,alldataI,today,dirs):
    from matplotlib.ticker import FormatStrFormatter
    fig, ax = plt.subplots(figsize=(16,14));
    tt = ax.imshow(np.flip(10.*np.log10(alldataI),axis=1),aspect='auto',extent=[xfreq[-1],xfreq[0],times[-1],times[0]],interpolation='None');
    ax.yaxis_date();
    date_format = mdates.DateFormatter('%m-%d %H:%M:%S');
    ax.yaxis.set_major_formatter(date_format);
    #fig.autofmt_xdate();
    plt.yticks(rotation = 45);
    ax2 = ax.twinx();
    ax2.set_ybound([float(dirs[-1].split('/')[-1]),float(dirs[0].split('/')[-1])]);
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'));
    ax2.invert_yaxis();
    ax2.set_ylabel('MJD');
    plt.xlabel('frequency [MHz]');
    plt.title(today + ' -- incoherent sum - total intensity [dB]');
    plt.colorbar(tt);
    plt.tight_layout();
    return ax;


## plot individual spectra


def plot_four_spec(xfreq,dirs,alldataI,today):
    fig,ax = plt.subplots(alldataI.shape[0],1,figsize=(18,10));
    for k in range(alldataI.shape[0]):
        ax[k].plot(xfreq,alldataI[k,:]);
        ax[k].set(title=Time(float(dirs[k].split('/')[-1]),format='mjd').to_value('datetime').replace(tzinfo=timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S'),ylabel='power [linear]',xlim=[np.min(xfreq),np.max(xfreq)]);
        ax[k].grid();
    ax[k].set(xlabel='frequency [MHz]');
    plt.tight_layout();
    return ax;


## power vs satellites passes


def sat_sep_plot(dirs,alldataI,trigghist,today):

    tst = [];
    ndirs = len(dirs);
    for k in range(ndirs):
        tst.append(Time(float(dirs[k].split('/')[-1]),format='mjd').to_value('datetime'));

    pass_glo, timdtm = sat_analysis(dirs,'GLONASS');
    pass_gal, timdtm = sat_analysis(dirs,'GALILEO');
    pass_bei, timdtm = sat_analysis(dirs,'BEIDOU');
    pass_gps, timdtm = sat_analysis(dirs,'GPS');
    mintime = np.min(mpl.dates.date2num(timdtm));
    maxtime = np.max(mpl.dates.date2num(timdtm));
    fig,ax = plt.subplots(8,1,figsize=(18,22));
    if ndirs < 5.:
        ax[0].plot_date(mpl.dates.date2num(tst),np.mean(alldataI[:,980:],axis=1),'*-',markersize=10);
    else:
        ax[0].plot_date(mpl.dates.date2num(tst),np.mean(alldataI[:,980:],axis=1),'-');
    ax[0].set(ylabel='received power 1.28-1.29 GHz',xlim=[mintime,maxtime],ylim=[0.,40000.]);
    ax[0].grid();
    if ndirs < 5.:
        ax[1].plot_date(mpl.dates.date2num(tst),np.mean(alldataI[:,82:205],axis=1),'*-',markersize=10);
    else:
        ax[1].plot_date(mpl.dates.date2num(tst),np.mean(alldataI[:,82:205],axis=1),'-');
    ax[1].set(ylabel='received power 1.48-1.51 GHz',xlim=[mintime,maxtime],ylim=[0.,10000.]);
    ax[1].grid();
    if ndirs < 5.:
        ax[2].plot_date(mpl.dates.date2num(tst),np.mean(alldataI[:,600:620],axis=1),'*-',markersize=10);
    else:
        ax[2].plot_date(mpl.dates.date2num(tst),np.mean(alldataI[:,600:620],axis=1),'-');
    ax[2].set(ylabel='received power 1378-1383 MHz (GPS L3)',xlim=[mintime,maxtime],ylim=[0.,10000.]);
    ax[2].grid();
    ax[3].plot_date(trigghist[1][:-1],trigghist[0],'.');
    ax[3].set(ylabel='num of trigger per '+"{:.2f}".format((trigghist[1][1]-trigghist[1][0])*24.*60.*60.)+'s',xlim=[mintime,maxtime]);
    ax[3].grid();
    ax[4].plot_date(mpl.dates.date2num(timdtm),pass_glo[0,:,:],'-');
    ax[4].axvspan(mpl.dates.date2num(tst[-1]), maxtime, facecolor='r', alpha=0.2);
    ax[4].set(ylabel='GLONASS sep [deg]',xlim=[mintime,maxtime],ylim=[0,8]);
    ax[4].grid();
    ax[5].plot_date(mpl.dates.date2num(timdtm),pass_gal[0,:,:],'-');
    ax[5].axvspan(mpl.dates.date2num(tst[-1]), maxtime, facecolor='r', alpha=0.2);
    ax[5].set(ylabel='GALILEO sep [deg]',xlim=[mintime,maxtime],ylim=[0,8]);
    ax[5].grid();
    ax[6].plot_date(mpl.dates.date2num(timdtm),pass_bei[0,:,:],'-');
    ax[6].axvspan(mpl.dates.date2num(tst[-1]), maxtime, facecolor='r', alpha=0.2);
    ax[6].set(ylabel='BEIDOU sep [deg]',xlim=[mintime,maxtime],ylim=[0,8]);
    ax[6].grid();
    ax[7].plot_date(mpl.dates.date2num(timdtm),pass_gps[0,:,:],'-');
    ax[7].axvspan(mpl.dates.date2num(tst[-1]), maxtime, facecolor='r', alpha=0.2);
    ax[7].set(ylabel='GPS sep [deg]',xlabel='date-time',xlim=[mintime,maxtime],ylim=[0,8]);
    ax[7].grid();
    plt.tight_layout();
    plt.suptitle(today + ' -- Satellite separation vs. power variation + 1 hour prediction');
    return ax, pass_glo, pass_gal, pass_bei, pass_gps, tst;


## sky plot of all satellites tracks


def plot_sat_map(pass_glo, pass_gal, pass_bei, pass_gps, today):
    # read mean antenna elevation from etcd
    my_ds = ds.DsaStore();
    tmp = [];
    for k in range(110):
         vv = my_ds.get_dict('/mon/ant/'+str(k+1));
         if vv != None and 'ant_el' in vv:
             tmp.append(vv['ant_el']);
    obs_el = np.mean(tmp);
    #obs_el = 107.65;

    fig,ax = plt.subplots(2,2,figsize=(14,14),subplot_kw={'projection': 'polar'});

#    plt.subplot(221, polar=True);
    for nsat in range(pass_glo.shape[2]):
        ax[0,0].plot(np.radians(pass_glo[1, pass_glo[2,:,nsat]>0. ,nsat]),90.-(pass_glo[2, pass_glo[2,:,nsat]>0. ,nsat]),'.',markersize=0.1);
    if 90.-obs_el < 0.:
        ax[0,0].plot(0.,np.abs(90.-obs_el),'r*',markersize=5.);
    else:
        ax[0,0].plot(np.pi,90.-obs_el,'r*',markersize=5.);
    ax[0,0].set_theta_zero_location("N");
    ax[0,0].set(title='GLONASS');

    for nsat in range(pass_gal.shape[2]):
        ax[0,1].plot(np.radians(pass_gal[1, pass_gal[2,:,nsat]>0. ,nsat]),90.-(pass_gal[2, pass_gal[2,:,nsat]>0. ,nsat]),'.',markersize=0.1);
    if 90.-obs_el < 0.:
        ax[0,1].plot(0.,np.abs(90.-obs_el),'r*',markersize=5.);
    else:
        ax[0,1].plot(np.pi,90.-obs_el,'r*',markersize=5.);
    ax[0,1].set_theta_zero_location("N");
    ax[0,1].set(title='GALILEO');
    
    for nsat in range(pass_bei.shape[2]):
        ax[1,0].plot(np.radians(pass_bei[1, pass_bei[2,:,nsat]>0. ,nsat]),90.-(pass_bei[2, pass_bei[2,:,nsat]>0. ,nsat]),'.',markersize=0.1);
    if 90.-obs_el < 0.:
        ax[1,0].plot(0.,np.abs(90.-obs_el),'r*',markersize=5.);
    else:
        ax[1,0].plot(np.pi,90.-obs_el,'r*',markersize=5.);
    ax[1,0].set_theta_zero_location("N");
    ax[1,0].set(title='BEIDOU');
    
    for nsat in range(pass_gps.shape[2]):
        ax[1,1].plot(np.radians(pass_gps[1, pass_gps[2,:,nsat]>0. ,nsat]),90.-(pass_gps[2, pass_gps[2,:,nsat]>0. ,nsat]),'.',markersize=0.1);
    if 90.-obs_el < 0.:
        ax[1,1].plot(0.,np.abs(90.-obs_el),'r*',markersize=5.);
    else:
        ax[1,1].plot(np.pi,90.-obs_el,'r*',markersize=5.);
    ax[1,1].set_theta_zero_location("N");
    ax[1,1].set(title='GPS');
    plt.tight_layout();
    plt.suptitle(today + ' -- Satellite constellations skyplot');
    return ax;


## air traffic map


def plot_air_traffic(dirs,today):
    
    ndirs = len(dirs);

    patAT = '/home/user/T3_detect/satellite/flightdata/';

    dirsAT = glob.glob(patAT+'*.npy');

    rightnow = datetime.now();

    # remove oldest files (> 25h old)
    for fil in dirsAT:
        datetime_object = datetime.strptime(fil.split('/')[-1].strip('.npy'), '%Y-%m-%d_%H:%M:%S')
        if datetime_object < rightnow - timedelta(hours=25.):
            os.remove(fil);

    if ndirs < 5:
        filsonehour = [];
        for fil in dirsAT:
            datetime_object = datetime.strptime(fil.split('/')[-1].strip('.npy'), '%Y-%m-%d_%H:%M:%S')
            if datetime_object >= rightnow - timedelta(hours=1.):
                filsonehour.append(fil);
                
    dirsAT = glob.glob(patAT+'*.npy');
    dirsAT = np.sort(dirsAT);
    ndirsAT = len(dirsAT);

    if ndirs < 5:
        dirsAT = np.sort(filsonehour);
        ndirsAT = len(dirsAT);
    
    fig, ax = plt.subplots(figsize=(16,12));
    fig_size = plt.rcParams["figure.figsize"];
    fig_size[0] = 10;
    fig_size[1] = 10;
    plt.rcParams["figure.figsize"] = fig_size;
    m = Basemap(projection = 'mill', llcrnrlat = 35.238, urcrnrlat = 38.665, llcrnrlon = -121.036, urcrnrlon = -114.482, resolution = 'h');
    m.drawcoastlines();
    m.drawmapboundary(fill_color='white');
    m.drawstates();
    m.drawrivers(color='aqua');
    
    dat = [];    #day/time
    numplanes = np.zeros((ndirsAT));    # number of planes within 10km from OVRO
    allda = np.empty((0,3));
    for k in range(ndirsAT):
        tmp = np.load(dirsAT[k],allow_pickle=True);
        dat.append(datetime.strptime(dirsAT[k].split('/')[-1].strip('.npy'), '%Y-%m-%d_%H:%M:%S').astimezone(pytz.timezone("America/Los_Angeles")));
        try:
            da = np.zeros((len(tmp),3));
            for kk in range(len(tmp)):
                da[kk,:] = [tmp[kk].longitude, tmp[kk].latitude, tmp[kk].altitude];
                if mpu.haversine_distance((37.233370, -118.283443), (tmp[kk].latitude, tmp[kk].longitude)) < 10.:
                    numplanes[k] += 1;
            allda = np.append(allda,da,0);
        except:
            pass;
    x, y = m(allda[:,0], allda[:,1]);

    sc = plt.scatter(x, y, s = 2, c=allda[:,2]);#'#EEB011');#, alpha=0.1+(k+1)/ndirsAT*0.9);

    x, y = m(-118.283443,37.233370);    # OVRO
    plt.scatter(x, y, s = 15, c='r');
    plt.annotate('OVRO',(x,y));
    x, y = m(-118.4066103,37.3665175);  # Bishop
    plt.scatter(x, y, s = 15, c='k');
    plt.annotate('Bishop',(x,y));
    x, y = m(-119.9346448,36.7857263);  # Fresno
    plt.scatter(x, y, s = 15, c='k');
    plt.annotate('Fresno',(x,y));
    x, y = m(-119.1992671,35.3494663);  # Bakersfield
    plt.scatter(x, y, s = 15, c='k');
    plt.annotate('Bakersfield',(x,y));
    x, y = m(-115.3150834,36.1251958);  # Las Vegas
    plt.scatter(x, y, s = 15, c='k');
    plt.annotate('Las Vegas',(x,y));

    # dat = [];    #day/time
    # numplanes = np.zeros((ndirsAT));    # number of planes within 30km from OVRO

    # for k in range(ndirsAT):
        # tmp = np.load(dirsAT[k]);
        # x, y = m(tmp['lon'], tmp['lat']);
        # plt.scatter(x, y, s = 3, c='#EEB011', alpha=0.1+(k+1)/ndirsAT*0.9);
        # dat.append(datetime.strptime(dirsAT[k].split('/')[-1].strip('.npz'), '%Y-%m-%d_%H:%M:%S').astimezone(pytz.timezone("America/Los_Angeles")));
        # for kk in range(len(tmp['lon'])):
            # if mpu.haversine_distance((37.233370, -118.283443), (tmp['lat'][kk], tmp['lon'][kk])) < 30.:
                # numplanes[k] += 1;

    plt.title(today + ' -- air traffic over OVRO');
    cbar = plt.colorbar(sc);
    cbar.ax.set_ylabel('altitude [ft]', rotation=270);
    plt.tight_layout();
    return ax, dat, numplanes;


## air traffic near OVRO (# of planes within 30 km)


def air_traffic_ovro(alldataI,tst,numplanes,dat,today):
    fig, ax = plt.subplots(2,1,figsize=(16,8));
    if len(tst) < 5.:
        ax[0].plot_date(mpl.dates.date2num(tst),np.mean(alldataI,axis=1),'*-',markersize=10);
    else:
        ax[0].plot_date(mpl.dates.date2num(tst),np.mean(alldataI,axis=1),'-');
    date_format = mdates.DateFormatter('%m-%d %H:%M');
    ax[0].xaxis.set_major_formatter(date_format);
    ax[0].tick_params(labelrotation=45);
    if dat != []:
        ax[0].set(ylabel='received power',xlim=[np.min((np.min(mpl.dates.date2num(tst)), np.min(mpl.dates.date2num(dat)))),np.max((np.max(mpl.dates.date2num(tst)), np.max(mpl.dates.date2num(dat))))],ylim=[0.,12000.]);
    else:
        ax[0].set(ylabel='received power',xlim=[np.min(mpl.dates.date2num(tst)),np.max(mpl.dates.date2num(tst))],ylim=[0.,12000.]);
    ax[0].grid();
    ax[1].plot_date(mpl.dates.date2num(dat), numplanes,'*');
    date_format = mdates.DateFormatter('%m-%d %H:%M');
    ax[1].tick_params(labelrotation=45);
    ax[1].xaxis.set_major_formatter(date_format);
    ax[1].tick_params(labelrotation=45);
    ax[1].grid();
    if dat != []:
        ax[1].set(ylabel='# of planes 10 km from OVRO',xlabel='date / time', xlim=[np.min((np.min(mpl.dates.date2num(tst)), np.min(mpl.dates.date2num(dat)))),np.max((np.max(mpl.dates.date2num(tst)), np.max(mpl.dates.date2num(dat))))]);
    else:
        ax[1].set(ylabel='# of planes 10 km from OVRO',xlabel='date / time', xlim=[np.min(mpl.dates.date2num(tst)),np.max(mpl.dates.date2num(tst))]);
    plt.tight_layout();
    plt.suptitle(today);
    return ax;

def air_traffic_skyplot(dirs,today):
    
    ndirs = len(dirs);

    patAT = '/home/user/T3_detect/satellite/flightdata/';

    dirsAT = glob.glob(patAT+'*.npy');

    rightnow = datetime.now();

    # remove oldest files (> 25h old)
    for fil in dirsAT:
        datetime_object = datetime.strptime(fil.split('/')[-1].strip('.npy'), '%Y-%m-%d_%H:%M:%S')
        if datetime_object < rightnow - timedelta(hours=25.):
            os.remove(fil);

    if ndirs < 5:
        filsonehour = [];
        for fil in dirsAT:
            datetime_object = datetime.strptime(fil.split('/')[-1].strip('.npy'), '%Y-%m-%d_%H:%M:%S')
            if datetime_object >= rightnow - timedelta(hours=1.):
                filsonehour.append(fil);

    dirsAT = glob.glob(patAT+'*.npy');
    dirsAT = np.sort(dirsAT);
    ndirsAT = len(dirsAT);

    if ndirs < 5:
        dirsAT = np.sort(filsonehour);
        ndirsAT = len(dirsAT);
    
    allda = np.empty((0,3));
    for k in range(ndirsAT):
        tmp = np.load(dirsAT[k],allow_pickle=True);
        try:
            da = np.zeros((len(tmp),3));
            for kk in range(len(tmp)):
                da[kk,:] = [tmp[kk].longitude, tmp[kk].latitude, tmp[kk].altitude];
            allda = np.append(allda,da,0);
        except:
            pass;
    
    angle = np.zeros((len(allda)));
    r = np.zeros((len(allda)));
    
    ovro = np.asarray([-118.4066103,37.3665175]);
    norvec = [0,1];
    
    for k in range(len(allda)):
        fl = np.asarray([allda[k,0],allda[k,1]]);
        tofl = fl-ovro;

        unit_vector_1 = tofl / np.linalg.norm(tofl)
        unit_vector_2 = norvec / np.linalg.norm(norvec)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle[k] = np.arccos(dot_product);
        if tofl[0] > 0:
            angle[k] = -angle[k];
        di = mpu.haversine_distance(np.roll(ovro,1),np.roll(fl,1)) * 1000.; # proj distance on earth in m
        r[k] = 90.-np.degrees(np.arctan(allda[k,2] / di));
    
    ax = plt.subplot(111, polar=True);
    # for kk in range(ndirsAT):
        # fdat = np.load(dirsAT[kk],allow_pickle=True);
        # altvec = fdat['alt'];
        # altvec[np.where(altvec==None)[0]] = 0;

        # for k in range(len(fdat['lat'])):
            # fl = np.asarray([fdat['lon'][k],fdat['lat'][k]]);
            # tofl = fl-ovro;

            # unit_vector_1 = tofl / np.linalg.norm(tofl)
            # unit_vector_2 = norvec / np.linalg.norm(norvec)
            # dot_product = np.dot(unit_vector_1, unit_vector_2)
            # angle = np.arccos(dot_product);
            # if tofl[0] > 0:
                # angle = -angle;
            # di = mpu.haversine_distance(np.roll(ovro,1),np.roll(fl,1)) * 1000.; # proj distance on earth in m
            # r = 90.-np.degrees(np.arctan(altvec[k] / di));
            # ax.plot(angle,r,'y.',markersize=3.,alpha = 0.1+(kk+1)/ndirsAT*0.9);
    ax.plot(angle,r,'y.',markersize=3.);
    ax.set_theta_zero_location("N");
    ax.set_rmax(90);
    ax.grid(True);
    ax.set_xticklabels(['N', '', 'W', '', 'S', '', 'E', '']);
    plt.tight_layout();
    plt.suptitle(today);
    return ax;


## occupancy vs freq for each antenna


def antenna_occupancy(nsnap,xfreq,mask,ants,today):
    fig,ax = plt.subplots(int(np.ceil(nsnap*3/6)),6,figsize=(16,int(np.ceil(nsnap*3/6))*2))
    for i in range(nsnap*3):
        ax[int(np.floor(i/6))][i%6].plot(xfreq,np.mean(mask[i,0,:,:],axis=0)*100.,label='B')
        ax[int(np.floor(i/6))][i%6].plot(xfreq,np.mean(mask[i,1,:,:],axis=0)*100.,label='A',color='black',alpha=0.6);
        ax[int(np.floor(i/6))][i%6].set(title='Antenna '+ants[i], xlim = [1280.,1530.] );
        ax[int(np.floor(i/6))][i%6].grid();
        if i>nsnap*3-7:
            ax[int(np.floor(i/6))][i%6].set(xlabel = 'Frequency (MHz)');
        else:
            ax[int(np.floor(i/6))][i%6].set(xticks=[]);
    for i in range(nsnap*3,int(np.ceil(nsnap*3/6))*6):
        ax[int(np.floor(i/6))][i%6].set_axis_off();
    plt.suptitle(today + ' -- Spectral occupancy per antenna');
    plt.tight_layout();
    return ax;


## power vs freq for each antenna


def antenna_power(nsnap,xfreq,alldata,ants,today):
    fig,ax = plt.subplots(int(np.ceil(nsnap*3/6)),6,figsize=(16,int(np.ceil(nsnap*3/6))*2))
    for i in range(nsnap*3):
        ax[int(np.floor(i/6))][i%6].plot(xfreq,np.mean(alldata[i,0,:,:],axis=0),label='B')
        ax[int(np.floor(i/6))][i%6].plot(xfreq,np.mean(alldata[i,1,:,:],axis=0),label='A',color='black',alpha=0.6);
        ax[int(np.floor(i/6))][i%6].set(title='Antenna '+ants[i], xlim = [1280.,1530.], ylim = [20,120]);
        ax[int(np.floor(i/6))][i%6].grid();
        if i>nsnap*3-7:
            ax[int(np.floor(i/6))][i%6].set(xlabel = 'Frequency (MHz)');
        else:
            ax[int(np.floor(i/6))][i%6].set(xticks=[]);
    for i in range(nsnap*3,int(np.ceil(nsnap*3/6))*6):
        ax[int(np.floor(i/6))][i%6].set_axis_off();
    plt.suptitle(today + ' -- Power spectrum per antenna [linear]');
    plt.tight_layout();
    return ax;


## median flag mask over all antennas


def median_flag_specgram(xfreq,times,mask,today):
    fig, ax = plt.subplots(1,2,figsize=(16,8));
    ax[0].imshow(np.flip(np.median(mask[:,0,:,:],axis=0),axis=1),aspect='auto',extent=[xfreq[-1],xfreq[0],times[-1],times[0]],interpolation='None');
    ax[0].yaxis_date();
    date_format = mdates.DateFormatter('%m-%d %H:%M:%S');
    ax[0].yaxis.set_major_formatter(date_format);
    ax[0].set(xlabel='frequency [MHz]', title='median mask -- pol X');
    ax[0].tick_params(labelrotation=45);
    plt.tight_layout();
    ax[1].imshow(np.flip(np.median(mask[:,1,:,:],axis=0),axis=1),aspect='auto',extent=[xfreq[-1],xfreq[0],times[-1],times[0]],interpolation='None');
    ax[1].yaxis_date();
    date_format = mdates.DateFormatter('%m-%d %H:%M:%S');
    ax[1].yaxis.set_major_formatter(date_format);
    ax[1].set(xlabel='frequency [MHz]', title='median mask -- pol Y');
    ax[1].tick_params(labelrotation=45);
    plt.suptitle(today + ' -- median mask');
    plt.tight_layout();
    return ax;

def weather_forecast(today):
    response = requests.get('https://api.openweathermap.org/data/2.5/forecast?lat=37.233370&lon=-118.283443&appid=f340f25957dbaa37fbad285da8789a36&units=metric');
    response = response.json();

    numdat = len(response['list']);
    timarray = [];
    temps = np.zeros((numdat)); 
    weath = [];

    for k in range(numdat):
        temps[k] = response['list'][k]['main']['temp'];
        timarray.append(datetime.utcfromtimestamp(response['list'][k]['dt']));
        weath.append(response['list'][k]['weather'][0]['main']);

    times = mdates.date2num(timarray);

    fig, ax = plt.subplots(figsize=(16,5));
    ax.plot_date(times,temps,'-*');
    for k in range(numdat):
        ax.annotate(weath[k],(times[k],temps[k]), textcoords='data');
    plt.grid();
    plt.xlabel('time - date');
    plt.ylabel('temperature [C]');
    plt.title(today + ' -- weather forecast');
    plt.tight_layout();
    return ax;
