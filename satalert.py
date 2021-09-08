import os
import requests
import time
import pycurl
from io import BytesIO
import datetime
import math
import numpy as np
import ephem
from astropy.time import Time
import matplotlib as mpl
import matplotlib.pyplot as plt
import dsautils.dsa_store as ds
import csv
from slack_sdk import WebClient
#mpl.rcParams['timezone'] = 'US/Pacific';

client = WebClient(token='xoxb-508911196752-2019052043654-uADCWGbhAN5mmdJbSvKub1Ho');

def updateTLE(system):
    # update TLE
    file=open('/home/user/T3_detect/satellite/satalert/tle.txt', 'w');
    file.write("# Last updated: %s\n"%(time.time()));

    b_obj = BytesIO();
    crl = pycurl.Curl();

    # Set URL value
    if system == 'GLONASS':
        crl.setopt(crl.URL, 'http://celestrak.com/NORAD/elements/glo-ops.txt');
    elif system == 'GPS':
        crl.setopt(crl.URL, 'http://celestrak.com/NORAD/elements/gps-ops.txt');
    elif system == 'GALILEO':
        crl.setopt(crl.URL, 'http://celestrak.com/NORAD/elements/galileo.txt');
    elif system == 'BEIDOU':
        crl.setopt(crl.URL, 'https://www.celestrak.com/NORAD/elements/beidou.txt');
    else:
        crl.setopt(crl.URL, 'https://www.celestrak.com/NORAD/elements/gnss.txt');
    crl.setopt(crl.WRITEDATA, b_obj);
    crl.perform();
    crl.close();
    # Get the content stored in the BytesIO object (in byte characters) 
    get_body = b_obj.getvalue();
    file.write(get_body.decode('utf8'));


def computeFlyBys(system):

    updateTLE(system);
    
    obs = ephem.Observer();
    # DSA : https://github.com/dsa110/dsa110-antpos/blob/master/antpos/data/DSA110_positions_RevF.csv
    obs.lat = '37.233386982';
    obs.long = '-118.283405115';
    obs_az = 180.;
    # read mean antenna elevation from etcd
    my_ds = ds.DsaStore();
    tmp = [];
    for k in range(110):
         vv = my_ds.get_dict('/mon/ant/'+str(k+1));
         if vv != None and 'ant_el' in vv:
             tmp.append(vv['ant_el']);
    obs_el = np.mean(tmp);

    if obs_el > 90.:
        obs_el = 90.-(obs_el-90.);
        obs_az = obs_az - 180.;

    ## make it new file to avoid file corruption with other codes
    file = open('/home/user/T3_detect/satellite/satalert/tle.txt');
    tlefile = file.readlines();
    file.close();
    nSats = int((len(tlefile)-1)/3.);

    iss = [];
    for k in range(nSats):
        iss.append(ephem.readtle(tlefile[k*3+1], tlefile[k*3+2], tlefile[k*3+3]));

    # added 1 hour at the end to have predictions
    base = datetime.datetime.utcnow()
    tim = np.array([base + datetime.timedelta(seconds=i) for i in range(3600)]);
    tim_nums = mpl.dates.date2num(tim);

    ## compute positions
    ntimes = len(tim);
    distsat = np.zeros((ntimes,nSats));
    for k in range(ntimes):
        start_time_utc = tim[k];
        obs.date = ephem.Date(start_time_utc);
        for kk in range(nSats):
            iss[kk].compute(obs);
            distsat[k,kk] = math.sqrt((obs_az - (-np.abs(math.degrees(iss[kk].az)-180)+180))**2 + (obs_el - math.degrees(iss[kk].alt))**2);
            
    return distsat, tim_nums, iss;

def updateDB(distsat, tim_nums, iss):
    with open('/home/user/T3_detect/satellite/satalert/sat_flybys.csv', 'r', encoding='UTF8', newline='') as fr: # read csv
        r = csv.reader(fr);
        with open('/home/user/T3_detect/satellite/satalert/sat_flybys.csv', 'a', encoding='UTF8', newline='') as fw:    # open to append to csv
            writer = csv.writer(fw)
            for k in range(distsat.shape[1]):  # for each satellite
                isunique = True;
                if np.sum(distsat[:,k]<8.) > 0. and distsat[-1,k] > 8.: # if satellite gets close to beam
                    idxs = np.where(distsat[:,k]<8.);
                    tstart = tim_nums[idxs[0][0]];
                    tstop = tim_nums[idxs[0][-1]];
                    tclosest = tim_nums[np.argmin(distsat[:,k])];
                    distclos = distsat[np.argmin(distsat[:,k]),k];
                    fr.seek(0);
                    for row in r:   # for each row, if current satellite does not already ecist in database....
                        if row[0] == iss[k].name and tstart > float(row[1])-1./24 and tstart < float(row[1])+1./24:
                            isunique = False;
                    if isunique:
                        #print('satellite '+iss[k].name+' starts at '+str(tstart)+' - stops at '+str(tstop)+' - closest at '+str(tclosest)+' with '+str(distclos)+' degrees apart');
                        data = [iss[k].name,tstart,tstop,tclosest,distclos];
                        writer.writerow(data);
                        
                        messslack = ':satellite: Satellite '+iss[k].name+\
                        '\nApproaches at '+mpl.dates.num2date(tstart).astimezone().strftime('%Y-%m-%d %H:%M:%S')+\
                        '\nLeaves at '+mpl.dates.num2date(tstop).astimezone().strftime('%Y-%m-%d %H:%M:%S')+\
                        '\nClosest to beam at '+mpl.dates.num2date(tclosest).astimezone().strftime('%Y-%m-%d %H:%M:%S')+\
                        '\nMinimum separation to beam : '+"{:.2f}".format(distclos)+' deg.';
                        client.chat_postMessage(channel="#rfimitigation",text=messslack);
                        #print(messslack);

distsat, tim_nums, iss = computeFlyBys('galileo');
updateDB(distsat, tim_nums, iss);
distsat, tim_nums, iss = computeFlyBys('beidou');
updateDB(distsat, tim_nums, iss);
distsat, tim_nums, iss = computeFlyBys('gps');
updateDB(distsat, tim_nums, iss);