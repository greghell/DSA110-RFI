## analyzes how close GLONASS satellites get to beam

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
import matplotlib.pyplot as plt
import dsautils.dsa_store as ds

def sat_analysis(dirs,system):
    ## Updates TLE file
    file=open("/home/user/T3_detect/satellite/tle.txt", 'w');
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

    ## Define telescope
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
    #obs_el = 107.65;
    
    if obs_el > 90.:
        obs_el = 90.-(obs_el-90.);
        obs_az = obs_az - 180.;

    ## Read TLE
    file = open('/home/user/T3_detect/satellite/tle.txt');
    tlefile = file.readlines();
    file.close();
    nSats = int((len(tlefile)-1)/3.);

    iss = [];
    for k in range(nSats):
        iss.append(ephem.readtle(tlefile[k*3+1], tlefile[k*3+2], tlefile[k*3+3]));
    
    # added 1 hour at the end to have predictions
    tim = np.linspace(float(dirs[0].split('/')[-1]),float(dirs[-1].split('/')[-1])+1./24.,len(dirs)*100);

    ## compute positions
    ndirs = len(dirs);
    distsat = np.zeros((3,len(tim),nSats));
    timdtm = [];
    for k in range(len(tim)):
        start_time_utc = Time(tim[k],format='mjd').to_value('datetime');
        timdtm.append(start_time_utc);
        obs.date = ephem.Date(start_time_utc);
        for kk in range(nSats):
            iss[kk].compute(obs);
#            distsat[0,k,kk] = math.sqrt((obs_az - math.degrees(iss[kk].az))**2 + (obs_el - math.degrees(iss[kk].alt))**2);
            distsat[0,k,kk] = math.sqrt((obs_az - (-np.abs(math.degrees(iss[kk].az)-180)+180))**2 + (obs_el - math.degrees(iss[kk].alt))**2);
            distsat[1,k,kk] = math.degrees(iss[kk].az);
            distsat[2,k,kk] = math.degrees(iss[kk].alt);
            
    return distsat, timdtm;

    # plt.figure();
    # plt.plot(distsat[0,:,:]%180);
    # plt.show();

    # plt.figure();
    # for k in range(10):
        # plt.plot(distsat[2,:,k],distsat[1,:,k],'b.');
    # k = 4; plt.plot(distsat[2,:,k],distsat[1,:,k],'g.');
    # plt.plot(obs_az,obs_el,'*r');
    # plt.show();

    

