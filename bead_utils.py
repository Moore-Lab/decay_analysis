import glob, os, h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
import datetime as dt
import matplotlib.dates
from scipy.special import erf
import scipy.stats


def get_data(fname, gain_error=1.0):
    ### Get bead data from a file.  Guesses whether it's a text file
    ### or a HDF5 file by the file extension

    _, fext = os.path.splitext( fname )
    if( fext == ".h5"):
        try:
            f = h5py.File(fname,'r')
            dset = f['beads/data/pos_data']
            dat = np.transpose(dset)
            dat = dat / 3276.7 ## hard coded scaling from DAQ
            attribs = dset.attrs

        except (KeyError, IOError):
            print("Warning, got no keys for: ", fname)
            dat = []
            attribs = {}
            f = []
    else:
        dat = np.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5])
        attribs = {}
        f = []

    return dat, attribs, f

def plot_raw_data(dat, attr, nfft=-1, do_psd=False, do_filt=True):

    tvec = np.arange(len(dat[:,0]))/attr['Fsamp']

    plt.figure(figsize = (14, 16))

    if(do_filt):
        fc = np.array([10, 100])/(attr['Fsamp']/2)
        b, a = sp.butter(3, fc, btype='bandpass')
        xdat = sp.filtfilt(b,a,dat[:,0])
        ydat = sp.filtfilt(b,a,dat[:,1])
        zdat = sp.filtfilt(b,a,dat[:,2])
    else:
        xdat, ydat, zdat = dat[:,0], dat[:,1], dat[:,2]

    plt.subplot(4,1,1)
    plt.plot(tvec, xdat, 'k', label='X')
    plt.xticks([])
    plt.legend(loc='upper right')
    plt.xlim([0, tvec[-1]])
    
    plt.subplot(4,1,2)
    plt.plot(tvec, ydat, 'b', label='Y')
    plt.xticks([])
    plt.legend(loc='upper right')
    plt.xlim([0, tvec[-1]])
    
    plt.subplot(4,1,3)
    plt.plot(tvec, zdat, 'r', label='Z')
    plt.xticks([])
    plt.legend(loc='upper right')
    plt.xlim([0, tvec[-1]])

    plt.subplot(4,1,4)
    plt.plot(tvec, dat[:,10], 'g', label='Drive')
    plt.xlabel("Time [s]")
    plt.legend(loc='upper right')
    plt.xlim([0, tvec[-1]])

    plt.subplots_adjust(hspace=0)

    if(do_psd):
        if(nfft < 0): nfft = len(dat[:,0])
       
        f, x_psd = sp.welch(dat[:,0], fs=attr['Fsamp'], nperseg=nfft)
        f, y_psd = sp.welch(dat[:,1], fs=attr['Fsamp'], nperseg=nfft)
        f, z_psd = sp.welch(dat[:,2], fs=attr['Fsamp'], nperseg=nfft)
        f, d_psd = sp.welch(dat[:,10], fs=attr['Fsamp'], nperseg=nfft)

        plt.figure(figsize = (14, 12))

        plt.subplot(4,1,1)
        plt.loglog(f, x_psd, 'k', label='X')
        plt.xticks([])
        plt.legend(loc='upper right')
        plt.xlim([10, 1e3])
        
        plt.subplot(4,1,2)
        plt.loglog(f, y_psd, 'b', label='Y')
        plt.xticks([])
        plt.legend(loc='upper right')
        plt.xlim([10, 1e3])
        
        plt.subplot(4,1,3)
        plt.loglog(f, z_psd, 'r', label='Z')
        plt.xticks([])
        plt.legend(loc='upper right')
        plt.xlim([10, 1e3])

        plt.subplot(4,1,4)
        plt.loglog(f, d_psd, 'g', label='Drive')
        plt.xlabel("Freq [Hz]")
        plt.legend(loc='upper right')
        plt.xlim([10, 1e3])

    plt.show()

def correlation_template(dat, attr, make_plots=False, lp_freq=20):
    ## first make the drive template (cut out the first pulse)
    drive_dat = dat[:,10]
    drive_dat_filt = drive_dat**2
    fc = lp_freq/(attr['Fsamp']/2) ## lowpass at 20 Hz
    b, a = sp.butter(3, fc, btype='lowpass')
    drive_dat_filt = sp.filtfilt(b,a,drive_dat_filt)
    thresh = 0.5*np.max(drive_dat_filt) ## threshold at 50%
    thresh_cross = np.gradient(1.0*(drive_dat_filt > thresh))
    drive_start = np.argwhere( thresh_cross > 0.25 )[0][0] ## first upward crossing
    drive_end = np.argwhere( thresh_cross < -0.25 )[0][0] ## first downward crossing

    tvec = np.arange(len(dat[:,0]))/attr['Fsamp']

    if(make_plots): ## diagnostics for template finding
        plt.figure()
        plt.plot(tvec, drive_dat)
        #plt.plot(tvec, drive_dat_filt)
        plt.plot(tvec[drive_start:drive_end], drive_dat[drive_start:drive_end])
        #plt.plot(tvec,thresh_cross)
        #plt.plot(tvec[:-1],thresh_cross)
        plt.xlim(0.9*tvec[drive_start], 1.1*tvec[drive_end])

    ## now correlate against original time stream
    template = drive_dat[drive_start:drive_end]

    return template


def plot_correlation_with_drive(dat, template, attr, skip_drive=False, lp_freq=20, make_plots=False):

    corr_vec = sp.correlate(dat[:,0], template, mode='same')

    ## filter for RMS calculation
    fc = lp_freq/(attr['Fsamp']/2) ## lowpass at 20 Hz
    b, a = sp.butter(3, fc, btype='lowpass')

    ## special case for 20230803 when two drives were on
    ## due to the beating want to remove the times when the second drive was there
    ## this is the opposite of what we usually want
    if(skip_drive):
        drive_dat = dat[:,10]
        drive_dat_filt = drive_dat**2
        drive_dat_filt = sp.filtfilt(b,a,drive_dat_filt)
        thresh = 0.1*np.max(drive_dat_filt) ## threshold at 50%
        bad_pts = drive_dat_filt > thresh
        corr_vec = corr_vec[~bad_pts] 

    corr_vec_rms = sp.filtfilt(b,a,corr_vec**2)

    if(make_plots):
        plt.figure()
        plt.plot(corr_vec)
        plt.plot(corr_vec_rms)
        plt.plot(np.ones_like(corr_vec)*np.median(corr_vec_rms),'r:')
        #plt.xlim(0,10000)
        plt.show()

    return np.median(corr_vec_rms)


def labview_time_to_datetime(lt):
    ### Convert a labview timestamp (i.e. time since 1904) to a 
    ### more useful format (pytho datetime object)
    
    ## first get number of seconds between Unix time and Labview's
    ## arbitrary starting time
    lab_time = dt.datetime(1904, 1, 1, 0, 0, 0)
    nix_time = dt.datetime(1970, 1, 1, 0, 0, 0)
    delta_seconds = (nix_time-lab_time).total_seconds()

    lab_dt = dt.datetime.fromtimestamp( lt - delta_seconds)
    
    return lab_dt
    
