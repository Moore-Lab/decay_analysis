### Functions for analyzing data from the alpha recoil paper
### Currently this file is imported by the analysis scripts for each data
### set, to use common functions for analyzing all spheres. In the future,
### it would be useful to organize into a class structure.

import glob, os, h5py, re
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
import datetime as dt
import matplotlib.dates
from scipy.special import erf
import scipy.stats
from natsort import natsorted
import matplotlib.dates as mdates
import numexpr as ne
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import pickle
from scipy.stats import zscore


## columns in the data files
x_idx, y_idx, z_idx = 0, 1, 2
drive_idx = 8 ## for data starting 9/27/2023
coords_dict = {'x': x_idx, 'y': y_idx, 'z': z_idx}

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
            f = None
    else:
        dat = np.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5])
        attribs = {}
        f = []

    return dat, attribs, f

def get_alpha_data(fname):
    ## get the data saved in the h5 files for the photodiode
    f = h5py.File(fname,'r')
    dat = f['voltage']
    time = f['time']

def plot_raw_data(dat, attr, nfft=-1, do_psd=False, do_filt=True):
    ## helper function to plot raw data and optionally PSD
    tvec = np.arange(len(dat[:,0]))/attr['Fsamp']

    plt.figure(figsize = (14, 16))

    if(do_filt):
        fc = np.array([30, 50])/(attr['Fsamp']/2)
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

def correlation_template_cw(dat, attr, length=0.1, use_window=True, make_plots=False):
    ## Make a template for correlating against the charge data, assuming constant drive:
    ## 1) find the drive frequency
    ## 2) make a sine wave at that frequency, for length given in seconds
    ## 3) window if desired
    drive_dat = dat[:,drive_idx]
    f, p = sp.welch(drive_dat, attr['Fsamp'], nperseg=2**int(np.log(len(drive_dat))))
    gpts = (f > 65) & (f < 500) ## search only reasonable range for drive, skip 60 Hz
    pmax = 1.0*p
    pmax[~gpts] = 0
    drive_freq = f[np.argmax(pmax)]
    print("Drive frequency is: %.2f Hz"%drive_freq)
    if(make_plots):
        plt.figure()
        plt.semilogy(f, p)
        plt.plot(drive_freq, p[np.argmax(pmax)], 'ro', label="Drive freq.")
        plt.xlim(0,200)
        plt.xlabel("Freq [Hz]")
        plt.ylabel("PSD [arb units/Hz]")
        plt.legend()
        plt.show()

    tvec = np.arange(0, length, 1/attr['Fsamp'])
    template = np.sin(2*np.pi*tvec*drive_freq)
    if(use_window):
        template *= sp.windows.hamming(len(template))  

    if(make_plots):
        plt.figure()
        plt.plot(tvec, template)
        plt.xlabel("Time [s]")
        plt.ylabel("Amp [arb units]")
        plt.show()

    return template, drive_freq

def correlation_template(dat, attr, make_plots=False, lp_freq=20):
    ## make a template for correlating against the charge data

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
        plt.plot(tvec, drive_dat_filt)

    ## now correlate against original time stream
    template = drive_dat[drive_start:(drive_start + 40*115)]
    window = sp.windows.hamming(len(template))

    return template*window

def get_noise_template(noise_files, nfft=-1, res_pars=[2*np.pi*30, 2*np.pi*5]):
    ## take noise files and find the average PSD for use in the optimal filter
    ## this version is for the 1D case (x direction only)

    J = 0
    nfiles = 0
    for nf in noise_files:
        cdat, attr, _ = get_data(nf)

        if(nfft < 0):
            nfft_to_use = len(cdat[:,x_idx])
        else:
            nfft_to_use = nfft

        cf, cpsd = sp.welch(cdat[:,x_idx], fs=attr['Fsamp'], nperseg=nfft_to_use)

        J += cpsd
        nfiles += 1

    J /= nfiles

    ## expected for resonator params
    eta = 2*res_pars[1]/res_pars[0]
    omega0, gamma = res_pars[0]/np.sqrt(1 - eta**2), 2*res_pars[1]
    omega = 2*np.pi*cf
    sphere_tf = (gamma/((omega0**2 - omega**2)**2 + omega**2*gamma**2))
    res_pos = np.argmin( np.abs(omega0-omega) )
    sphere_tf *= J[res_pos]/sphere_tf[res_pos]

    Jout = 1.0*J
    Jout[J/sphere_tf > 20] = 1e20
    bad_freqs = (cf < 5) | (cf > 115) ## coarse band pass in x
    Jout[bad_freqs] = 1e20

    plt.figure()
    plt.semilogy(cf, J, 'k', label="Measured")
    plt.semilogy(cf, sphere_tf, "-", color='orange', label="Expected")
    plt.semilogy(cf, Jout, 'b', label="Filter")
    plt.xlim(0,200)
    gpts = (cf > 1) & (cf<200)
    plt.ylim(0.1*np.min(J[gpts]), 10*np.max(J[gpts]))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V$^2$/Hz]")
    plt.legend(loc="upper right")
    plt.show()

    noise_dict = {"freq": cf, "J": Jout, "Jorig": J}

    return noise_dict

def get_noise_template_3D(noise_files, fit_vals, range_dict, nfft=-1):
    ## take noise files and find the average PSD for use in the optimal filter
    ## this version works for 3D (x, y, and z)

    coords_dict = {'x': x_idx, 'y': y_idx, 'z': z_idx}
    noise_dict = {}

    plt.figure(figsize=(20,5))

    for cidx, coord in enumerate(coords_dict.keys()):
        noise_dict[coord] = {}
        coord_idx = coords_dict[coord]
        J = 0
        nfiles = 0

        res_pars = [fit_vals[coord][coord][1], fit_vals[coord][coord][2]]

        for nidx, nf in enumerate(noise_files):
            cdat, attr, _ = get_data(nf)

            if(nfft < 0):
                nfft_to_use = len(cdat[:,coord_idx])
            else:
                nfft_to_use = nfft

            cf, cpsd = sp.welch(cdat[:,coord_idx], fs=attr['Fsamp'], nperseg=nfft_to_use)

            if(nidx==0):
                Jmat = cpsd
            else:
                Jmat = np.vstack((Jmat, cpsd))
            nfiles += 1

        J = np.median(Jmat, axis=0)

        ## expected for resonator params
        eta = 2*res_pars[1]/res_pars[0]
        omega0 = res_pars[0]/np.sqrt(1 - eta**2) if eta < 1 else res_pars[0]
        gamma = 2*res_pars[1]
        omega = 2*np.pi*cf
        sphere_tf = (gamma/((omega0**2 - omega**2)**2 + omega**2*gamma**2))
        res_pos = np.argmin( np.abs(omega0-omega) )
        pts = J[(res_pos-10):(res_pos+10)]/np.median(sphere_tf[(res_pos-10):(res_pos+10)])
        sphere_tf *= np.median(pts[pts>0])

        Jout = 1.0*J
        ## cut frequencies with bad signal-to-noise
        if(coord != 'z'):
            Jout[J/sphere_tf > 20] = 1e20
        ## if desired, also limit the range within reasonable cutoffs
        bad_freqs = (cf < range_dict[coord][0]) | (cf > range_dict[coord][1])
        Jout[bad_freqs] = 1e20

        plt.subplot(1,3,cidx+1)
        plt.semilogy(cf, J, 'k', label="Measured")
        plt.semilogy(cf, sphere_tf, "-", color='orange', label="Expected")
        plt.semilogy(cf, Jout, 'b', label="Filter")
        plt.xlim(0,range_dict[coord][1]*1.5)
        gpts = (cf > 1) & (cf<200)
        plt.ylim(0.1*np.min(J[gpts]), 10*np.max(J[gpts]))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [V$^2$/Hz]")
        plt.legend(loc="upper right")
        plt.title("%s direction"%coord)

        noise_dict[coord] = {"freq": cf, "J": Jout, "Jorig": J}

    return noise_dict

def plot_noise_paper(noise_files, fit_vals, range_dict, ylim_dict, nfft=-1, cal={'x': 1, 'y': 1, 'z': 1}):
    ## take noise files and find the average PSD for use in the optimal filter
    ## this is a helper function just used to make plots for the paper without
    ## interfering with the template functions just above

    colors_dict = {'x': ['darkblue', 'blue'], 'y': ['darkred', 'red'], 'z': ['k', 'orange']}

    coords_dict = {'x': x_idx, 'y': y_idx, 'z': z_idx}
    noise_dict = {}

    plt.figure(figsize=(12, 2.75))

    for cidx, coord in enumerate(coords_dict.keys()):
        noise_dict[coord] = {}
        coord_idx = coords_dict[coord]
        J = 0
        nfiles = 0

        res_pars = [fit_vals[coord][coord][1], fit_vals[coord][coord][2]]

        for nidx, nf in enumerate(noise_files):
            cdat, attr, _ = get_data(nf)

            if(nfft < 0):
                nfft_to_use = len(cdat[:,coord_idx])
            else:
                nfft_to_use = nfft

            cf, cpsd = sp.welch(cdat[:,coord_idx]/cal[coord]*1e9, fs=attr['Fsamp'], nperseg=nfft_to_use)

            if(nidx==0):
                Jmat = cpsd
            else:
                Jmat = np.vstack((Jmat, cpsd))
            nfiles += 1

        J = np.median(Jmat, axis=0)

        ## expected for resonator params
        eta = 2*res_pars[1]/res_pars[0]
        omega0 = res_pars[0]/np.sqrt(1 - eta**2) if eta < 1 else res_pars[0]
        gamma = 2*res_pars[1]
        omega = 2*np.pi*cf
        sphere_tf = (gamma/((omega0**2 - omega**2)**2 + omega**2*gamma**2))
        res_pos = np.argmin( np.abs(omega0-omega) )
        pts = J[(res_pos-10):(res_pos+10)]/np.median(sphere_tf[(res_pos-10):(res_pos+10)])
        sphere_tf *= np.median(pts[pts>0])

        if(coord in ['x', 'y']):
            sphere_tf_new = 1.0*sphere_tf
        else:
            sphere_tf_new = lorentz(omega, omega0, gamma, 9e4) ## fit amplitude

        plt.subplot(1,3,cidx+1)
        plt.semilogy(cf, np.sqrt(J), 'k', lw=1)
        plt.semilogy(cf, np.sqrt(sphere_tf_new), "-", color=colors_dict[coord][1])
        plt.xlim(1,range_dict[coord][1])
        plt.gca().set_xscale('log')
        gpts = (cf > 1) & (cf<200)
        plt.ylim(ylim_dict[coord])
        plt.xlabel("Frequency [Hz]")
        if(cidx==0):
            plt.ylabel("Position ASD [nm/$\sqrt{\mathrm{Hz}}$]")

        noise_dict[coord] = {"freq": cf, "J": Jout, "Jorig": J}
        plt.subplots_adjust(wspace=0.25)
    return noise_dict

def gauss_fun(x,A,mu,sig):
    return A*np.exp(-(x-mu)**2/(2*sig**2))

#def plot_charge_steps(charge_vec):
#    ## simple helper function to 
#    dt = []
#    for j,t in enumerate(charge_vec[:,0]):
#        dt.append(labview_time_to_datetime(t-charge_vec[0,0]))
#
#    plt.figure()
#    plt.plot(charge_vec[:,1], 'k.-')
#    plt.show()

def signed_correlation_with_drive(dat, attr, nperseg=-1, recal_fac = 1/170, use_window=False, drive_idx=drive_idx, has_drive=True):
    ### take the signed correlation with the driving field to determine the net charge of the particle

    xdat = dat[:,x_idx]
    ddat = dat[:,drive_idx]

    time_vec = attr["Time"] + np.arange(-len(xdat), 0)/attr['Fsamp'] ## time stamp at end of file

    if(nperseg < 0):
        nperseg = len(xdat)
    nwindows = int(len(xdat)/nperseg)
    st, en = 0, nperseg

    p = np.abs(np.fft.rfft(ddat[st:en]))**2
    f = np.fft.rfftfreq(len(ddat[st:en]), d=1/attr['Fsamp'])
    gpts = (f > 65) & (f < 500) ## search only reasonable range for drive, skip 60 Hz
    pmax = 1.0*p
    pmax[~gpts] = 0
    didx = np.argmax(pmax)

    corr_vec = []
    corr2_vec = []
    corr3_vec = []
    drive_vec = []
    time_out = []

    for i in range(nwindows):
        st, en = i*nperseg, (i+1)*nperseg
        ctime = np.median(time_vec[st:en])
        if(use_window):
            window = sp.windows.hamming(en-st)
        else:
            window = np.ones(en-st)
        if(has_drive):
            cratio = (np.fft.rfft(xdat[st:en]*window)/np.fft.rfft(ddat[st:en]*window))[didx]
            corr = -np.real(cratio) ## negative sign gives the charge with positve as excess protons
            corr2 = -np.abs(cratio)*np.sign(np.real(cratio))
            corr3 = -np.abs(np.fft.rfft(xdat[st:en]*window)[didx])*np.sign(np.real(cratio))*recal_fac
        else:
            cratio = (np.fft.rfft(xdat[st:en]*window))[didx] * recal_fac
            corr = -np.real(cratio) ## negative sign gives the charge with positve as excess protons
            corr2 = -np.abs(cratio)*np.sign(np.real(cratio))
            corr3 = -np.abs(np.fft.rfft(xdat[st:en]*window)[didx])*np.sign(np.real(cratio))*recal_fac

        drive_vec.append(np.abs(np.fft.rfft(ddat[st:en])[didx]))

        corr_vec.append(corr)
        corr2_vec.append(corr2)
        corr3_vec.append(corr3)
        time_out.append(ctime)

    data_out = np.vstack((np.array(corr_vec), np.array(corr2_vec), np.array(corr3_vec), np.array(drive_vec), np.array(time_out))).T
    return data_out


def simple_correlation_with_drive(dat, attr, drive_freq, bw=1, decstages = -1, cal_fac=5e-6, make_plots=False):
    ### simple bandpass filter to calculate the correlation. This function doesn't preserve the
    ### phase information 

    fc = np.array([drive_freq-bw/2, drive_freq+bw/2])/(attr['Fsamp']/2) ## lowpass at 20 Hz
    b, a = sp.butter(3, fc, btype='bandpass')
    corr_vec = sp.filtfilt(b,a,dat[:,x_idx])

    fc2 = bw/(attr['Fsamp']/2) ## lowpass at 20 Hz
    b2, a2 = sp.butter(3, fc2, btype='lowpass')
    corr_vec_rms = sp.filtfilt(b2,a2,corr_vec**2)

    ## cut edge effects
    nsamps = int(2*attr['Fsamp']/bw)
    corr_vec_rms = corr_vec_rms[nsamps:-nsamps]

    if(make_plots):
        tvec = np.arange(len(dat[:,x_idx]))/attr['Fsamp']
        tvec = tvec[nsamps:-nsamps]
        plt.figure()
        plt.plot(tvec, corr_vec_rms/cal_fac, '.-')

    ## if desired do another lowpass to see steps
    if(decstages>1):
        step_size = int(len(corr_vec_rms)/decstages)
        out_vec = []
        for n in range(decstages-1):
            out_vec.append( np.median( corr_vec_rms[(n*step_size):((n+1)*step_size)]) )

        corr_vec_rms = np.array(out_vec)
        if(make_plots):
            out_tvec = []
            for n in range(decstages-1):
                out_tvec.append( np.median(tvec[(n*step_size):((n+1)*step_size)]) )
            plt.plot(out_tvec, corr_vec_rms/cal_fac, '.-')

    if(make_plots):
        plt.show()

    return np.median(corr_vec_rms)/cal_fac, corr_vec_rms/cal_fac

# def plot_correlation_with_drive(dat, template, attr, skip_drive=False, lp_freq=20, make_plots=False):

#     corr_vec = sp.correlate(dat[:,x_idx], template, mode='same')

#     ## filter for RMS calculation
#     fc = lp_freq/(attr['Fsamp']/2) ## lowpass at 20 Hz
#     b, a = sp.butter(3, fc, btype='lowpass')


#     ## special case for 20230803 when two drives were on
#     ## due to the beating want to remove the times when the second drive was there
#     ## this is the opposite of what we usually want
#     if(skip_drive):
#         drive_dat = dat[:,drive_idx]
#         drive_dat_filt = drive_dat**2
#         drive_dat_filt = sp.filtfilt(b,a,drive_dat_filt)
#         thresh = 0.1*np.max(drive_dat_filt) ## threshold at 50%
#         bad_pts = drive_dat_filt > thresh
#         corr_vec = corr_vec[~bad_pts] 

#     corr_vec_rms = sp.filtfilt(b,a,corr_vec**2)

#     if(make_plots):
#         tvec = np.arange(len(dat[:,0]))/attr['Fsamp']
#         plt.figure(figsize = (14, 3))
#         plt.plot(tvec,corr_vec)
#         plt.plot(tvec,corr_vec_rms)
#         plt.plot(tvec,np.ones_like(corr_vec)*np.median(corr_vec_rms),'r:')
#         plt.show()

#     return np.median(corr_vec_rms), corr_vec_rms

def get_lamp_and_filament(dat, npts):
    ## helper function to determine when the lamp and filament are on

    lamp_thresh = 0.1 ## threshold for lamp above which it is on
    filament_thresh = 0.1
    fil_col = 12
    lamp_col = 13

    nsegs = int(len(dat[:,0])/npts)

    lamp_dat = np.zeros(nsegs, dtype='bool')
    fil_dat = np.zeros(nsegs, dtype='bool')

    for n in range(nsegs):
        lamp_dat[n] = np.max(dat[(n*npts):((n+1)*npts),lamp_col]) > lamp_thresh
        fil_dat[n] = np.max(dat[(n*npts):((n+1)*npts),fil_col]) > filament_thresh
    
    return lamp_dat, fil_dat

def fill_dps(dead_period_edges, color='gray', lab="Dead time", ax=[], line=False, start_time=0):
    ## helper function to plot dead periods (e.g. when the lamp and filament are on)

    if(not ax):
        ax = plt.gca()
    ## define plotting function for dead times
    yy = ax.get_ylim()
    for i,dp in enumerate(dead_period_edges):
        if(dp[1] < start_time): continue
        if(i==0 and len(lab)>0):
            if(line):
                ax.plot([dp[0], dp[0]], [yy[0],yy[1]], color=color, alpha=0.2, label=lab, lw=0.5)
            else:
                ax.fill_between(dp, [yy[0],yy[0]], [yy[1],yy[1]], color=color, alpha=0.2, label=lab)
        else:
            if(line):
                ax.plot([dp[0], dp[0]], [yy[0],yy[1]], color=color, alpha=0.2, label=lab)
            else:
                ax.fill_between(dp, [yy[0],yy[0]], [yy[1],yy[1]], color=color, alpha=0.2, lw=0.5)  
    ax.set_ylim(yy)

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
    
def chi(omega, omega0, gamma, A):
    return A / (omega0**2 - omega**2 - 1j*omega*gamma)

def lorentz(omega, omega0, gamma, A):
    return np.abs(chi(omega, omega0, gamma, A))**2

def fit_tf(dat, attr, nfft=2**19, frange=[20,80]):
    ## fit the transfer function to a lorentzian

    f, x_psd = sp.welch(dat[:,0], fs=attr['Fsamp'], nperseg=nfft)

    fit_pts = (f >= frange[0]) & (f <= frange[1])

    spars = [2*np.pi*50, 2*np.pi*10, 1]
    bp, bc = curve_fit(lorentz, 2*np.pi*f[fit_pts], x_psd[fit_pts], p0=spars)

    print("Best fit, f_0=%.1f, gamma=%.1f"%(bp[0]/(2*np.pi), bp[1]/(2*np.pi)))

    plt.figure()
    plt.loglog(f,x_psd)
    plt.loglog(f[fit_pts], lorentz(2*np.pi*f[fit_pts], *bp))
    plt.show()

    return np.abs(bp[0]), np.abs(bp[1])

def sfunc(t, A, delta, omega0):
    return A*np.sin(omega0*t + delta)

def sfunc_step(t, A1, A2, delta, omega0, t0):
    ## stepped sign function
    o1 = A1*np.sin(omega0*t + delta)
    o1[t > t0] = A2*np.sin(omega0*t[t>t0] + delta)
    return o1

def impulse_response(t, A, omega1, gamma, t0):
    ## impulse response of damped harmonic oscillator
    outvec = np.zeros_like(t)
    gpts = t>t0
    outvec[gpts] = A*np.exp(-(t[gpts]-t0)*gamma)*np.sin(omega1*(t[gpts]-t0))
    return outvec

# def plot_impulse(dat, template, attr, res_pars, trange=[0, np.inf], skip_drive=False, lp_freq=20, make_plots=False, two_panel=True, fname=""):

#     corr_vec = sp.correlate(dat[:,0], template, mode='same')

#     ## filter for RMS calculation
#     fc = lp_freq/(attr['Fsamp']/2) ## lowpass at 20 Hz
#     b, a = sp.butter(3, fc, btype='lowpass')
#     corr_vec_rms = sp.filtfilt(b,a,corr_vec**2)

#     tvec = np.arange(len(dat[:,0]))/attr['Fsamp']

#     ## deconvolve the data by the transfer function
#     #fpts = (tvec > trange[0]) & (tvec < trange[1])
#     fpts = (tvec > trange[0]) & (tvec < trange[1])
#     sfunc_fixed = lambda t, A, d: sfunc(t, A, d, 2*np.pi*87)
#     #sfunc_fixed = lambda t, A1, A2, d: sfunc_step(t, A1, A2, d, 2*np.pi*87, 37)
#     sfit, scov = curve_fit(sfunc_fixed, tvec[fpts], dat[fpts,0], p0=[1, 0])
#     fc = np.array([20,50])/(attr['Fsamp']/2)
#     b, a = sp.butter(3, fc, btype='bandpass')
#     xfilt_orig = dat[:,0] - sfunc_fixed(tvec, *sfit)
#     xfilt2 = sp.filtfilt(b,a,xfilt_orig)
    
#     tt = tvec[:4000]
#     ttemp = np.exp(-tt*(5.7))*np.sin(2*np.pi*38.1*tt)
#     ttemp = np.hstack((np.zeros_like(ttemp), ttemp))

#     fc2 = 20/(attr['Fsamp']/2)
#     b2, a2 = sp.butter(3, fc2, btype='lowpass')
#     #xfilt3 = sp.filtfilt(b2,a2,xfilt_orig)
#     xfilt3 = sp.correlate(xfilt_orig, ttemp, mode='same')
#     xfilt3 = sp.filtfilt(b2, a2, xfilt3**2)
#     xfilt3 *= np.max(xfilt2)/np.max(xfilt3) * 1/2.5

#     xtilde = np.fft.rfft(xfilt_orig)
#     f = np.fft.rfftfreq(len(dat[:,0]), 1/attr['Fsamp'])
#     Ftilde = xtilde/chi(2*np.pi*f, res_pars[0], res_pars[1], 1e4)
#     Fdecon = np.fft.irfft(Ftilde)
#     fc = np.array([1,200])/(attr['Fsamp']/2)
#     b, a = sp.butter(3, fc, btype='bandpass')
#     Fdecon = sp.filtfilt(b,a,Fdecon)
#     Fdecon *= np.max(dat[:,0])/np.max(Fdecon[50000:-50000])

#     if(skip_drive):
#         fc = 100/(attr['Fsamp']/2)
#         b, a = sp.butter(3, fc, btype='lowpass')
#         drive_dat = dat[:,10]
#         drive_dat_filt = drive_dat**2
#         drive_dat_filt = sp.filtfilt(b,a,drive_dat_filt)
#         thresh = 0.1*np.max(drive_dat_filt) ## threshold at 50%
#         bad_pts = drive_dat_filt > thresh
#         ## pad around each crossing
#         buff=10000
#         upward_pts = np.where((bad_pts) & ~(np.roll(bad_pts,1)))[0]
#         for up in upward_pts:
#             bad_pts[(up-buff):up] = True
#         upward_pts = np.where(~(bad_pts) & (np.roll(bad_pts,1)))[0]
#         for up in upward_pts:
#             bad_pts[up:(up+buff)] = True

#         ## feedback issue
#         bpts2 = (tvec > 31.5) & (tvec < 33.5)
#         bad_pts = bad_pts | bpts2
#         bpts2 = (tvec < 5) 
#         bad_pts = bad_pts | bpts2

#     else:
#         bad_pts = np.zeros_like(dat[:,10]) > 1

#     if(make_plots):

#         plt.figure()
#         plt.plot(tvec,dat[:,0])
#         plt.plot(tvec[fpts], sfunc_fixed(tvec[fpts], *sfit), 'r')

#         xx = trange
#         plt.figure(figsize = (6,4.5))
#         #corr_vec_z = sp.correlate(dat[:,2], template, mode='same')
#         #corr_vec_z_rms = sp.filtfilt(b,a,corr_vec_z**2)
#         impulse_time = 63.6
#         wind_size = 0.75

#         if(two_panel):
#             plt.subplot(2,1,1)
#             #plt.plot(tvec, dat[:,0]-sfunc_fixed(tvec, *sfit))
#             #plt.plot(tvec, xfilt)
#             pulse_wind_size = 0.010

#             if(skip_drive):
#                 tvec = np.arange(0, len(tvec[~bad_pts]))/attr['Fsamp']

#             bpts = (tvec < impulse_time-wind_size) | (tvec > impulse_time+wind_size)
#             max_vec = xfilt3[~bad_pts]*1.0
#             max_vec[bpts]=0

#             #plt.ylim(-0.75,0.75)
#             plt.ylim(-1,1)
#             yy=plt.ylim()
#             pulse_time = tvec[np.argmax(max_vec)]
#             plt.plot([pulse_time, pulse_time], yy, 'k:')
#             plt.plot(tvec,xfilt2[~bad_pts], label="Data")
#             plt.plot(tvec,xfilt3[~bad_pts]*2.2, lw=2, label="Matched filt.")
#             #plt.plot(tt+impulse_time, ttemp/2, 'r')
#             plt.xlim(xx)

#             #plt.plot(impulse_time*np.ones(2), yy, 'k:')      
#             plt.ylim(yy)
#             plt.xticks([])
#             plt.ylabel('Position [arb units]')

#             #plt.fill_between([pulse_time-pulse_wind_size, pulse_time+pulse_wind_size], [yy[0], yy[0]], [yy[1], yy[1]], color='red', alpha=0.3)
#             plt.legend(loc='lower left')

#             #plt.subplot(4,1,2)
#             #plt.plot(tvec,dat[:,1])
#             #plt.xlim(xx)
#             #yy=plt.ylim()
#             #plt.plot(impulse_time*np.ones(2), yy, 'k:')      
#             #plt.ylim(yy)

#             #plt.subplot(4,1,3)
#             #plt.plot(tvec,dat[:,2])
#             #plt.xlim(xx)
#             #yy=plt.ylim()
#             #plt.plot(impulse_time*np.ones(2), yy, 'k:')      
#             #plt.ylim(yy)

#             plt.subplot(2,1,2)
#             cal_fac = 1.55e1*4.5
#             plt.plot(tvec,corr_vec_rms[~bad_pts]/cal_fac, 'gray')
#             b,a = sp.butter(3,0.00005)
#             corr_filt = sp.filtfilt(b,a,corr_vec_rms[~bad_pts]/cal_fac)
#             plt.plot(tvec,corr_filt, 'r')
#             #plt.plot(tvec,drive_dat_filt)    
#             #plt.plot(tvec[~bad_pts],drive_dat_filt[~bad_pts])    
#             #plt.plot(tvec,corr_vec_z_rms)
#             plt.xlim(xx)        
#             plt.grid(True)
#             #plt.ylim(74,80)
#             yy=plt.ylim()

#             plt.fill_between([impulse_time-wind_size, impulse_time+wind_size], [yy[0], yy[0]], [yy[1], yy[1]], color='orange', alpha=0.3)

#             #window = [80, 88]
#             #gpts = (tvec > window[0]) & (tvec < window[1])
#             #plt.plot(window, np.median(corr_vec_rms[gpts])*np.ones(2), 'r')

#             #plt.plot(impulse_time*np.ones(2), yy, 'k:')  
#             plt.ylim(yy)
#             #plt.ylim(77,85)
#             plt.xlabel("Time [s]")
#             plt.ylabel("Charge [$e$]")

#             plt.subplots_adjust(hspace=0)

#         else:
#             if(skip_drive):
#                 tvec = np.arange(0, len(tvec[~bad_pts]))/attr['Fsamp']

#             bpts = (tvec < impulse_time-wind_size) | (tvec > impulse_time+wind_size)
#             max_vec = xfilt3[~bad_pts]*1.0
#             max_vec[bpts]=0

#             plt.ylim(-0.6,0.6)
#             yy=plt.ylim()
#             pulse_time = tvec[np.argmax(max_vec)]
#             plt.plot([pulse_time, pulse_time], yy, 'k:')
#             plt.plot(tvec,xfilt2[~bad_pts], label="Data")
#             plt.plot(tvec,xfilt3[~bad_pts]*2.2, lw=2, label="Matched filt.")
#             #plt.plot(tt+impulse_time, ttemp/2, 'r')
#             plt.xlim(xx)

#             #plt.plot(impulse_time*np.ones(2), yy, 'k:')      
#             plt.ylim(yy)
#             #plt.xticks([])
#             plt.ylabel('Position [arb units]')
#             plt.xlabel("Time [s]")
#             #plt.fill_between([pulse_time-pulse_wind_size, pulse_time+pulse_wind_size], [yy[0], yy[0]], [yy[1], yy[1]], color='red', alpha=0.3)
#             plt.legend(loc='lower left')

#             #plt.figure(figsize=(6,4.5))
#             #stp = len(ttemp)
#             #h, b = np.histogram(2.2*xfilt3[~bad_pts][::int(stp/2)], bins=np.linspace(0,0.1,50))
#             #bc = b[:-1] + np.diff(b)/2
#             #plt.errorbar(bc, h, yerr=np.sqrt(h), fmt='ko')

#         if(len(fname)> 0):
#             plt.savefig(fname)
#         plt.show()

#     return np.median(corr_vec_rms)

def parse_impulse_amplitude(input_string, label="MeV"):
    ## helper function to figure out the impulse amplitude from the file name

    pattern = r'(\d+(\.\d+)?)\s*' + label

    # Use re.search to find the match in the input string
    match = re.search(pattern, input_string)
    if match:
        # Extract the matched number as a float
        number = float(match.group(1))
        return number
    else:
        return None

def assemble_file_dict(path_to_use, base_path, dict_to_use={}):

    for curr_path in path_to_use:
        curr_file_list = natsorted(glob.glob(os.path.join(base_path, curr_path, "**/*.h5"), recursive=True))

        for file in curr_file_list:
            impulse_amp = parse_impulse_amplitude(file, label='mV')
            if impulse_amp in dict_to_use.keys():
                dict_to_use[impulse_amp].append(file)
            else:
                dict_to_use[impulse_amp] = [file,]
    
    return dict_to_use

def find_crossing_indices(signal, threshold):
    """
    Find the indices where a signal crosses a threshold (both above and below).

    Parameters:
    - signal (list or numpy array): The input signal.
    - threshold (float): The threshold value.

    Returns:
    - rising_edges (numpy array): Indices where the signal crosses above the threshold.
    - falling_edges (numpy array): Indices where the signal crosses below the threshold.
    """
    above_threshold = signal >= threshold
    below_threshold = signal < threshold

    rising_edges = np.where(above_threshold & ~np.roll(above_threshold, 1))[0]
    falling_edges = np.where(below_threshold & ~np.roll(below_threshold, 1))[0]

    return rising_edges, falling_edges

def get_average_template(calib_dict, make_plots=False, fit_pars=[], drive_idx=drive_idx, coords_to_use=['x']):
    ## function to average many calibration data waveforms and determine the impulse response. This function
    ## works only in 1D (x direction)

    pulse_dict = {}
    fit_dict = {}
    fit_vals = []

    coords_dict = {'x': x_idx, 'y': x_idx+1, 'z': x_idx+2}

    for impulse_amp in calib_dict.keys():

        curr_files = calib_dict[impulse_amp]

        if(make_plots):
            plt.figure(figsize=(6, len(coords_to_use)*4))

        curr_dict = {}
        curr_fit_dict = {}
        for j, coord in enumerate(coords_to_use):
            curr_temp = 0
            for fname in curr_files:
                cdat, attr, _ = get_data(fname)

                ## find the impulse times
                drive_file = cdat[:,drive_idx]
                impulse_times, _ = find_crossing_indices(drive_file/np.max(drive_file), 0.5)
                window_length = 4000

                for time_idx in impulse_times:
                    coord_idx = coords_dict[coord]
                    ## make sure the full impulse is contained
                    if(time_idx + window_length/2 >= len(cdat[:,coord_idx]) or 
                    time_idx < window_length/2):
                        continue
                    sidx = int(time_idx-window_length/2)
                    eidx = int(time_idx+window_length/2)
                    curr_temp += cdat[sidx:eidx,coord_idx]

            curr_temp -= np.median(curr_temp[:int(window_length/4)]) #baseline sub

            if(len(coords_to_use)==1): ## backwards compatibility
                curr_temp /= np.max(curr_temp) ## normalize to unity amplitude
                pulse_dict[impulse_amp] = curr_temp
            else:
                curr_dict[coord] = curr_temp

            ## also fit to a damped harmonic oscillator
            tvec = np.arange(len(curr_temp))/attr['Fsamp']
            if(len(fit_pars)>0):
                bp, bc = curve_fit(impulse_response, tvec, curr_temp, p0=fit_pars)
            else:
                bp = [0,0,0,0]
                bc = np.zeros((len(bp), len(bp)))

            if(len(coords_to_use)==1): ## backwards compatibility
                fit_dict[impulse_amp] = impulse_response(tvec, *bp)
                ## save fit parameters as well
            else:
                curr_fit_dict[impulse_amp] = impulse_response(tvec, *bp)          

            fit_vals.append( [impulse_amp, bp[1], np.sqrt(bc[1,1]), bp[2], np.sqrt(bc[2,2]), j] )

            if(make_plots):
                plt.subplot(len(coords_to_use), 1, j+1)
                plt.plot(tvec, curr_temp)
                plt.plot(tvec, impulse_response(tvec, *bp))
                plt.title("Impulse amplitude = %d MeV"%impulse_amp)
                plt.xlabel("Time (s)")
                plt.ylabel("Normalized amplitude [arb units]")

        if(len(coords_to_use)>1):
            pulse_dict[impulse_amp] = curr_dict 
            fit_dict[impulse_amp] = curr_fit_dict 

    return pulse_dict, fit_dict, np.array(fit_vals)

def get_average_template_3D(calib_dict, make_plots=False, fit_pars=[], drive_idx=drive_idx, 
                            coords_to_use=['x', 'y', 'z'], xrange=[-1,-1], drive_dict=None,
                            lowpass_dict={'x': -1, 'y': -1, 'z': -1}):
    ## Function to average many calibration data waveforms and determine the impulse response. This function
    ## This version of the function makes templates for all 3 coordinates (x, y, z)

    pulse_dict = {}
    fit_dict = {}
    fit_vals = {}

    coords_dict = {'x': x_idx, 'y': x_idx+1, 'z': x_idx+2}
    if(not drive_dict):
        drive_dict = {'x': drive_idx, 'y': drive_idx+1, 'z': drive_idx-1}

    if(make_plots):
        plt.figure(figsize=(18, len(coords_to_use)*4))

    norm_dict = {}
    impulse_amp_list = []
    for j, coord in enumerate(coords_to_use): ## coord in which the drive is applied
        
        pulse_dict[coord] = {}
        fit_dict[coord] = {}
        fit_vals[coord] = {}

        #choose the smallest impulse amp for each coordinate
        impulse_amp = np.min(list(calib_dict[coord].keys()))
        impulse_amp_list.append(impulse_amp)

        curr_files = calib_dict[coord][impulse_amp]

        for k, resp_coord in enumerate(coords_to_use): ## coord to look at response for
            
            coord_idx = coords_dict[resp_coord]
            drive_coord_idx = drive_dict[coord]

            curr_temp = 0
            ntraces = 0
            for fname in curr_files:

                cdat, attr, _ = get_data(fname)

                if(lowpass_dict[coord] > 0):
                    b,a = sp.butter(3, lowpass_dict[coord]/(attr['Fsamp']/2), btype='lowpass') ## get rid of high freq noise
                    cdat[:,coord_idx] = sp.filtfilt(b,a,cdat[:,coord_idx])

                ## find the impulse times
                drive_file = cdat[:,drive_coord_idx]
                impulse_times, _ = find_crossing_indices(drive_file/np.max(drive_file), 0.5)
                window_length = 4000

                for time_idx in impulse_times:
                    ## make sure the full impulse is contained
                    if(time_idx + window_length/2 >= len(cdat[:,coord_idx]) or 
                    time_idx < window_length/2):
                        continue
                    sidx = int(time_idx-window_length/2)
                    eidx = int(time_idx+window_length/2)
                    curr_temp += cdat[sidx:eidx,coord_idx]
                    ntraces += 1

            curr_temp -= np.median(curr_temp[:int(window_length/4)]) #baseline sub
            curr_temp /= ntraces
            pulse_dict[coord][resp_coord] = curr_temp

            if( resp_coord == coord):
                norm_dict[coord] = np.max(np.abs(curr_temp))
                norm_dict[coord + "_amp"] = impulse_amp

    ## now go back through, normalize, and fit
    ## first fit the diagonal elements
    for j, coord in enumerate(coords_to_use):

        resp_coord = coord
        pulse_dict[coord][resp_coord] /= norm_dict[coord]

        curr_temp = pulse_dict[coord][resp_coord]
        tvec = np.arange(len(curr_temp))/attr['Fsamp']
        if(len(fit_pars)>0):
            bp, bc = curve_fit(impulse_response, tvec, curr_temp, p0=fit_pars[coord])
        else:
            bp = [0,0,0,0]
            bc = np.zeros((len(bp), len(bp)))

        fit_dict[coord][resp_coord] = impulse_response(tvec, *bp)
        fit_vals[coord][resp_coord] = [bp[0], bp[1], bp[2], bp[3], np.sqrt(bc[0,0]), np.sqrt(bc[1,1]), np.sqrt(bc[2,2]), np.sqrt(bc[3,3])]

    ## now fit the off diagonal elements with a cross talk and a cross force term
    for j, coord in enumerate(coords_to_use):
            for k, resp_coord in enumerate(coords_to_use): ## coord to look at response for
                
                if(k == j): continue

                pulse_dict[coord][resp_coord] /= norm_dict[coord]

                curr_temp = pulse_dict[coord][resp_coord]
                tvec = np.arange(len(curr_temp))/attr['Fsamp']
                curr_func = lambda tvec, Ax, Ay, Az: Ax*impulse_response(tvec, *fit_vals['x']['x'][:4]) + Ay*impulse_response(tvec, *fit_vals['y']['y'][:4]) + Az*impulse_response(tvec, *fit_vals['z']['z'][:4])
                if(len(fit_pars)>0):
                    bp, bc = curve_fit(curr_func, tvec, curr_temp, p0=[1, 1, 1])
                else:
                    bp = [0,0,0]
                    bc = np.zeros((len(bp), len(bp)))

                fit_dict[coord][resp_coord] = curr_func(tvec, *bp)
                fit_vals[coord][resp_coord] = bp

    if(make_plots):
        ## now fit the off diagonal elements with a cross talk and a cross force term
        for j, coord in enumerate(coords_to_use):
                for k, resp_coord in enumerate(coords_to_use): ## coord to look at response for        

                    subplot_idx = k*len(coords_to_use) + j + 1 
                    curr_temp = pulse_dict[coord][resp_coord]

                    plt.subplot(len(coords_to_use), len(coords_to_use), subplot_idx)
                    plt.plot(tvec, curr_temp)
                    plt.plot(tvec, fit_dict[coord][resp_coord])
                    if(k == 0):
                        plt.title("Impulse amplitude = %d MeV (%s dir)"%(impulse_amp_list[j], coord))
                    if(k == 2):
                        plt.xlabel("Time (s)")
                    else:
                        plt.gca().set_xticks([])
                    if(j == 0):
                        plt.ylabel("Normalized %s amplitude [arb units]"%resp_coord)
                    else:
                        plt.gca().set_yticks([])
                    plt.ylim(-1.1,1.1)
                    if(xrange[0]>0):
                        plt.xlim(xrange)

        plt.subplots_adjust(hspace=0, wspace=0)

    fit_dict['t'] = tvec
    return pulse_dict, fit_dict, fit_vals, norm_dict


def get_average_template_paper(calib_dict, make_plots=False, fit_pars=[], drive_idx=drive_idx, 
                            coords_to_use=['x', 'y', 'z'], xrange=[-1,-1], drive_dict=None,
                            lowpass_dict={'x': -1, 'y': -1, 'z': -1}, norm_vals = [1,1,1], ylims=[10,10,5]):
    ## helper function for making the paper plot of the impulse response templates from calibration
    ## Eventually this can be merged with get_average_template_3D

    pulse_dict = {}
    fit_dict = {}
    fit_vals = {}

    colors_dict = {'x': ['darkblue', 'blue'], 'y': ['darkred', 'red'], 'z': ['k', 'orange']}

    coords_dict = {'x': x_idx, 'y': x_idx+1, 'z': x_idx+2}
    if(not drive_dict):
        drive_dict = {'x': drive_idx, 'y': drive_idx+1, 'z': drive_idx-1}

    if(make_plots):
        fig=plt.figure(figsize=(12, 2.75))

    norm_dict = {}
    impulse_amp_list = []
    for j, coord in enumerate(coords_to_use): ## coord in which the drive is applied
        
        pulse_dict[coord] = {}
        fit_dict[coord] = {}
        fit_vals[coord] = {}

        #choose the smallest impulse amp for each coordinate
        impulse_amp = np.min(list(calib_dict[coord].keys()))
        impulse_amp_list.append(impulse_amp)

        curr_files = calib_dict[coord][impulse_amp]

        for k, resp_coord in enumerate(coords_to_use[j]): ## coord to look at response for
            
            coord_idx = coords_dict[resp_coord]
            drive_coord_idx = drive_dict[coord]

            curr_temp = 0
            ntraces = 0
            for fname in curr_files:

                cdat, attr, _ = get_data(fname)

                if(lowpass_dict[coord] > 0):
                    b,a = sp.butter(3, lowpass_dict[coord]/(attr['Fsamp']/2), btype='lowpass') ## get rid of high freq noise
                    cdat[:,coord_idx] = sp.filtfilt(b,a,cdat[:,coord_idx])

                ## find the impulse times
                drive_file = cdat[:,drive_coord_idx]
                impulse_times, _ = find_crossing_indices(drive_file/np.max(drive_file), 0.5)
                window_length = 4000

                for time_idx in impulse_times:
                    ## make sure the full impulse is contained
                    if(time_idx + window_length/2 >= len(cdat[:,coord_idx]) or 
                    time_idx < window_length/2):
                        continue
                    sidx = int(time_idx-window_length/2)
                    eidx = int(time_idx+window_length/2)
                    curr_temp += cdat[sidx:eidx,coord_idx]
                    ntraces += 1

            curr_temp -= np.median(curr_temp[:int(window_length/4)]) #baseline sub
            curr_temp /= ntraces
            pulse_dict[coord][resp_coord] = curr_temp

            if( resp_coord == coord):
                norm_dict[coord] = np.max(np.abs(curr_temp))
                norm_dict[coord + "_amp"] = impulse_amp

    ## now go back through, normalize, and fit
    ## first fit the diagonal elements
    for j, coord in enumerate(coords_to_use):
        resp_coord = coord
        pulse_dict[coord][resp_coord] /= norm_dict[coord]

        curr_temp = pulse_dict[coord][resp_coord]
        tvec = np.arange(len(curr_temp))/attr['Fsamp']
        if(len(fit_pars)>0):
            bp, bc = curve_fit(impulse_response, tvec, curr_temp, p0=fit_pars[coord])
        else:
            bp = [0,0,0,0]
            bc = np.zeros((len(bp), len(bp)))

        fit_dict[coord][resp_coord] = impulse_response(tvec, *bp)
        fit_vals[coord][resp_coord] = bp

    lp_freq = [2000, 600, 2000]
    if(make_plots):
        ## now fit the off diagonal elements with a cross talk and a cross force term
        for j, coord in enumerate(coords_to_use):
                for k, resp_coord in enumerate(coords_to_use[j]): ## coord to look at response for        

                    subplot_idx = k*len(coords_to_use) + j + 1 
                    curr_temp = pulse_dict[coord][resp_coord]

                    if(coord == 'y'): 
                        b,a = sp.butter(3,np.array([200, 600])/(attr['Fsamp']/2), btype='bandstop')
                        curr_temp = sp.filtfilt(b,a,curr_temp)

                        b,a = sp.butter(3,np.array([1200])/(attr['Fsamp']/2), btype='lowpass')
                        curr_temp = sp.filtfilt(b,a,curr_temp)

                    t0=fit_vals[coord][resp_coord][3]
                    print(t0)

                    plt.subplot(1,3,j+1)
                    plt.plot(tvec-t0, curr_temp*norm_vals[j], 'k') 
                    plt.plot(tvec-t0, fit_dict[coord][resp_coord]*norm_vals[j], colors_dict[coord][1], lw=1)

                    plt.xlabel("Time (s)")
                    if(j==0):
                        plt.ylabel("Amplitude [nm]")

                    if(xrange[0]>-1 and j<2):
                        plt.xlim(xrange)
                    else:
                        plt.xlim(-0.01, 0.05)

                    plt.ylim(-ylims[j],ylims[j])

        plt.subplots_adjust(wspace=0.15)

    fit_dict['t'] = tvec
    return fig

def bandpass_filt(calib_dict, template_dict, time_offset = 0, bandpass=[], notch_list = [], 
                  omega0 = 2*np.pi*40, gamma = 2*np.pi*4, subtract_sine_step=False, pulse_data=False, 
                  cal_fac = 1, wind=30, drive_idx=drive_idx, make_plots=False):
    ## simple time domain correlation between template and data for impulse reconstruction
    filt_dict = {}

    for impulse_amp in calib_dict.keys():

        curr_files = calib_dict[impulse_amp]
        filt_dict[impulse_amp] = []
        off_key = str(impulse_amp) + "_offsets"
        filt_dict[off_key] = []
        
        for j,fname in enumerate(curr_files):
            cdat, attr, _ = get_data(fname)

            xdata = cdat[:,x_idx]

            nyquist = attr['Fsamp']/2

            if(subtract_sine_step): ## remove the impulse caused by the sine wave step from the drive
                step_impulse = predict_step_impulse(cdat, nyquist*2, omega0, gamma, make_plots=False)     
                xdata -= step_impulse

            filtconsts = np.array(bandpass)/nyquist # normalized to Nyquist
            b,a = sp.butter(3,filtconsts, btype='bandpass')
            xdata = sp.filtfilt(b,a,xdata)

            impulse_rise, impulse_fall = find_crossing_indices(cdat[:,drive_idx]/np.max(cdat[:,drive_idx]), 0.5)
            
            impulse_cent = get_impulse_cents(cdat, attr['Fsamp'], time_offset=time_offset, pulse_data=pulse_data, 
                                             drive_idx=drive_idx, drive_freq = 120)

            corr_data = np.abs(xdata)
            corr_vals = []
            corr_idx = []
            idx_offsets = []
            for ic in impulse_cent:
                current_search = corr_data[(ic-wind):(ic+wind)]
                if(len(current_search)==0): continue
                corr_vals.append(np.max(current_search))
                corr_idx.append(ic-wind+np.argmax(current_search))
                idx_offsets.append( np.argmax(current_search) - wind)
            filt_dict[impulse_amp] = np.hstack((filt_dict[impulse_amp], corr_vals))
            filt_dict[off_key] = np.hstack((filt_dict[off_key], idx_offsets))

            if(make_plots and j==0):
                sfac = cal_fac
                plt.figure(figsize=(15,3))
                plt.plot(cdat[:,drive_idx]/np.max(cdat[:,drive_idx])*cal_fac)
                plt.plot(np.abs(xdata)*sfac)
                plt.plot(corr_idx, np.abs(corr_vals)*sfac, 'ro')
                plt.xlim(0,1e5)
                plt.ylim(0,500)
                plt.title(impulse_amp)

    return filt_dict

def correlation_filt(calib_dict, template_dict, f0=40, time_offset=0, bandpass=[], notch_list = [], 
                     omega0 = 2*np.pi*40, gamma = 2*np.pi*4, subtract_sine_step=False, pulse_data=True, 
                     drive_idx=drive_idx, wind=30, make_plots=False):
    ## simple time domain correlation between template and data
    filt_dict = {}

    biggest_pulse = np.max( np.fromiter(template_dict.keys(), dtype=float) )
    curr_template = template_dict[biggest_pulse]

    for impulse_amp in calib_dict.keys():

        curr_files = calib_dict[impulse_amp]
        filt_dict[impulse_amp] = []

        for fname in curr_files:
            cdat, attr, _ = get_data(fname)

            xdata = cdat[:,x_idx]

            nyquist = attr['Fsamp']/2

            if(subtract_sine_step): ## remove the impulse caused by the sine wave step from the drive
                step_impulse = predict_step_impulse(cdat, nyquist*2, omega0, gamma, make_plots=False)     
                xdata -= step_impulse

            ## coarse bandpass pre filter if desired
            if(len(bandpass)>0):
                filtconsts = np.array(bandpass)/nyquist # normalized to Nyquist
                b,a = sp.butter(3,filtconsts, btype='bandpass')
                xdata = sp.filtfilt(b,a,xdata)

            for notch in notch_list:
                filtconsts = np.array(notch)/nyquist # normalized to Nyquist
                b,a = sp.butter(3,filtconsts, btype='bandstop')
                xdata = sp.filtfilt(b,a,xdata)

            corr_data = sp.correlate(xdata, curr_template, mode='same')
            bcorr, acorr = sp.butter(3,f0/(5*nyquist), btype='lowpass')
            corr_data = np.sqrt(sp.filtfilt(bcorr, acorr, corr_data**2))


            impulse_cent = get_impulse_cents(cdat, attr['Fsamp'], time_offset=time_offset, pulse_data=pulse_data, drive_freq = 120, drive_idx=drive_idx)

            corr_vals = []
            corr_idx = []
            for ic in impulse_cent:
                current_search = corr_data[(ic-wind):(ic+wind)]
                corr_vals.append(np.max(current_search))
                corr_idx.append(ic-wind+np.argmax(current_search))
            filt_dict[impulse_amp] = np.hstack((filt_dict[impulse_amp], corr_vals))

            if(make_plots):
                sfac = 1/5
                plt.figure(figsize=(15,3))
                impulse_times, _ = find_crossing_indices(cdat[:,drive_idx]/np.max(cdat[:,drive_idx]), 0.5)
                plt.plot(cdat[:,drive_idx]/np.max(cdat[:,drive_idx]))
                plt.plot(corr_data*sfac)
                plt.plot(corr_idx, np.abs(corr_vals)*sfac, 'ro')
                plt.xlim(0,8e4)
                plt.ylim(0,2)
                plt.title(impulse_amp)
                plt.show()

    return filt_dict

def get_impulse_cents(cdat, fs, time_offset=0, pulse_data=True, drive_freq = 120, drive_idx=drive_idx):
    ## find the impulse times from teh drive waveforms

    xdata = cdat[:,0]

    impulse_cent = []
    if(pulse_data):
        ## search for the pulses
        impulse_rise, impulse_fall = find_crossing_indices(cdat[:,drive_idx]/np.max(cdat[:,drive_idx]), 0.5)
        for impr, impf in zip(impulse_rise, impulse_fall):
            cidx = int((impr+impf)/2) + time_offset
            if(cidx > len(xdata)): break
            impulse_cent.append(cidx)
    else:
        ## assume this is stepped sine wave data and find the times for that
        buffer =  int(5*fs/drive_freq)
        b,a = sp.butter(3, drive_freq/(fs/2), btype='low')
        ddat = cdat[:,drive_idx]
        ddat -= np.median(ddat)
        filtered_drive = sp.filtfilt(b,a,ddat**2)
        impulse_rise, impulse_fall = find_crossing_indices(filtered_drive/np.max(filtered_drive[buffer:-buffer]), 0.5)             
        for impr in impulse_rise:
            ## skip edge effects
            if(impr < buffer  or impr > (len(xdata)-buffer)): continue
            impulse_cent.append( impr + time_offset)

    return impulse_cent

def predict_step_impulse(cdat, fs, omega0, gamma, make_plots=False, drive_idx=drive_idx):
    ## predict the impulse for a stepped sine wave drive given known resonator paramters

    xdata = cdat[:,x_idx]

    drive_dat = cdat[:,drive_idx]
    f, p = sp.welch(drive_dat, fs, nperseg=2**int(np.log(len(drive_dat))))
    gpts = (f > 65) & (f < 500) ## search only reasonable range for drive, skip 60 Hz
    pmax = 1.0*p
    pmax[~gpts] = 0
    drive_freq = f[np.argmax(pmax)]
    #print("Drive frequency is: %.2f Hz"%drive_freq)

    ## get the drive frequency (normalized template)
    ddata = cdat[:,drive_idx]
    ddata -= np.mean(ddata)
    ddata = ddata/np.max(ddata)
    ddata_tilde = np.fft.rfft(ddata)

    drive_wind = 2 ## window around drive frequency to keep
    bpf = [5/(fs/2), 200/(fs/2)]
    b, a = sp.butter(1, bpf, btype='bandpass')
    xdata_drive = sp.filtfilt(b,a,xdata)

    fvec = np.fft.rfftfreq(len(xdata), d=1/fs)

    omega_vec = 2*np.pi*fvec
    xtilde = ddata_tilde/(omega0**2 - omega_vec**2 + 1j*gamma*omega_vec)
    xdrive_inv = np.fft.irfft(xtilde)
    scale_func = lambda xdata, A: A*xdrive_inv

    best_scale, scale_err = curve_fit(scale_func, 0, xdata-np.median(xdata), p0=[1,])

    if(make_plots):
        plt.figure(figsize=(12,5))
        plt.plot(xdata_drive, label="Data")
        plt.plot(best_scale*xdrive_inv, label="Predicted impulse")
        #plt.plot(xdata-np.median(xdata))
        plt.xlim(9000,15000)
        plt.legend()
        plt.show()
    
    return best_scale*xdrive_inv

def optimal_filt(calib_dict, template_dict, noise_dict, pulse_data=True, time_offset=0, 
                 omega0 = 2*np.pi*40, gamma = 2*np.pi*4, cal_fac=1, drive_idx=drive_idx, 
                 subtract_sine_step=False, make_plots=False):
    ## initial implementation of optimal filter (only 1D)
    
    filt_dict = {}

    biggest_pulse = np.max( np.fromiter(template_dict.keys(), dtype=float) )
    curr_template = template_dict[biggest_pulse]

    for impulse_amp in calib_dict.keys():

        curr_files = calib_dict[impulse_amp]

        stilde = np.fft.rfft(curr_template)
        sfreq = np.fft.rfftfreq(len(curr_template),d=1e-4)

        J = np.interp(sfreq, noise_dict['freq'], noise_dict['J'])

        ## sharp bandpass
        #J = np.ones_like(J)*1e20
        #bpts = (sfreq < 1) | (sfreq > 100)
        #bpts = (sfreq < 5) | (sfreq > 100)
        #J[bpts] = 1e20
        phi = stilde/J

        phi_t = np.fft.irfft(phi)
        phi_t = phi_t/np.max(phi_t)

        if(make_plots):
            plt.figure()
            plt.plot(phi_t)
            plt.plot(curr_template)
            plt.show()

        filt_dict[impulse_amp] = []
        off_key = str(impulse_amp) + "_offsets"
        filt_dict[off_key] = []
        for fname in curr_files:
            cdat, attr, _ = get_data(fname)
            fs = attr['Fsamp']
            xdata = cdat[:,x_idx]

            if(subtract_sine_step): ## remove the impulse caused by the sine wave step from the drive
                step_impulse = predict_step_impulse(cdat, fs, omega0, gamma, make_plots=False)     
                xdata -= step_impulse


            corr_data = np.abs(sp.correlate(xdata, phi_t, mode='same'))

            impulse_cent = get_impulse_cents(cdat, fs, time_offset=time_offset, pulse_data=pulse_data, 
                                             drive_idx=drive_idx, drive_freq = 120)

            corr_vals = []
            corr_idx = []
            idx_offsets = []
            wind=30
            for ic in impulse_cent:
                current_search = corr_data[(ic-wind):(ic+wind)]
                corr_vals.append(np.max(current_search))
                corr_idx.append(ic-wind+np.argmax(current_search))
                idx_offsets.append( np.argmax(current_search) - wind)
            filt_dict[impulse_amp] = np.hstack((filt_dict[impulse_amp], corr_vals))
            filt_dict[off_key] = np.hstack((filt_dict[off_key], idx_offsets))

            if(make_plots):
                fstr = str.split(fname,'/')[-1]
                sfac = cal_fac
                plt.figure(figsize=(15,3))
                plt.plot(cdat[:,drive_idx]/np.max(cdat[:,drive_idx]) * cal_fac)
                plt.plot(np.abs(corr_data*sfac) )
                plt.plot(corr_idx, np.abs(corr_vals)*sfac, 'ro')
                plt.xlim(0,3e5)
                #plt.xlim(5000, 7500)
                #plt.xlim(impulse_cent[0]-1000, impulse_cent[0]+1000)
                plt.ylim(0,2000)
                plt.title("opt filt: " + str(impulse_amp) + ", " + fstr)

    return filt_dict, phi_t

def optimal_filt_1D(calib_dict, template_dict, noise_dict, pulse_data=True, time_offset=0, 
                 cal_fac=1, drive_idx=drive_idx, do_lp_filt=False, wind=10, 
                 subtract_sine_step=False, make_plots=False, coord='x', resp_coord='', template_fit_vals={},
                 drive_dict=None, paper_plot=False):
    
    ## Implement optimal filter including noise spectrum
    ## this version works only in 1D (x direction)

    filt_dict = {}

    coords_dict = {'x': x_idx, 'y': y_idx, 'z': z_idx}
    
    if(not drive_dict):
        drive_dict = {'x': drive_idx, 'y': drive_idx+1, 'z': drive_idx-1}

    if(len(resp_coord)==0): ## if we don't specify, assume we want in same direction as drive
        resp_coord = coord

    curr_template = template_dict[coord][coord]
    
    ## need to roll this template to start at zero or else we will have an offset
    nsamp_before = np.where(np.abs(curr_template)>0)[0][0]
    curr_template = np.roll(curr_template,-nsamp_before + time_offset)

    for amp_idx, impulse_amp in enumerate(calib_dict[coord].keys()):
        #if(amp_idx>0): break

        curr_files = calib_dict[coord][impulse_amp]
        cdat, attr, _ = get_data(curr_files[0])
        Npts = len(cdat[:,x_idx])

        ## zero pad the current_template
        curr_template = np.hstack((curr_template, np.zeros(Npts-len(curr_template))))

        stilde = np.conjugate(np.fft.rfft(curr_template))
        sfreq = np.fft.rfftfreq(len(curr_template),d=1/attr['Fsamp'])
        J = np.interp(sfreq, noise_dict[coord]['freq'], noise_dict[coord]['J'])

        gpts = J < 1e10 ## points to use in sum
        
        prefac = stilde/J

        filt_dict[impulse_amp] = []
        off_key = str(impulse_amp) + "_offsets"
        filt_dict[off_key] = []
        for fnidx, fname in enumerate(curr_files):
            #print("Working on file: %s"%fname)
            cdat, attr, _ = get_data(fname)
            fs = attr['Fsamp']
            nyquist = fs/2
            xdata = cdat[:,coords_dict[resp_coord]]
            
            if(subtract_sine_step): ## remove the impulse caused by the sine wave step from the drive
                omega0, gamma = template_fit_vals[coord][coord][1], template_fit_vals[coord][coord][2]
                step_impulse = predict_step_impulse(cdat, nyquist*2, omega0, gamma, make_plots=True, drive_idx=drive_idx) 

                bf, af = sp.butter(3, [5/nyquist, 200/nyquist], btype='bandpass')
                tvec = np.arange(Npts)/attr['Fsamp']
                plt.figure()
                plt.plot(tvec,sp.filtfilt(bf,af,xdata))    
                xdata -= step_impulse
                plt.plot(tvec,sp.filtfilt(bf,af,xdata))    
                plt.xlim(0,10)
                plt.show()

            xtilde = np.fft.rfft(xdata)
            corr_data = np.fft.irfft(prefac * xtilde)       ## by far most efficient to use fft for this time offset
                                                            ## note we have to reverse the vector because of the def'n
                                                            ## of irfft (we could use fft but not with rfft variants)

            ## additional low pass filter, if desired for plotting
            if(do_lp_filt and not paper_plot):
                if(coord == 'x'):
                    ff = [50]
                elif(coord == 'y'):
                    ff = [200]
                elif(coord == 'z'):
                    ff = [500]

                if(len(ff)==1):
                    b_lp, a_lp = sp.butter(3, ff/nyquist, btype='lowpass') ## optional low pass for opt filt
                else:
                    b_lp, a_lp = sp.butter(3, ff/nyquist, btype='bandpass')
                    
                corr_data = sp.filtfilt(b_lp, a_lp, corr_data)

            elif(do_lp_filt and paper_plot):
                b_lp, a_lp = sp.butter(3, 20/nyquist, btype='lowpass') ## optional low pass for opt filt
                b_lpz, a_lpz = sp.butter(3, 200/nyquist, btype='lowpass') ## optional low pass for opt filt

                if(coord in ['x', 'y']):
                    corr_data = np.sqrt(sp.filtfilt(b_lp, a_lp, np.abs(corr_data)**2))
                else:
                    corr_data = np.sqrt(sp.filtfilt(b_lpz, a_lpz, np.abs(corr_data)**2))

            impulse_cent = get_impulse_cents(cdat, fs, time_offset=0, pulse_data=pulse_data, 
                                             drive_idx=drive_dict[coord], drive_freq = 120)

            corr_vals = []
            corr_idx = []
            idx_offsets = []
            for ic in impulse_cent:
                icd = int(ic)
                sidx, eidx = icd-wind, icd+wind-1
                if(sidx < 0): sidx = 0
                if(eidx  >= len(corr_data)): eidx = len(corr_data)-1
                current_search = corr_data[sidx:eidx]
                corr_vals.append(np.max(current_search))
                corr_idx.append(sidx+np.argmax(current_search))
                idx_offsets.append( np.argmax(current_search) - wind)
            filt_dict[impulse_amp] = np.hstack((filt_dict[impulse_amp], corr_vals))
            filt_dict[off_key] = np.hstack((filt_dict[off_key], idx_offsets))
            
            if(len(corr_vals)==0):
                print("No impulses found for file: %s"%fname)
                continue

            if(make_plots and fnidx==0):
                corr_vals = np.array(corr_vals)
                tvec = np.arange(Npts)/attr['Fsamp']
                fstr = str.split(fname,'/')[-1]
                sfac = cal_fac
                plt.figure(figsize=(15,3))
                gpts = ~np.isnan(cdat[:,drive_dict[coord]])
                nfac = 1/np.max(cdat[gpts,drive_dict[coord]])
                plt.plot(tvec, cdat[:,drive_dict[coord]]*nfac * 250)
                plt.plot(tvec, corr_data*sfac )
                plt.plot(tvec[corr_idx], np.array(corr_vals)*sfac, 'ro')
                plt.xlim(0, 2)
                gpts = ~np.isnan(corr_vals)
                plt.ylim(-50,np.max([250, np.median(np.abs(corr_vals[gpts])*sfac)*2]))
                plt.title("opt filt: " + str(impulse_amp) + ", " + fstr)

    return filt_dict

# def force_recon_1D(calib_dict, res_params, pulse_data=True, time_offset=0, 
#                  cal_fac=1, drive_idx=drive_idx, do_lp_filt=False, wind=10, 
#                  subtract_sine_step=False, make_plots=False, coord='x', resp_coord='', template_fit_vals={},
#                  drive_dict=None):
#     ## optimally filter including noise spectrum
#     filt_dict = {}

#     coords_dict = {'x': x_idx, 'y': y_idx, 'z': z_idx}
    
#     if(not drive_dict):
#         drive_dict = {'x': drive_idx, 'y': drive_idx+1, 'z': drive_idx-1}

#     if(len(resp_coord)==0): ## if we don't specify, assume we want in same direction as drive
#         resp_coord = coord


#     for amp_idx, impulse_amp in enumerate(calib_dict[coord].keys()):
#         #if(amp_idx>0): break

#         curr_files = calib_dict[coord][impulse_amp]

#         filt_dict[impulse_amp] = []
#         off_key = str(impulse_amp) + "_offsets"
#         filt_dict[off_key] = []
#         for fnidx, fname in enumerate(curr_files):
#             #print("Working on file: %s"%fname)
#             cdat, attr, _ = get_data(fname)
#             fs = attr['Fsamp']
#             nyquist = fs/2
#             xdata = cdat[:,coords_dict[resp_coord]]
            
#             if(subtract_sine_step): ## remove the impulse caused by the sine wave step from the drive
#                 omega0, gamma = template_fit_vals[coord][coord][1], template_fit_vals[coord][coord][2]
#                 step_impulse = predict_step_impulse(cdat, nyquist*2, omega0, gamma, make_plots=True, drive_idx=drive_idx) 

#                 bf, af = sp.butter(3, [5/nyquist, 200/nyquist], btype='bandpass')
#                 tvec = np.arange(Npts)/attr['Fsamp']
#                 plt.figure()
#                 plt.plot(tvec,sp.filtfilt(bf,af,xdata))    
#                 xdata -= step_impulse
#                 plt.plot(tvec,sp.filtfilt(bf,af,xdata))    
#                 plt.xlim(0,10)
#                 plt.show()


#             f0 = res_params[coord][coord][1]/(2*np.pi)
#             gamma = res_params[coord][coord][2]/(2*np.pi)
#             if(coord in ['x', 'y', 'z']):
#                 filt_freqs = [10, 20]

#             pars = [res_params[coord][coord][1], res_params[coord][coord][2]]
#             corr_data = position_to_force(cdat[:,coords_dict[coord]], pars, fs, filt_freqs)

#             if(do_lp_filt):
#                 if(resp_coord == 'z'):
#                     b_lp, a_lp = sp.butter(3, 200/nyquist, btype='lowpass') ## optional low pass for opt filt
#                 else:
#                     b_lp, a_lp = sp.butter(3, 20/nyquist, btype='lowpass') ## optional low pass for opt filt

#                 corr_data = np.sqrt(sp.filtfilt(b_lp, a_lp, np.abs(corr_data)**2))

#             impulse_cent = get_impulse_cents(cdat, fs, time_offset=0, pulse_data=pulse_data, 
#                                              drive_idx=drive_dict[coord], drive_freq = 120)

#             corr_vals = []
#             corr_idx = []
#             idx_offsets = []
#             for ic in impulse_cent:
#                 icd = int(ic)
#                 sidx, eidx = icd-wind, icd+wind-1
#                 if(sidx < 0): sidx = 0
#                 if(eidx  >= len(corr_data)): eidx = len(corr_data)-1
#                 current_search = np.abs(corr_data[sidx:eidx])
#                 corr_vals.append(np.max(current_search))
#                 corr_idx.append(sidx+np.argmax(current_search))
#                 idx_offsets.append( np.argmax(current_search) - wind)
#             filt_dict[impulse_amp] = np.hstack((filt_dict[impulse_amp], corr_vals))
#             filt_dict[off_key] = np.hstack((filt_dict[off_key], idx_offsets))

#             if(make_plots and fnidx==0):
#                 Npts = len(corr_data)
#                 tvec = np.arange(Npts)/attr['Fsamp']
#                 fstr = str.split(fname,'/')[-1]
#                 sfac = cal_fac
#                 plt.figure(figsize=(15,3))
#                 nfac = 1/np.max(cdat[:,drive_dict[coord]])
#                 plt.plot(tvec, cdat[:,drive_dict[coord]]*nfac * 250)
#                 #plt.plot(tvec[impulse_cent], cdat[impulse_cent,drive_idx]*nfac, 'ro')
#                 plt.plot(tvec, corr_data*sfac )
#                 plt.plot(tvec[corr_idx], np.array(corr_vals)*sfac, 'ro')
#                 plt.xlim(0, 10)
#                 #plt.xlim(impulse_cent[0]-1000, impulse_cent[0]+1000)
#                 plt.ylim(-50,500) #np.median(np.abs(corr_vals)*sfac)*2)
#                 plt.title("opt filt: " + str(impulse_amp) + ", " + fstr)

#     return filt_dict


def optimal_filt_3D(calib_dict, template_dict, noise_dict, pulse_data=True, time_offset=0, 
                 drive_idx=drive_idx, make_plots=False, coord=['x','y','z'], noise_scale = [1,1,1],
                 drive_dict=None):
    
    ## Implement optimal filter including noise spectrum
    ## this version works in 3D (x, y, z directions), accounting for crosstalk
    ## between response directions

    filt_dict = {}

    coords_dict = {'x': x_idx, 'y': y_idx, 'z': z_idx}
    
    if(not drive_dict):
        drive_dict = {'x': drive_idx, 'y': drive_idx+1, 'z': drive_idx-1}

    ## need to roll this template to start at zero or else we will have an offset
    nsamp_before = np.where(np.abs(template_dict['x']['x'])>0)[0][0]

    impulse_amp = list(calib_dict[coord[0]].keys())
    curr_files = calib_dict[coord[0]][impulse_amp[0]]
    cdat, attr, _ = get_data(curr_files[0])
    Npts = len(cdat[:,x_idx])
    sfreq = np.fft.rfftfreq(Npts,d=1/attr['Fsamp'])

    ## first assemble the required matrix, M
    M = np.zeros((len(coord), len(coord)))
    ct_dict = {}
    cj_dict = {}
    for i,drive_coord in enumerate(coord):
        ct_dict[drive_coord] = {}
        for j,resp_coord in enumerate(coord):
            
            curr_template = template_dict[drive_coord][resp_coord]
            ## zero pad the current_template
            curr_template = np.hstack((curr_template, np.zeros(Npts-len(curr_template))))
            curr_template = np.roll(curr_template,-nsamp_before + time_offset)
            stilde = np.conjugate(np.fft.rfft(curr_template))
            ct_dict[drive_coord][resp_coord] = stilde

            if(i == 0):
                J = np.interp(sfreq, noise_dict[resp_coord]['freq'], noise_dict[resp_coord]['J'])
                cj_dict[resp_coord] = J*noise_scale[j]

    
    ## now that we have all the templates, loop back over to make the matrix
    for i,drive_coord in enumerate(coord):
        for j,resp_coord in enumerate(coord):
            curr_mat_el = 0
            ## sum over alpha
            for alpha_coord in coord:
                stemp = np.conjugate(ct_dict[resp_coord][alpha_coord]) * ct_dict[drive_coord][alpha_coord]
                curr_mat_el += np.real(np.sum(stemp/cj_dict[alpha_coord]))
            M[j,i] = curr_mat_el
    Minv = np.linalg.inv(M)


    ## now process all the files
    for dc_idx, drive_coord in enumerate(coord):

        #if(dc_idx != 0): continue

        filt_dict[drive_coord] = {}
        for resp_coord in coord:
            filt_dict[drive_coord][resp_coord] = {}

        ## now loop over files
        for amp_idx, impulse_amp in enumerate(calib_dict[drive_coord].keys()):

            curr_files = calib_dict[drive_coord][impulse_amp]

            filt_dict[drive_coord][impulse_amp] = []
            off_key = str(impulse_amp) + "_offsets"
            filt_dict[drive_coord][off_key] = []
            for resp_coord in coord:
                filt_dict[drive_coord][resp_coord][impulse_amp] = []

            for fname in curr_files:
                #print("Working on file: %s"%fname)
                cdat, attr, _ = get_data(fname)
                fs = attr['Fsamp']
                
                ## rows are coordinates, columns freq idx
                n = len(cdat[:,0])
                nfft_pts = int((n/2)+1) if n%2==0 else int((n+1)/2)
                coord_vec = np.zeros((len(coord), nfft_pts), dtype=complex)
                for coord_idx in range(len(coord)):
                    coord_vec[coord_idx, :] = np.fft.rfft(cdat[:,coords_dict[coord[coord_idx]]])
                
                ## now assemble the b vector by looping over the coordinates
                btilde = np.zeros_like(coord_vec, dtype=complex)            
                for rc_idx, resp_coord in enumerate(coord):
                    for alpha_coord in coord:
                        #if(alpha_coord != resp_coord): continue

                        xtemp = coord_vec[rc_idx,:] * ct_dict[resp_coord][alpha_coord]
                        btilde[rc_idx, :] += xtemp/cj_dict[alpha_coord]

                ## now inverse FFT back
                b = np.fft.irfft(btilde, axis=1) 

                ## finally to get the optimal amplitudes, then multiple by inverse of M:
                corr_matrix = Minv @ b
                
                impulse_cent = get_impulse_cents(cdat, fs, time_offset=0, pulse_data=pulse_data, 
                                                drive_idx=drive_dict[drive_coord], drive_freq = 120)

                corr_data = np.abs(corr_matrix[dc_idx,:])

                corr_vals = []
                corr_idx = []
                idx_offsets = []
                wind=10
                for ic in impulse_cent:
                    icd = int(ic)
                    sidx, eidx = icd-wind, icd+wind
                    if(sidx < 0): sidx = 0
                    if(eidx  >= len(corr_data)): eidx = len(corr_data)-1
                    current_search = np.abs(corr_data[sidx:eidx])
                    corr_vals.append(np.max(current_search))
                    corr_idx.append(sidx+np.argmax(current_search))
                    idx_offsets.append( np.argmax(current_search) - wind)

                ## orig vals
                filt_dict[drive_coord][impulse_amp] = np.hstack((filt_dict[drive_coord][impulse_amp], corr_vals))
                filt_dict[drive_coord][off_key] = np.hstack((filt_dict[drive_coord][off_key], idx_offsets))

                for rc_idx, resp_coord in enumerate(coord):
                    amp_vals = corr_matrix[rc_idx, corr_idx]
                    filt_dict[drive_coord][resp_coord][impulse_amp] = np.hstack((filt_dict[drive_coord][resp_coord][impulse_amp], amp_vals))

                if(make_plots):
                    tvec = np.arange(Npts)/attr['Fsamp']
                    fstr = str.split(fname,'/')[-1]

                    plt.figure(figsize=(15,10))
                    for figidx in range(len(coord)):
                        plt.subplot(len(coord), 1, figidx+1)
                        nfac = 1/np.max(cdat[:,drive_dict[drive_coord]])
                        sfac = 1/np.median(np.abs(corr_matrix[dc_idx,corr_idx]))
                        plt.plot(tvec, cdat[:,drive_dict[drive_coord]]*nfac)
                        plt.plot(tvec, np.abs(corr_matrix[figidx,:])*sfac )
                        plt.plot(tvec[corr_idx], np.abs(corr_matrix[figidx,corr_idx])*sfac, 'ro')
                        plt.xlim(1,2)
                        plt.ylim(0,2)
                        plt.title("opt filt 3D: drive %s resp %s, amp="%(drive_coord,coord[figidx]) + str(impulse_amp) + ", " + fstr)    


    return filt_dict

def plot_impulse_with_recon(data, attributes, opt_filt, xrange=[-1,-1], cal_facs=[1,1], amp_cal_facs=[1,1], 
                            drive_idx=drive_idx, plot_wind=5, charge_wind=5, charge_range=[-1,-1], 
                            ylim_init=600, plot_wind_zoom=0.5, filt_time_offset = 0, figout=None):

    ## old version of plotting function to plot waveforms and optimal filter, along with 
    ## charge reconstruction for a given event

    nyquist =(attributes['Fsamp']/2)

    ## coarse LP filter
    fc_x = np.array([5, 70])/nyquist
    b_x,a_x = sp.butter(3, fc_x, btype='bandpass')
    x_position_data = sp.filtfilt(b_x, a_x, data[:,x_idx])
    y_position_data = sp.filtfilt(b_x, a_x, data[:,x_idx+1])
    z_position_data = sp.filtfilt(b_x, a_x, data[:,x_idx+2])

    ## charge data
    xdata = data[:,x_idx]
    drive_data = data[:,drive_idx]

    fc_drive = np.array([110, 112])/nyquist
    fc_data = np.array([104, 119])/nyquist
    b_data,a_data = sp.butter(3, fc_data, btype='bandpass')
    b_drive,a_drive = sp.butter(3, fc_drive, btype='bandpass')
    tvec = np.arange(len(xdata))/attributes['Fsamp']

    nfine = 2**7
    ncoarse = 2**14

    drive_data_tilde = np.fft.rfft(drive_data)

    drive_data = sp.filtfilt(b_drive,a_drive,drive_data)
    xdata = sp.filtfilt(b_data,a_data,xdata)

    window_fine=sp.windows.hamming(nfine)
    window_coarse=sp.windows.hamming(ncoarse)

    fine_points = range(int(nfine/2),len(xdata),nfine)
    coarse_points = range(int(ncoarse/2),len(xdata),int(ncoarse/2))

    corr_dat_fine = 1.0*np.zeros_like(fine_points)
    corr_dat_coarse = 1.0*np.zeros_like(coarse_points)

    for i, pt in enumerate(fine_points[1:-1]):
        st, en = int(pt - nfine/2), int(pt + nfine/2)
        corr_dat_fine[i+1] = -np.sum(xdata[st:en]*drive_data[st:en]*window_fine)

    for i, pt in enumerate(coarse_points[1:-1]):
        st, en = int(pt - ncoarse/2), int(pt + ncoarse/2)
        corr_dat_coarse[i+1] = -np.sum(xdata[st:en]*drive_data[st:en]*window_coarse)


    ## find the index of any charge changes
    coarse_diff = np.diff(corr_dat_coarse)
    coarse_diff[0] = 0 ## throw out edge effects
    coarse_diff[-1] = 0

    fine_diff = np.diff(corr_dat_fine)
    coarse_diff[0:10] = 0 ## throw out edge effects
    coarse_diff[-10:] = 0

    if( xrange[0] < 0):
        charge_change_idx = np.argmax(np.abs(coarse_diff))

        cent = tvec[coarse_points][charge_change_idx]
        ## now use the finely spaced data
        fine_wind = (tvec[fine_points] > cent - 2*plot_wind_zoom) & (tvec[fine_points] < cent + 2*plot_wind_zoom)
        charge_change_idx_fine = np.argmax(np.abs(fine_diff[fine_wind[:-1]]))
        cent += (tvec[fine_points][fine_wind][charge_change_idx_fine] - np.median(tvec[fine_points][fine_wind]))

        xmin = cent-plot_wind
        xmax = cent+plot_wind
        
        xmin_zoom = cent-plot_wind_zoom
        xmax_zoom = cent+plot_wind_zoom

        charge_range = [xmin_zoom, xmax_zoom]

    else:        
        xmin = xrange[0] 
        xmax = xrange[1] 

        cent = np.mean(xrange)
        xmin_zoom = cent - plot_wind_zoom
        xmax_zoom = cent + plot_wind_zoom
       
    charge_change_idx = np.argmin(np.abs(tvec[coarse_points]-cent))

    charge_before = np.median(corr_dat_coarse[1:charge_change_idx])*cal_facs[1]
    charge_after = np.median(corr_dat_coarse[(charge_change_idx+1):-1])*cal_facs[1]
    
    if(not figout):
        figout = plt.figure(figsize=(21,12))
        
    plt.figure(figout.number)
    coord_dat = [x_position_data, y_position_data, z_position_data]
    range_fac = [1,0.8,0.35]
    xlims = [[0, tvec[-1]], [xmin, xmax], [xmin_zoom, xmax_zoom]]
    coord_labs = ['X [MeV]', 'Y [arb units]', 'Z [arb units]', 'Charge [$e$]']
    for i in range(3):

        corr_data = np.abs(sp.correlate(coord_dat[i], opt_filt, mode='same'))

        for col_idx in range(3):
            sp_idx = 3*i + col_idx + 1
            plt.subplot(4,3,sp_idx)
            bp_data = np.roll(coord_dat[i],-filt_time_offset)*amp_cal_facs[0]
            opt_data = corr_data*amp_cal_facs[1]
            plt.plot(tvec, bp_data, color='k', rasterized=True)
            plt.plot(tvec, opt_data, 'orange', rasterized=True)
            plt.ylim(-ylim_init*range_fac[i],ylim_init*range_fac[i])
            plt.gca().set_xticklabels([])
            y1, y2 = -ylim_init*range_fac[i],ylim_init*range_fac[i]

            ## find max around the pulse time in the X data
            markstyle = ['bo', 'ko']
            if(sp_idx == 3):
                max_vals = []
                max_idxs = []
                for ms,d in zip(markstyle, [opt_data, bp_data]):
                    search_wind = 0.1
                    gpts = (tvec > cent -search_wind) & (tvec < cent+search_wind)
                    vec_for_max = 1.0*np.abs(d)
                    vec_for_max[~gpts] = 0  
                    max_idx = np.argmax(vec_for_max)
                    max_val = np.abs(d[max_idx])
                    plt.plot(tvec[max_idx], d[max_idx], ms, label = "%.1f MeV"%max_val)
                    max_vals.append(np.abs(max_val))
                    max_idxs.append(max_idx)

                plt.legend(loc='upper right')


            if( sp_idx % 3 == 0):
                plt.plot([tvec[max_idxs[0]],tvec[max_idxs[0]]], [y1, y2], 'b:')

            if(col_idx==0):
                plt.fill_between([charge_range[0], charge_range[1]], [y1, y1], [y2, y2], color='blue', alpha=0.4, zorder=100)
                plt.ylabel(coord_labs[i])
            elif(col_idx==1):
                plt.fill_between([charge_range[0], charge_range[1]], [y1, y1], [y2, y2], color='blue', alpha=0.4)
            
            plt.xlim(xlims[col_idx])

    for col_idx in range(3):
        plt.subplot(4, 3, 10 + col_idx)
        plt.plot(tvec[fine_points], corr_dat_fine*cal_facs[0], 'gray', rasterized=True)
        plt.plot(tvec[coarse_points], corr_dat_coarse*cal_facs[1], 'red', rasterized=True)
        if(col_idx==0):
            plt.ylabel(coord_labs[3])
        plt.xlim(xlims[col_idx])

        plt.ylim(charge_before-charge_wind, charge_after+charge_wind)
        plt.grid(True)
        y1, y2 = charge_before-charge_wind, charge_after+charge_wind
        if(col_idx < 2):
            plt.fill_between([charge_range[0], charge_range[1]], [y1, y1], [y2, y2], color='blue', alpha=0.4, zorder=100)
        plt.xlabel("Time [s]")

        if( col_idx == 2):
            plt.plot([tvec[max_idxs[0]],tvec[max_idxs[0]]], [y1, y2], 'b:')


    plt.subplots_adjust( hspace=0.0, left=0.04, right=0.99, top=0.95, bottom=0.05)

    step_params = [max_vals[0], max_vals[1], charge_after-charge_before]
    return step_params


def plot_step_with_alphas(data, attributes, xrange=[-1,-1], cal_facs=[1,1], drive_idx=drive_idx, 
                          plot_wind=3, charge_wind=5, charge_range=[-1,-1], plot_wind_zoom=0.5,
                          drive_freq=111, drive_wind=1, data_wind=7):

    ## plot the alpha detector data aligned with the pulse reconstruction data

    nyquist =(attributes['Fsamp']/2)

    ## charge data
    xdata = data[:,x_idx]
    drive_data = data[:,drive_idx]

    fc_drive = np.array([drive_freq-drive_wind,drive_freq+drive_wind])/nyquist
    fc_data = np.array([drive_freq-data_wind,drive_freq+data_wind])/nyquist
    b_data,a_data = sp.butter(3, fc_data, btype='bandpass')
    b_drive,a_drive = sp.butter(3, fc_drive, btype='bandpass')
    tvec = np.arange(len(xdata))/attributes['Fsamp']

    nfine = 2**7
    ncoarse = 2**14
    drive_data_tilde = np.fft.rfft(drive_data)

    drive_data = sp.filtfilt(b_drive,a_drive,drive_data)
    xdata = sp.filtfilt(b_data,a_data,xdata)

    window_fine=sp.windows.hamming(nfine)
    window_coarse=sp.windows.hamming(ncoarse)

    fine_points = range(int(nfine/2),len(xdata),nfine)
    coarse_points = range(int(ncoarse/2),len(xdata),int(ncoarse/2))

    corr_dat_fine = 1.0*np.zeros_like(fine_points)
    corr_dat_coarse = 1.0*np.zeros_like(coarse_points)

    for i, pt in enumerate(fine_points[1:-1]):
        st, en = int(pt - nfine/2), int(pt + nfine/2)
        corr_dat_fine[i+1] = -np.sum(xdata[st:en]*drive_data[st:en]*window_fine)

    for i, pt in enumerate(coarse_points[1:-1]):
        st, en = int(pt - ncoarse/2), int(pt + ncoarse/2)
        corr_dat_coarse[i+1] = -np.sum(xdata[st:en]*drive_data[st:en]*window_coarse)


    ## find the index of any charge changes
    coarse_diff = np.diff(corr_dat_coarse)
    coarse_diff[0] = 0 ## throw out edge effects
    coarse_diff[-1] = 0

    if( xrange[0] < 0):
        charge_change_idx = np.where(np.abs(coarse_diff) > np.std(coarse_diff)*3)[0]
        ## group any neighbors
        dups = np.where(np.abs(np.diff(charge_change_idx)) == 1)[0]
        charge_change_idx = np.delete(charge_change_idx, dups+1)
        if(len(charge_change_idx) != 1):
            print("Warning, didn't find exactly 1 charge change")
        charge_change_idx = charge_change_idx[0]

        xmin = tvec[coarse_points][charge_change_idx]-plot_wind
        xmax = tvec[coarse_points][charge_change_idx]+plot_wind
        
        xmin_zoom = tvec[coarse_points][charge_change_idx]-plot_wind_zoom
        xmax_zoom = tvec[coarse_points][charge_change_idx]+plot_wind_zoom

    else:        
        xmin = xrange[0] 
        xmax = xrange[1] 

        cent = np.mean(xrange)
        xmin_zoom = cent - plot_wind_zoom
        xmax_zoom = cent + plot_wind_zoom
        charge_change_idx = np.argmin(np.abs(tvec[coarse_points]-cent))

    charge_before = np.median(corr_dat_coarse[1:charge_change_idx])*cal_facs[1]
    charge_after = np.median(corr_dat_coarse[(charge_change_idx+1):-1])*cal_facs[1]
    
    figout = plt.figure(figsize=(21,6))
    coord_dat = [data[:,3]] ## alpha detector trigger data
    range_fac = [1,0.8,0.35]
    xlims = [[0, tvec[-1]], [xmin, xmax], [xmin_zoom, xmax_zoom]]
    coord_labs = [r'$\alpha$ det trigger', 'Y [arb units]', 'Z [arb units]', 'Charge [$e$]']
    for i in range(1):

        for col_idx in range(3):
            sp_idx = 3*i + col_idx + 1
            plt.subplot(2,3,sp_idx)

            plt.plot(tvec, coord_dat[i], color='k', rasterized=True)
            plt.gca().set_xticklabels([])
            y1, y2 = -1,0
            plt.ylim(y1,y2)

            if(col_idx==0):
                plt.fill_between([charge_range[0], charge_range[1]], [y1, y1], [y2, y2], color='blue', alpha=0.4, zorder=100)
                plt.ylabel(coord_labs[i])
            elif(col_idx==1):
                plt.fill_between([charge_range[0], charge_range[1]], [y1, y1], [y2, y2], color='blue', alpha=0.4)
            
            plt.xlim(xlims[col_idx])

    for col_idx in range(3):
        plt.subplot(2, 3, 4 + col_idx)
        plt.plot(tvec[fine_points], corr_dat_fine*cal_facs[0], 'gray', rasterized=True)
        plt.plot(tvec[coarse_points], corr_dat_coarse*cal_facs[1], 'red', rasterized=True)
        if(col_idx==0):
            plt.ylabel(coord_labs[3])
        plt.xlim(xlims[col_idx])

        plt.ylim(charge_before-charge_wind, charge_after+charge_wind)
        plt.grid(True)
        y1, y2 = charge_before-charge_wind, charge_after+charge_wind
        if(col_idx < 2):
            plt.fill_between([charge_range[0], charge_range[1]], [y1, y1], [y2, y2], color='blue', alpha=0.4, zorder=100)
        plt.xlabel("Time [s]")


    plt.subplots_adjust( hspace=0.0, left=0.04, right=0.99, top=0.95, bottom=0.05)
    return figout

def calc_expected_spectrum(noise_vals, make_plots=False, docut=False):
    ## plots of expected spectrum vs D.O.F.
    ## this simple estimate has been superseded by the more detailed monte carlo
    ## with SRIM
    npts = int(1e6)
    phi = 2*np.pi*np.random.rand(npts)
    theta = np.arccos(2*np.random.rand(npts)-1)

    ## beta, Po212, Bi212
    decay_momenta = [0, 265, 220] ## MeV
    branching_fractions = [0, 0.27, 0.15] ## fraction of decays  # beta was 0.58
    labels = [r'$\beta$', '$^{212}$Po', '$^{212}$Bi']

    ndof = [1,2,3]
    epdf_ndof = []
    epdf_1 = []

    if(make_plots):
        plt.figure(figsize=(15,4))
    for k,nd in enumerate(ndof):
        bins = np.linspace(0, 500, 100)
        htot = np.zeros(len(bins)-1)
        hlist = []
        for j,dm in enumerate(decay_momenta):

            Ax = dm*np.cos(phi)*np.sin(theta) + noise_vals[0]*np.random.randn(npts)
            Ay = dm*np.sin(phi)*np.sin(theta) + noise_vals[1]*np.random.randn(npts)
            Az = dm*np.cos(theta) + noise_vals[2]*np.random.randn(npts)

            if(nd==1):
                curr_vals = Ax
                if(docut):
                    gpts = (np.abs(Ay) < 2*noise_vals[1]) & (np.abs(Az) < 2*noise_vals[2]) 
                    curr_vals = curr_vals[gpts]
            elif(nd==2):
                curr_vals = np.sqrt(Ax**2 + Ay**2) 
                if(docut):
                    gpts = (np.abs(Az) < 2*noise_vals[2])
                    curr_vals = curr_vals[gpts]
            else:
                curr_vals = np.sqrt(Ax**2 + Ay**2 + Az**2)
                if(docut):
                    gpts = curr_vals > 0
                    curr_vals = curr_vals[gpts]

            hh, be = np.histogram(curr_vals, bins=bins)
            hlist.append(branching_fractions[j]*hh)
            htot += branching_fractions[j]*hh

            if(nd==1):
                epdf_1.append(hh)


        norm = np.sum(htot)
        htot /= norm

        bc = bins[:-1] + 0.5*np.diff(bins)

        gpts = bc > 200


        if(make_plots):
            plt.subplot(1,3,k+1)
            plt.plot(bc, htot, 'k', label='Total')

            for j,hh in enumerate(hlist):
                plt.plot(bc, hh/norm, label=labels[j])

            plt.xlabel("Reconstructed momentum [MeV]")
            plt.ylabel("Probability density [MeV$^{-1}$]")
            plt.xlim(0,400)
            plt.ylim(0, 1.5*np.max(htot[gpts]))
        exp_bins = bc
        exp_pdf = htot
        epdf_ndof.append(exp_pdf)

    return exp_bins, epdf_ndof, epdf_1

def get_edges_from_livetime_vec(live_vec, time_hours, dp_edges_orig):
    ## take a boolean vector with livetimes and return the edges of cut out windows
    thresh = 0.005 ## hrs
    time_separation = np.diff(live_vec)
    dead_periods = np.where(time_separation > thresh)[0] ## count any deadtime > 5 seconds

    dead_period_edges = []
    for j,dp in enumerate(dead_periods):
        do_skip=False
        for dp_orig in dp_edges_orig:
            if(live_vec[dp] >= dp_orig[0]-thresh and live_vec[dp] <= dp_orig[1]+thresh): 
                do_skip = True
                break
        if(do_skip): continue
        dead_period_edges.append([live_vec[dp], live_vec[dp+1]])
    dead_period_edges.append([live_vec[-1], time_hours[-1]])

    return dead_period_edges

def stepped_sine_response(A, B, drive, omega0, gamma, omega_vec, Nvec):
    svec_tilde = np.fft.rfft( (A + (B-A)*Nvec)*drive )
    xtilde = svec_tilde/(omega0**2 - omega_vec**2 + 1j*gamma*omega_vec)
    xdrive_inv = np.fft.irfft(xtilde)
    return xdrive_inv

def remove_outliers_iteratively(data, threshold=3, max_iterations=10, convergence_threshold=0.1):
    """
    Iteratively remove outliers from a time stream until standard deviation converges.

    Parameters:
    - data: numpy array, list, or pandas Series
        The time stream data.
    - threshold: float, optional (default=3)
        The Z-score threshold beyond which data points are considered outliers.
    - max_iterations: int, optional (default=100)
        The maximum number of iterations to perform.
    - convergence_threshold: float, optional (default=1e-6)
        The threshold for standard deviation convergence.

    Returns:
    - cleaned_data: numpy array
        Time stream data with outliers removed.
    """

    # Convert data to a numpy array if it's not already
    data = np.array(data)

    for iteration in range(max_iterations):
        # Calculate Z-scores for the data
        z_scores = zscore(data)

        # Identify outliers based on the threshold
        outliers = np.abs(z_scores) > threshold

        # Remove outliers
        cleaned_data = data[~outliers]

        # Check for convergence
        if iteration > 0:
            prev_std = np.std(data)
            current_std = np.std(cleaned_data)
            if abs(prev_std - current_std) < convergence_threshold*current_std:
                #print(f"Converged after {iteration + 1} iterations.")
                break

        # Update data for the next iteration
        data = cleaned_data
    m,s = np.median(cleaned_data), np.std(cleaned_data)

    return m, s

def fit_prepulse_baseline(data, nyquist, t, noise_dict, coord_list = ['x', 'y', 'z'], prepulse_fig = None):
    ## see if we should modify the template based on the prepulse
    fc_x = np.array([5, 70])/nyquist
    b_x,a_x = sp.butter(3, fc_x, btype='bandpass')

    range_dict = {'x': [5,115], 'y': [5,200], 'z': [150, 400]}

    out_dict = {}

    if(not prepulse_fig):
        prepulse_fig = plt.figure(figsize=(12, 8))
    
    plt.figure(prepulse_fig.number)

    for j in range(3):
        plt.subplot(3, 2, 1+2*j)
        coord = coord_list[j]
        fc_x = np.array([range_dict[coord][0], range_dict[coord][1]])/nyquist
        b_x,a_x = sp.butter(3, fc_x, btype='bandpass')
        filt_dat = sp.filtfilt(b_x, a_x, data[:,x_idx+j])
        plt.plot(t, filt_dat, 'tab:blue', rasterized=True)
        plt.ylabel("Filtered %s [V]"%coord)
        if(coord=='z'):
            plt.xlabel("Time [s]")

        ## find the pulses and zero out
        mu, st = remove_outliers_iteratively(filt_dat)
        ppts = np.abs(filt_dat-mu) < 5*st
        wind = 0.5
        ppts_wind = ppts.copy()
        for bad_times in np.where(~ppts)[0]:
            ppts_wind = ppts_wind & ~((t > t[bad_times]-wind) & (t < t[bad_times]+wind))
        filt_dat_rem = 1.0*filt_dat
        filt_dat_rem[~ppts_wind] = np.nan
        plt.plot(t, filt_dat_rem, 'orange')

        ## now step through and make the PSD
        nperseg = 2**14
        nsteps = int(len(t)/nperseg)
        freqs = np.fft.rfftfreq(nperseg, d=1/(2*nyquist))
        curr_psd = np.zeros(len(freqs))
        ngood = 0
        hann = sp.windows.hann(nperseg)
        for n in range(nsteps):
            curr_data = data[:,x_idx+j][(n*nperseg):((n+1)*nperseg)]
            if np.any(np.isnan(curr_data)):
                continue
            curr_psd += np.abs(np.fft.rfft(curr_data*hann))**2
            ngood += 1
        curr_psd /= ngood
        norm = 2*t[nperseg]/nperseg**2
        curr_psd *= norm ## PSD units of V^2/Hz

        plt.subplot(3, 2, 2+2*j)
        freqs, psd = sp.welch(data[:,x_idx+j], fs=nyquist*2, nperseg=2**14)
        J = np.interp(freqs, noise_dict[coord]['freq'], noise_dict[coord]['J'])
        plt.semilogy(freqs, J, 'k', label="Calib. noise")
        plt.semilogy(freqs, psd, 'tab:blue', label="Noise (with bursts)")
        plt.semilogy(freqs, curr_psd, 'orange', label="Prepulse noise (no bursts)")
        plt.xlim(range_dict[coord][0],range_dict[coord][1])
        gpts = (freqs > range_dict[coord][0]) & (freqs < range_dict[coord][1])
        plt.ylim(0.1*np.percentile(curr_psd[gpts], 5), 10*np.percentile(curr_psd[gpts], 95))
        plt.ylabel("%s PSD [V$^2$/Hz]"%coord)
        if(coord=='z'):
            plt.xlabel("Time [s]")

        out_dict[coord] = dict(f=freqs, psd=curr_psd)
    
    return out_dict


def plot_impulse_with_recon_3D(data, attributes, template_dict, noise_dict, xrange=[-1,-1], cal_facs=[1,1], amp_cal_facs=[], 
                            drive_idx=drive_idx, plot_wind=5, charge_wind=5, charge_range=[-1,-1], do_lowpass=False, 
                            ylim_init=[-10,50], ylim2_scale=4.5, plot_wind_zoom=0.30, filt_time_offset = 0, figout=None, 
                            filament_col=12, toffset=0, tmax=-1, subtract_sine_step=False, res_pars=[0,0], ylim_nm=[-20,32], 
                            ylim_nm_z=[-7.5,32], filt_charge_data = False, field_cal_fac=1, do_subtract_plots=False, figsub=[],
                            plot_wind_offset=0, paper_plot=False, rasterized=False, plot_peak=False, fit_prepulse=False, 
                            prepulse_fig=[], drive_freq=111, drive_wind=1, data_wind=7, search_wind=0.1, col_to_use=[0,1]):

    ## helper function to plot the reconstructed optimally filtered pulses along with the data

    coord_list = ['x', 'y', 'z']
    nyquist =(attributes['Fsamp']/2)

    curr_step_params = {}

    ## coarse LP filter
    fc_x = np.array([5, 70])/nyquist
    b_x,a_x = sp.butter(3, fc_x, btype='bandpass')
    x_position_data = sp.filtfilt(b_x, a_x, data[:,x_idx])
    fc_y = np.array([5, 100])/nyquist
    b_y,a_y = sp.butter(3, fc_y, btype='bandpass')
    y_position_data = sp.filtfilt(b_y, a_y, data[:,x_idx+1])
    fc_z = np.array([100, 400])/nyquist
    b_z,a_z = sp.butter(3, fc_z, btype='bandpass')
    z_position_data = sp.filtfilt(b_z, a_z, data[:,x_idx+2])

    b_lp, a_lp = sp.butter(3, 20/nyquist, btype='lowpass') ## optional low pass for opt filt
    b_lpz, a_lpz = sp.butter(3, 200/nyquist, btype='lowpass') ## optional low pass for opt filt

    ## charge data
    xdata = data[:,x_idx]
    drive_data = data[:,drive_idx]

    fc_drive = np.array([drive_freq-drive_wind,drive_freq+drive_wind])/nyquist
    fc_data = np.array([drive_freq-data_wind,drive_freq+data_wind])/nyquist
    b_data,a_data = sp.butter(3, fc_data, btype='bandpass')
    b_drive,a_drive = sp.butter(3, fc_drive, btype='bandpass')
    tvec = np.arange(len(xdata))/attributes['Fsamp'] - toffset

    nfine = 2**7
    ncoarse = 2**14

    drive_data = sp.filtfilt(b_drive,a_drive,drive_data)

    xdata = sp.filtfilt(b_data,a_data,xdata)

    window_fine=sp.windows.hamming(nfine)
    window_coarse=sp.windows.hamming(ncoarse)

    fine_points = range(int(nfine/2),len(xdata),nfine)
    coarse_points = range(int(ncoarse/2),len(xdata),int(ncoarse/2))

    corr_dat_fine = 1.0*np.zeros_like(fine_points)
    corr_dat_coarse = 1.0*np.zeros_like(coarse_points)

    for i, pt in enumerate(fine_points[1:-1]):
        st, en = int(pt - nfine/2), int(pt + nfine/2)
        corr_dat_fine[i+1] = -np.sum(xdata[st:en]*drive_data[st:en]*window_fine)

    for i, pt in enumerate(coarse_points[1:-1]):
        st, en = int(pt - ncoarse/2), int(pt + ncoarse/2)
        corr_dat_coarse[i+1] = -np.sum(xdata[st:en]*drive_data[st:en]*window_coarse)


    ## find the index of any charge changes
    coarse_diff = np.diff(corr_dat_coarse)
    coarse_diff[0] = 0 ## throw out edge effects
    coarse_diff[-1] = 0

    fine_diff = np.diff(corr_dat_fine)
    coarse_diff[0:10] = 0 ## throw out edge effects
    coarse_diff[-10:] = 0

    if( xrange[0] < 0 and xrange[1] < 0):
        charge_change_idx = np.argmax(np.abs(coarse_diff))

        cent = tvec[coarse_points][charge_change_idx]
        ## now use the finely spaced data
        fine_wind = (tvec[fine_points] > cent - 4*plot_wind_zoom) & (tvec[fine_points] < cent + 4*plot_wind_zoom)
        charge_change_idx_fine = np.argmax(np.abs(fine_diff[fine_wind[:-1]]))
        cent += (tvec[fine_points][fine_wind][charge_change_idx_fine] - np.median(tvec[fine_points][fine_wind]))

        xmin = cent-plot_wind
        xmax = cent+plot_wind
        
        xmin_zoom = cent-plot_wind_zoom
        xmax_zoom = cent+plot_wind_zoom

        charge_range = [xmin_zoom, xmax_zoom]

    else:        
        xmin = xrange[0] 
        xmax = xrange[1] 

        cent = np.mean(xrange)
        xmin_zoom = cent - plot_wind_zoom 
        xmax_zoom = cent + plot_wind_zoom 
       

    charge_change_idx = np.argmin(np.abs(tvec[coarse_points]-cent))

    charge_before = np.median(corr_dat_coarse[1:charge_change_idx])*cal_facs[1]
    charge_after = np.median(corr_dat_coarse[(charge_change_idx+1):-1])*cal_facs[1]
    
    if(fit_prepulse):
        prepulse_noise = fit_prepulse_baseline(data, nyquist, tvec, noise_dict, prepulse_fig=prepulse_fig)
    else:
        prepulse_noise = {}

    if(filt_charge_data):
        fc = np.array([8.7,9.3])
        ny_ch = 0.5/(tvec[fine_points][1]-tvec[fine_points][0])
        b_ch, a_ch = sp.butter(3,fc/ny_ch, btype='bandstop')
        corr_dat_fine = sp.filtfilt(b_ch, a_ch, corr_dat_fine)

    if(not figout):
        figout = plt.figure(figsize=(21,12))
        
    plt.figure(figout.number)
    coord_dat = [x_position_data, y_position_data, z_position_data]
    range_fac = [1,1.5,5]
    ttm = tvec[-1] if tmax < 0 else tmax
    xlims = [[0, ttm], [xmin_zoom, xmax_zoom]]
    coord_labs_pos = ['X pos. [nm]', 'Y pos. [nm]', 'Z pos. [nm]', 'Charge [$e$]']
    coord_labs = ['$A_x$ [MeV/c]', '$A_y$ [MeV/c]', '$A_z$ [MeV/c]', 'Charge, $Q$ [$e$]']
    coord_labs_in = ['$A_x$', '$A_y$', '$A_z$', 'Q']

    fil_vec = (data[:,filament_col]>0.5)
    fil_times = tvec[fil_vec]
    ax2y1, ax2y2 = ylim_init[0]*ylim2_scale, ylim_init[1]*ylim2_scale
    bsfac = 10
    ax_dict = {}
    for i in range(3):

        coord = coord_list[i]
        curr_template = template_dict[coord][coord]
        ## need to roll this template to start at zero or else we will have an offset
        nsamp_before = np.where(np.abs(curr_template)>0)[0][0]
        curr_template = np.roll(curr_template,-nsamp_before)
        Npts = len(data[:,x_idx+i])
        ## zero pad the current_template
        curr_template = np.hstack((curr_template, np.zeros(Npts-len(curr_template))))
        stilde = np.conjugate(np.fft.rfft(curr_template))
        sfreq = np.fft.rfftfreq(len(curr_template),d=1/attributes['Fsamp'])
        J = np.interp(sfreq, noise_dict[coord]['freq'], noise_dict[coord]['J'])
        prefac = stilde/J
        curr_pos_data = data[:,x_idx+i]

        ## subtract sine step if desired
        if(subtract_sine_step):
            if(i==0): ## use x signal for start time

                ## initial guess
                xtilde = np.fft.rfft(curr_pos_data)
                corr_data = np.fft.irfft(prefac * xtilde)
                corr_data *= field_cal_fac ## factor to account for COMSOL simulation of fields


                filt_data = np.sqrt(sp.filtfilt(b_lp, a_lp, corr_data**2)) #lowpass to get best estimate of decay time
                search_wind = 0.1 ## +/- 100 ms
                gpts = (tvec > cent-search_wind) & (tvec < cent+search_wind) & (filt_data>0)
                vec_for_max = 1.0*np.abs(filt_data)
                vec_for_max[~gpts] = 0  
                max_idx_full_waveform = np.argmax(vec_for_max)

                ## fit the drive times two scalings
                ddata = data[:,drive_idx]
                ddata -= np.mean(ddata)
                ddata = ddata/np.max(ddata)
                ddata_tilde = np.fft.rfft(ddata)
                fvec = np.fft.rfftfreq(len(ddata), d=1/attributes['Fsamp'])
                gpts = (fvec > 65) & (fvec < 500) ## search only reasonable range for drive, skip 60 Hz
                pmax = 1.0*np.abs(ddata_tilde)**2
                pmax[~gpts] = 0
                drive_idx = np.argmax(pmax)
                svec_tilde = np.zeros_like(ddata_tilde)
                freq_wind = 1000 ## filter around drive peak
                svec_tilde[(drive_idx-freq_wind):(drive_idx+freq_wind)] = ddata_tilde[(drive_idx-freq_wind):(drive_idx+freq_wind)]
                svec = np.fft.irfft(svec_tilde)

                ## now for the response data
                Npts = len(ddata)
                data_before_step = data[:max_idx_full_waveform,x_idx]
                data_after_step = data[max_idx_full_waveform:,x_idx]
                pts_before, pts_after = 2**int(np.log(len(data_before_step))), 2**int(np.log(len(data_after_step)))
                ft = np.fft.rfft(data_before_step[-pts_before:])
                ft_freq_before = np.fft.rfftfreq(pts_before, d=1/attributes['Fsamp'])
                fidx_before = int(np.round(np.interp(fvec[drive_idx], ft_freq_before, np.arange(len(ft_freq_before)))))
                sign = np.sign( np.real(ft[fidx_before]/svec_tilde[drive_idx]) )
                amp_before = sign*np.abs(ft)[fidx_before]

                ft = np.fft.rfft(data_after_step[:pts_after])
                ft_freq_after = np.fft.rfftfreq(pts_after, d=1/attributes['Fsamp'])
                fidx_after = int(np.round(np.interp(fvec[drive_idx], ft_freq_after, np.arange(len(ft_freq_after)))))
                sign = np.sign( np.real(ft[fidx_after]/svec_tilde[drive_idx]) )
                amp_after = sign*np.abs(ft)[fidx_after]

                omega0, gamma = res_pars[0], res_pars[1]
                omega_vec = 2*np.pi*fvec
                Nvec = np.arange(len(ddata)) > max_idx_full_waveform
        
                fc_wide = np.array([5, 150])/nyquist
                b_wide,a_wide = sp.butter(3, fc_wide, btype='bandpass')

                xdata_drivefilt = sp.filtfilt(b_data, a_data, data[:,x_idx])
                xdata_widefilt = sp.filtfilt(b_wide, a_wide, data[:,x_idx])

                drive_resp = stepped_sine_response(amp_before, amp_after, svec, omega0, gamma, omega_vec, Nvec)
                sfac = np.sum(drive_resp * xdata_drivefilt)/np.sum(drive_resp**2)

                ## subtract from original waveform
                curr_pos_data -= sfac*drive_resp

                cal_fac = 1/amp_cal_facs[0][coord] * 1e9 ## in nm                

                if(do_subtract_plots):
                    subfig=figsub 
                    plt.figure(subfig.number)

                    plt.subplot(2,2,1)
                    plt.plot(tvec, xdata_widefilt*cal_fac, 'k', label='Minimal filtering')
                    plt.plot(tvec, xdata_drivefilt*cal_fac, 'r', label='Filt. to drive freq.')
                    plt.plot(tvec, drive_resp*sfac*cal_fac, 'b', label='Pred. response')
                    plt.xlim(xmin_zoom, xmax_zoom)
                    plt.ylim(-30,30)
                    plt.xlabel("Time (s)")
                    plt.ylabel("X position (nm)")
                    plt.legend(loc='upper right', fontsize=9)

                    before_sub = xdata_widefilt*cal_fac
                    after_sub = (xdata_widefilt-drive_resp*sfac)*cal_fac

                    plt.subplot(2,2,2)
                    plt.plot(tvec, before_sub, 'k', label='Before sub.')
                    plt.plot(tvec, after_sub, 'orange', label='After sub.')
                    plt.xlim(xmin_zoom, xmax_zoom)
                    plt.ylim(-30,30)
                    plt.xlabel("Time (s)")
                    plt.ylabel("X position (nm)")
                    plt.legend(loc='upper right', fontsize=9)

                    plt.subplot(2,2,3)
                    freqs, psd_before = sp.welch(before_sub, fs=attributes['Fsamp'], nperseg=2**14)
                    freqs, psd_after = sp.welch(after_sub, fs=attributes['Fsamp'], nperseg=2**14)
                    plt.semilogy(freqs, psd_before, 'k', label='Before sub.')
                    plt.semilogy(freqs, psd_after, 'orange', label='After sub.')
                    plt.xlim(0,150)
                    gpts = (freqs > 20) & (freqs < 50)
                    plt.ylim(1e-3,2*np.percentile(psd_after[gpts],95))
                    plt.xlabel("Frequency (Hz)")
                    plt.ylabel("PSD (nm$^2$/Hz)")
                    plt.legend(loc='upper right', fontsize=9)

                    plt.subplot(2,2,4)
                    ## initial guess
                    xtilde = np.fft.rfft(xdata_widefilt-drive_resp*sfac)
                    corr_data_after = np.fft.irfft(prefac * xtilde)
                    corr_data_after *= field_cal_fac ## factor to account for COMSOL simulation of fields
                    corr_data_before = corr_data

                    ## low pass if desired
                    if(do_lowpass):
                            corr_data_after = np.sqrt(sp.filtfilt(b_lp, a_lp, corr_data_after**2))
                            corr_data_before = np.sqrt(sp.filtfilt(b_lp, a_lp, corr_data_before**2))

                    plt.plot(tvec, corr_data_before*amp_cal_facs[1][0], 'k', label='Before sub.')
                    plt.plot(tvec, corr_data_after*amp_cal_facs[1][0], 'orange', label='After sub.')
                    plt.xlim(xmin_zoom, xmax_zoom)
                    gpts = (tvec > xmin_zoom) & (tvec < xmax_zoom) & (corr_data_before>0)
                    plt.ylim(-50, np.max(corr_data_before[gpts]*amp_cal_facs[1][0])*2)
                    plt.xlabel("Time (s)")
                    plt.ylabel("X position (nm)")
                    plt.legend(loc='upper right', fontsize=9)            

                    ## back to orig fig
                    plt.figure(figout.number)

        
        ## now the correlation
        if(subtract_sine_step and coord=='x'):
            curr_pos_data = xdata_widefilt-drive_resp*sfac
        xtilde = np.fft.rfft(curr_pos_data)
        corr_data = np.fft.irfft(prefac * xtilde)
        corr_data *= field_cal_fac[i] ## factor to account for COMSOL simulation of fields
        corr_data_orig = 1.0*corr_data

        if(coord=='x'):
            bandpass = [50]
        elif(coord=='y'):
            bandpass = [200]
        elif(coord=='z'):
            bandpass = [500]

        corr_data_unfilt = compute_opt_filt(data, coord, template_dict, noise_dict, attributes['Fsamp'], bandpass=bandpass)   
        if(i < 2):
            b_lp, a_lp = sp.butter(3, np.array([1,20])/nyquist, btype='bandpass') ## optional low pass for opt filt
            gpts = (tvec > xmin_zoom) & (tvec < xmax_zoom)
            filt_data_filt = sp.filtfilt(b_lp, a_lp, np.abs(corr_data_unfilt))
        else:
            b_lp, a_lp = sp.butter(3, np.array([1,500])/nyquist, btype='bandpass') ## optional low pass for opt filt
            filt_data_filt = sp.filtfilt(b_lp, a_lp, np.abs(corr_data_unfilt)) 

        ## low pass if desired
        if(do_lowpass):
            if(i < 2):
                corr_data = np.sqrt(sp.filtfilt(b_lp, a_lp, np.abs(corr_data)**2))
            else:
                corr_data = np.sqrt(sp.filtfilt(b_lpz, a_lpz, np.abs(corr_data)**2))

        for col_idx in range(2):
            if(paper_plot):
                if(col_idx == 0):
                    sp_idx = i + 1
                    plt.subplot(4,1,sp_idx)
                    ax1 = plt.gca()
                else:
                    ax1 = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(.45, .5, .57, .56), bbox_transform=ax.transAxes)
                    ax1.tick_params(axis='x', pad=0, labelsize=9)
                    ax1.tick_params(axis='y', pad=0, labelsize=9)
                    ax1.xaxis.labelpad = 0
                    ax1.yaxis.labelpad = 0
                    ax1.set_ylabel(coord_labs_in[i], fontsize=9)
                lw = 1
            else:
                if(col_idx not in col_to_use):
                    continue

                sp_idx = i + 1
                plt.subplot(4,1,sp_idx)
                ax1 = plt.gca()
                ax1.tick_params(axis='x', pad=0)
                ax1.tick_params(axis='y', pad=0)
                ax1.xaxis.labelpad = 0
                if(i < 2 or col_idx==0):
                    ax1.yaxis.labelpad = 0
                else:
                    ax1.yaxis.labelpad = 20
                lw=1

            bp_data = np.roll(coord_dat[i],-filt_time_offset)/amp_cal_facs[0][coord] * 1e9 ## in nm
            opt_data = filt_data_filt * amp_cal_facs[1][i] * field_cal_fac[i]
            opt_data_orig = corr_data_orig*amp_cal_facs[1][i]
            y1, y2 = ylim_init[0]*range_fac[i],ylim_init[1]*range_fac[i]

            if(col_idx==0):
                ax1.plot(tvec, opt_data, 'k', zorder=0, rasterized=rasterized, lw=lw)

                ## calculate signal to noise
                if(paper_plot and False):
                    hh, be = np.histogram(opt_data, bins=100)
                    bc = be[:-1] + 0.5*np.diff(be)
                    gfit, gcov = curve_fit(gauss_fun, bc, hh, p0=[np.max(hh), np.mean(opt_data), np.std(opt_data)])
                    gpts = (tvec > 16) & (tvec < 17)
                    peak = np.max(opt_data[gpts])
                    print("peak: ", peak, tvec[gpts][np.argmax(opt_data[gpts])])
                    print("sigma: ", gfit[2])
                    plt.close('all')
                    plt.errorbar(bc, hh, yerr=np.sqrt(hh), fmt='o', color='k', label='Data')
                    plt.plot(bc, gauss_fun(bc, *gfit), 'r', label='Gaussian fit')
                    print("SNR: ", peak/gfit[2])
                    plt.show()

            else:
                ax1.plot(tvec, opt_data, 'k', zorder=0, rasterized=rasterized)

            if(i==2):
                plt.ylim(0.2*ax2y1*range_fac[i], ax2y2*range_fac[i])  
            else:
                plt.ylim(ax2y1*range_fac[i], ax2y2*range_fac[i]) 

            for ax in [ax1]: 
                ax.set_xticklabels([])

            if(col_idx==1):
                ax2 = ax1.twinx()
                ax2.plot(tvec, bp_data, color='orange', zorder=1, rasterized=rasterized)
                if(paper_plot):
                    ax2.tick_params(axis='x', pad=0, labelsize=9)
                    ax2.tick_params(axis='y', pad=0, labelsize=9)
                    ax2.xaxis.labelpad = 0
                    ax2.yaxis.labelpad = 4
                    ax2.set_ylabel(coord_labs_pos[i], fontsize=9)
                else:
                    ax2.tick_params(axis='x', pad=0)
                    ax2.tick_params(axis='y', pad=4)
                    ax2.xaxis.labelpad = 0
                    
                    ax2.set_ylabel(coord_labs_pos[i])    
                    
                    if(i < 2 or col_idx==0):
                        ax2.yaxis.labelpad = 2
                    else:
                        ax2.yaxis.labelpad = 10                

                print(ax2y2)
                ax1.set_ylim(ylim_nm[0]/ylim_nm[1] * ax2y2, ax2y2)
                yy = ax2.get_ylim()
                if(i==2):
                    scale_fac = 3 #1.5
                    ax2.set_ylim(-0.7*ylim_nm_z[1]*scale_fac, ylim_nm_z[1]*scale_fac)
                    ax1.set_ylim(-0.7*ax2y2*1.5, ax2y2*1.5)
                    ax1.yaxis.labelpad = -7
                    ax2.set_yticks([0,50])
                else:
                    ax2.set_ylim(ylim_nm[0], ylim_nm[1])

            ## find max around the pulse time in the X data
            markstyle = ['bo', 'ko']
            if(i==0 and col_idx==1):
                max_vals = []
                max_idxs = []
                for ms,d in zip(markstyle, [opt_data, bp_data]):
                    gpts = (tvec > cent -search_wind) & (tvec < cent+search_wind) & (d>0)
                    vec_for_max = 1.0*np.abs(d)
                    vec_for_max[~gpts] = 0  
                    max_idx = np.argmax(vec_for_max)
                    max_val = np.abs(d[max_idx])
                    max_vals.append(np.abs(max_val))
                    max_idxs.append(max_idx)
                    vec_for_max = 1.0*np.abs(opt_data)
                    curr_max_val = vec_for_max[max_idxs[0]]
                if(plot_peak):
                    plt.sca(ax1)
                    plt.plot(tvec[max_idxs[0]], curr_max_val, 'bo', label = "%.1f MeV"%curr_max_val)     
                    plt.legend(loc='upper right')
            
            elif(i > 0 and col_idx==1):
                search_buff=1
                vec_for_max = 1.0*np.abs(opt_data)
                vec_for_max[:(max_idxs[0]-search_buff)] = 0  
                vec_for_max[(max_idxs[0]+search_buff):] = 0  
                curr_max_val = np.max(vec_for_max)
                if(plot_peak):
                    plt.sca(ax1)
                    plt.plot(tvec[max_idxs[0]], curr_max_val, 'bo', label = "%.1f MeV"%curr_max_val)                
                    plt.legend(loc='upper right')

            if(col_idx == 1):
                curr_step_params[coord + "_amp"] = curr_max_val
                curr_step_params[coord + "_time"] = tvec[max_idxs[0]]

            if( col_idx == 1):
                yy = ax1.get_ylim()
                ax1.plot([tvec[max_idxs[0]], tvec[max_idxs[0]]], [yy[0], yy[1]], 'b:')

            if(col_idx == 0):
                ax1.fill_between([xlims[col_idx+1][0]+plot_wind_offset, xlims[col_idx+1][1]+plot_wind_offset], [bsfac*y1, bsfac*y1], [bsfac*y2, bsfac*y2], color='blue', alpha=0.1, zorder=0)
                ax1.set_ylabel(coord_labs[i])
            
            if(col_idx == 1 and not paper_plot):
                ax1.set_ylabel(coord_labs[i])            

            if(len(fil_times)>0):
                ff=np.ones_like(fil_times)
                ax1.fill_between(fil_times, bsfac*y1*ff, y1 + bsfac*ff*(y2-y1), color='red', alpha=0.2)
            
            plt.xlim(xlims[col_idx])

    for col_idx in range(2):

        if(paper_plot):
            if(col_idx == 0):
                plt.subplot(4, 1, 4)
                ax1 = plt.gca()
                outer_ax = ax1
            else:
                ax1 = inset_axes(ax1, width="100%", height="100%", bbox_to_anchor=(.45, 0.35, .57, .4), bbox_transform=plt.gca().transAxes)
                ax1.tick_params(axis='x', pad=0, labelsize=9)
                ax1.tick_params(axis='y', pad=0, labelsize=9)
                ax1.xaxis.labelpad = 0
                ax1.yaxis.labelpad = -5

                rect = Rectangle((-0.2, -0.3), 1, 1.3, transform=ax1.transAxes, color='white', zorder=3)
                outer_ax.add_patch(rect)
                rect2 = Rectangle((0.35, -0.6), 0.3, 0.3, transform=ax1.transAxes, color='white', zorder=3)
                outer_ax.add_patch(rect2)
        else:
            if(col_idx not in col_to_use):
                continue
            plt.subplot(4, 1, 4)
            ax1 = plt.gca()
            ax1.tick_params(axis='x', pad=0)
            ax1.tick_params(axis='y', pad=0)
            ax1.xaxis.labelpad = 0
            ax1.yaxis.labelpad = 0

        plt.sca(ax1)
        if(col_idx==0):
            plt.plot(tvec[fine_points], corr_dat_fine*cal_facs[0], 'gray', rasterized=rasterized, lw=lw) 
        else:
            plt.plot(tvec[fine_points], corr_dat_fine*cal_facs[0], 'gray', rasterized=rasterized) 

        if(col_idx==0):
            plt.plot(tvec[coarse_points], corr_dat_coarse*cal_facs[1], 'red', rasterized=rasterized)
            plt.ylabel(coord_labs[3])

            integer_charge_before = np.round(charge_before)
            integer_charge_after = np.round(charge_after)

            buff=3
            t = tvec[coarse_points][1:(charge_change_idx+buff)]
            plt.plot(t, np.ones_like(t)*integer_charge_before, 'k:', zorder=5)
            text = plt.text(t[-1], integer_charge_before, "%d$e$"%integer_charge_before, ha='left', va='top', fontsize=10, zorder=4)
            text.set_bbox(dict(facecolor='white', edgecolor='none', pad=0))
            t = tvec[coarse_points][(charge_change_idx-buff):]
            plt.plot(t, np.ones_like(t)*integer_charge_after, 'k:', zorder=5)
            text = plt.text(t[0], integer_charge_after, "+%d$e$"%integer_charge_after, ha='right', va='center', fontsize=10, zorder=4)
            text.set_bbox(dict(facecolor='white', edgecolor='none', pad=0))

        plt.xlim(xlims[col_idx])

        plt.ylim(charge_before-charge_wind, charge_after+charge_wind)
        plt.grid(True)
        y1, y2 = charge_before-charge_wind, charge_after+charge_wind
        if(col_idx ==0 ):
            plt.fill_between([charge_range[0]+plot_wind_offset, charge_range[1]+plot_wind_offset], [y1, y1], [y2, y2], color='blue', alpha=0.1, zorder=0)
        
        if(col_idx==1):
            if(paper_plot):
                plt.xlabel("Time [s]", fontsize=9)
                plt.ylabel("$Q$", fontsize=9)
            else:
                plt.xlabel("Time [s]")
                plt.ylabel("Charge, $Q$ [$e$]")
        else:
            plt.xlabel("Time [s]")
            plt.ylabel("Charge, $Q$ [$e$]")

        if( col_idx == 1):
            plt.plot([tvec[max_idxs[0]],tvec[max_idxs[0]]], [y1, y2], 'b:')

        if(len(fil_times)>0):
            plt.fill_between(fil_times, y1*ff, y1 + ff*(y2-y1), color='red', alpha=0.2)

    curr_step_params['charge_before'] = charge_before
    curr_step_params['charge_after'] = charge_after

    curr_step_params['prepulse_noise'] = prepulse_noise

    curr_step_params['t_fine'] = tvec[fine_points]
    curr_step_params['charge_fine'] = corr_dat_fine*cal_facs[0]
    curr_step_params['t_coarse'] = tvec[coarse_points]
    curr_step_params['charge_coarse'] = corr_dat_coarse*cal_facs[1]

    if(paper_plot or col_to_use==[0]):
        figout.align_labels()
        plt.subplots_adjust( hspace=0.0, top=0.98, left=0.15, bottom=0.10, right=0.91 )
    else:
         plt.subplots_adjust( hspace=0.0, top=0.98, left=0.15, bottom=0.10, right=0.86 )       

    return curr_step_params

def pulse_data_dict_to_hist(pd, tcut=[], do_plot=False):
    ## take a dictionary with pulse data and return a histogram

    data_out = []
    for k in pd.keys():
        print(pd[k])
        currx, curry, currz = pd[k]['x_amp'], pd[k]['y_amp'], pd[k]['z_amp']
        qbefore, qafter = pd[k]['charge_before'], pd[k]['charge_after']
        data_out.append([currx, curry, currz, qbefore, qafter, pd[k]['time_hours']])

    data_out = np.array(data_out)

    if(len(tcut)>0):
        cut = (data_out[:,-1]>tcut[0]) & (data_out[:,-1]<tcut[1])
    else:
        cut = np.ones(len(data_out)).astype(bool)
    
    pvals = np.sqrt( data_out[cut,0]**2 + data_out[cut,1]**2 )
    bins = np.linspace(0,5000,50)
    hh, be = np.histogram(pvals, bins=bins)
    bc = be[:-1] + np.diff(be)/2

    if(do_plot):
        plt.figure()
        plt.errorbar(bc, hh, yerr=np.sqrt(hh), fmt='ko')
        plt.show()
    
    return data_out, hh, bc

def position_to_force(data, res_params, fs, filt_freqs=[]):
    ## take pulse data and resonator parameters and convert a position time stream to force
    ## using the susceptibility
    
    nyquist = fs/2
    omega0, gamma = res_params[0], res_params[1]

    xtilde = np.fft.rfft(data)
    fvec = np.fft.rfftfreq(len(data), d=1/fs)

    omega_vec = 2*np.pi*fvec
    Ftilde = xtilde * (omega0**2 - omega_vec**2 + 1j*gamma*omega_vec) * sp.windows.hann(len(xtilde))
    force = np.fft.irfft(Ftilde)

    if(len(filt_freqs)>0):
        fc_f = np.array(filt_freqs)/nyquist
        b_f,a_f = sp.butter(3, fc_f, btype='bandpass')
        force = sp.filtfilt(b_f, a_f, force)

    return force


def compute_opt_filt(data, coord, template_dict, noise_dict, fs, bandpass=[]):

    ## helper function to compute the 1D optimal filter for a given coordinate

    curr_template = template_dict[coord][coord]
    ## need to roll this template to start at zero or else we will have an offset
    nsamp_before = np.where(np.abs(curr_template)>0)[0][0]
    curr_template = np.roll(curr_template,-nsamp_before)
    Npts = len(data[:,coords_dict[coord]])
    ## zero pad the current_template
    curr_template = np.hstack((curr_template, np.zeros(Npts-len(curr_template))))
    stilde = np.conjugate(np.fft.rfft(curr_template))
    sfreq = np.fft.rfftfreq(len(curr_template),d=1/fs)
    J = np.interp(sfreq, noise_dict[coord]['freq'], noise_dict[coord]['J'])
    prefac = stilde/J
    curr_pos_data = data[:,coords_dict[coord]]
    xtilde = np.fft.rfft(curr_pos_data)
    corr_data = np.fft.irfft(prefac * xtilde)

    if(len(bandpass)>0):
        bp = 'bandpass' if len(bandpass)==2 else 'lowpass'
        fc_x = np.array(bandpass)/(fs/2)
        b_x,a_x = sp.butter(3, fc_x, btype=bp)
        corr_data = sp.filtfilt(b_x, a_x, corr_data)

    return corr_data

def get_drive(data, fs, drive_idx=drive_idx):
    ## get the drive used for charge monitoring

    ## fit the drive times two scalings
    ddata = data[:,drive_idx]
    ddata -= np.mean(ddata)
    ddata = ddata/np.max(ddata)
    ddata_tilde = np.fft.rfft(ddata)
    fvec = np.fft.rfftfreq(len(ddata), d=1/fs)
    gpts = (fvec > 65) & (fvec < 500) ## search only reasonable range for drive, skip 60 Hz
    pmax = 1.0*np.abs(ddata_tilde)**2
    pmax[~gpts] = 0
    drive_idx = np.argmax(pmax)
    svec_tilde = np.zeros_like(ddata_tilde)
    freq_wind = 1000 ## filter around drive peak
    svec_tilde[(drive_idx-freq_wind):(drive_idx+freq_wind)] = ddata_tilde[(drive_idx-freq_wind):(drive_idx+freq_wind)]
    svec = np.fft.irfft(svec_tilde)
    omega_vec = 2*np.pi*fvec

    return svec, omega_vec


def pulse_recon(step_params, res_params, template_dict, noise_dict, amp_cal_facs = [1,1], do_bandpass=False, sigma=[1,1,1]):
    ## function to take a second pass at reconstructing the pulses with a refined
    ## implemenation of the optimal filter and drive impulse subtraction

    coord_list = ['x', 'y', 'z']

    pulse_wind = 1
    search_wind = 0.01 #0.05
    pulse_time = step_params['x_time']
    prepulse_samps = 2**13 ## 800 ms

    cdat, attr, _ = get_data(step_params['filename'])
    tvec = np.arange(len(cdat[:,0]))/attr['Fsamp']

    nyquist = attr['Fsamp']/2

    fig=plt.figure(figsize=(20,12))

    coord_data = np.zeros((3, len(cdat[:,0])))

    opt_waveforms = {}
    filt_waveforms = {}

    coord_labs_pos = ['X pos. [nm]', 'Y pos. [nm]', 'Z pos. [nm]', 'Charge [$e$]']
    coord_labs_in = ['$A_x$ [MeV]', '$A_y$ [MeV]', '$A_z$ [MeV]', 'Charge [$e$]']
    renorm = 2 # rescale after lp filt

    random_recon = {}

    right_ax_list = []
    for j, coord in enumerate(coord_list):
        f0 = res_params[coord][coord][1]/(2*np.pi)
        gamma = res_params[coord][coord][2]/(2*np.pi)

        if(coord in ['x', 'y']):
            fc_x = np.array([5, 70])/nyquist
        else:
            fc_x = np.array([100, 500])/nyquist           
        
        b_x, a_x = sp.butter(3, fc_x, btype='bandpass')
        x_filt = sp.filtfilt(b_x, a_x, cdat[:,x_idx+j])

        if(do_bandpass):
            if(coord=='x'):
                bandpass = [50]
            elif(coord=='y'):
                bandpass = [200]
            elif(coord=='z'):
                bandpass = [500]
        else:
            bandpass = []

        filt_data = compute_opt_filt(cdat, coord, template_dict, noise_dict, attr['Fsamp'], bandpass=bandpass)
        filt_data_unfilt = compute_opt_filt(cdat, coord, template_dict, noise_dict, attr['Fsamp'], bandpass=[])        

        b_lp, a_lp = sp.butter(3, np.array([1,20])/nyquist, btype='bandpass') ## optional low pass for opt filt
        filt_data_filt = sp.filtfilt(b_lp, a_lp, np.abs(filt_data_unfilt))


        gpts = (tvec > pulse_time-pulse_wind) & (tvec < pulse_time+pulse_wind)
        norm = amp_cal_facs[1][j]

        plt.subplot(6,1,1+j)
        plt.plot(tvec, filt_data*norm, 'gray')
        #plt.plot(tvec, np.abs(filt_data*norm), 'gray')
        plt.plot(tvec, filt_data_filt*norm*renorm, 'k')
        ax1 = plt.gca()
        ax2 = plt.twinx()
        ax2.plot(tvec, x_filt/amp_cal_facs[0][coord] * 1e9, 'orange')
        ax2.set_ylim([-20,40])
        ax2.set_ylabel(coord_labs_pos[j])
        opt_waveforms[coord] = filt_data*norm
        filt_waveforms[coord] = x_filt/amp_cal_facs[0][coord] * 1e9
        right_ax_list.append(ax2)

        plt.sca(ax1)
        plt.xlim(pulse_time-pulse_wind, pulse_time+pulse_wind)
        ym = 350 #np.max(np.abs(filt_data_filt[gpts]*norm))*2
        plt.ylim(-ym, ym) 
        plt.ylabel(coord_labs_in[j])

        coord_data[j,:] = filt_data * norm * 1/(sigma[j])

    plt.subplot(6,1,4)
    sigma = np.array(sigma)
    norm2 = np.sum(1/sigma**2)
    sum_coord = np.sqrt(np.sum(coord_data**2, axis=0)/norm2)
    plt.plot(tvec, sum_coord, 'k')
    lp_sum_coord = sp.filtfilt(b_lp, a_lp, sum_coord)
    gpts = (tvec > pulse_time-search_wind) & (tvec < pulse_time+search_wind)
    max_location_wind = np.argmax(np.abs(lp_sum_coord[gpts]*sum_coord[gpts]))
    max_idx_overall = np.where(gpts)[0][0] + max_location_wind
    max_loc_overall = tvec[max_idx_overall]

    #search also in 100 random places in the waveform
    rand_locs = np.random.rand(100)*len(sum_coord)
    rand_idxs = []
    for rl in rand_locs:
        gpts = (tvec > tvec[int(rl)]-search_wind) & (tvec < tvec[int(rl)]+search_wind)
        if(np.sum(gpts)<1.9*search_wind*attr['Fsamp']):
            continue
        rand_location_wind = np.argmax(np.abs(lp_sum_coord[gpts]*sum_coord[gpts]))
        rand_idx_overall = np.where(gpts)[0][0] + rand_location_wind
        rand_idxs.append(rand_idx_overall)

    plt.plot(tvec, lp_sum_coord * renorm, 'r')
    plt.xlim(pulse_time-pulse_wind, pulse_time+pulse_wind)
    ym = np.max([350, 1.5*np.abs(sum_coord[max_idx_overall])])
    plt.plot([max_loc_overall, max_loc_overall], [-2*ym, 2*ym], 'b:')
    plt.ylim(-100, ym)
    plt.ylabel(r'$|\vec{A}|$ [MeV]')

    refined_OF_amps = []
    refined_bandpass_amps = []

    ## now go back through and get the amplitudes and prepulse
    buff_samps = int(search_wind*attr['Fsamp'])
    prepulse_wfs = []
    prepulse_psds = []
    prepulse_rms = []
    prepulse_pos_rms = []

    search_samps_pos = int(0.02*attr['Fsamp']) ## 10 ms

    for j, coord in enumerate(coord_list):
        plt.subplot(6,1,1+j)
        plt.plot([max_loc_overall, max_loc_overall], [-2*ym, 2*ym], 'b:')
        max_val = opt_waveforms[coord][max_idx_overall]
        plt.plot(max_loc_overall, max_val, 'bo', label='%.1f MeV'%max_val)
        ym = np.max([350, 1.5*np.abs(max_val)])
        plt.ylim(-ym, ym) 
        plt.legend()

        rand_amps = []
        for ridx in rand_idxs:
            rand_amps.append(opt_waveforms[coord][ridx])

        st, en = int(max_idx_overall-search_samps_pos), int(max_idx_overall+search_samps_pos)
        max_pos_loc = st + np.argmax(np.abs(filt_waveforms[coord][st:en]))
        max_pos = filt_waveforms[coord][max_pos_loc]
        right_ax_list[j].plot(tvec[max_pos_loc], max_pos, 'o', color='orange', label='%.1f nm'%max_pos)
        refined_bandpass_amps.append(max_pos)

        st, en = max_idx_overall-buff_samps-prepulse_samps, max_idx_overall-buff_samps
        refined_OF_amps.append(max_val)
        prepulse_wfs.append(opt_waveforms[coord][st:en])
        freqs, psd = sp.welch(opt_waveforms[coord][st:en], fs=attr['Fsamp'], nperseg=prepulse_samps)
        prepulse_psds.append(psd)
        prepulse_rms.append(np.std(opt_waveforms[coord][st:en]))
        prepulse_pos_rms.append(np.std(filt_waveforms[coord][st:en]))
        random_recon[coord] = rand_amps

    ## now do the subtraction for x
    omega0, gamma = res_params['x']['x'][1], res_params['x']['x'][2]
    amp_before, amp_after = step_params['charge_before'], step_params['charge_after']
    svec, omega_vec = get_drive(cdat, attr['Fsamp'], drive_idx=drive_idx)
    Nvec = np.arange(len(cdat[:,drive_idx])) > max_idx_overall
    drive_resp = stepped_sine_response(amp_before, amp_after, svec, omega0, gamma, omega_vec, Nvec)
    edge_buff = 2000
    amp = np.sum(drive_resp[edge_buff:-edge_buff] * cdat[edge_buff:-edge_buff,x_idx])/np.sum(drive_resp[edge_buff:-edge_buff]**2)

    fc_dr = np.array([104, 118])/nyquist
    b_dr, a_dr = sp.butter(3, fc_dr, btype='bandpass')
    drive_data = sp.filtfilt(b_dr, a_dr, cdat[:,x_idx])


    position_calib_x = 1e9/amp_cal_facs[0][coord]
    plt.subplot(6,1,6)
    plt.plot(tvec, amp*drive_resp*position_calib_x, 'r', label='Pred. response')
    plt.plot(tvec, drive_data*position_calib_x, 'k', label='Data (filt)')
    plt.xlim(pulse_time-pulse_wind, pulse_time+pulse_wind)
    plt.ylabel("Response at\ndrive freq. [nm]")
    plt.legend()

    ## now plot subtracted waveforms
    plt.subplot(6,1,1)
    cdat[:,x_idx] -= amp*drive_resp
    coord, bandpass = 'x', [50]
    norm = amp_cal_facs[1][0]
    filt_data_sub = compute_opt_filt(cdat, coord, template_dict, noise_dict, attr['Fsamp'], bandpass=bandpass)
    plt.plot(tvec, filt_data_sub*norm, 'cyan', zorder=0)
    max_val_sub = filt_data_sub[max_idx_overall]*norm
    plt.plot(tvec[max_idx_overall], max_val_sub, 'o', color='cyan', zorder=0, label='%.1f MeV'%max_val_sub)
    step_params['refined_x_amp_sub'] = max_val_sub
    plt.gca().set_zorder(ax2.get_zorder()+1)
    plt.gca().set_frame_on(False)
    plt.legend()

    prepulse_psd_freq = freqs

    plt.subplot(6,1,5)
    plt.plot(step_params['t_fine'], step_params['charge_fine'], 'r')
    plt.ylabel("Charge [$e$]")
    ax2 = plt.twinx()
    ax2.plot(tvec, sp.filtfilt(b_dr, a_dr, cdat[:, drive_idx]), 'gray')
    ax2.plot(tvec, drive_data, 'k')
    ax2.set_ylabel("Charge drive [arb. units]")
    plt.xlim(pulse_time-pulse_wind, pulse_time+pulse_wind)
    yy = plt.ylim()
    plt.plot([max_loc_overall, max_loc_overall], [-2*ym, 2*ym], 'b:')
    plt.ylim(yy)

    plt.xlabel("Time [s]")

    plt.subplot(6,1,1)
    plt.title(step_params['filename'])

    step_params['refined_OF_amps'] = refined_OF_amps
    step_params['refined_bandpass_amps'] = refined_bandpass_amps
    step_params['prepulse_wfs'] = prepulse_wfs
    step_params['prepulse_psds'] = prepulse_psds
    step_params['prepulse_psd_freq'] = prepulse_psd_freq
    step_params['prepulse_rms'] = prepulse_rms
    step_params['prepulse_pos_rms'] = prepulse_rms
    step_params['random_recon'] = random_recon

    return step_params, fig



def pulse_recon_paper(step_params, template_dict, noise_dict, xrange, charge_times, amp_cal_facs = [1,1], do_bandpass=False, sigma=[1,1,1], ylims=None):
    ## function to take a second pass at reconstructing the pulses with a refined
    ## implemenation of the optimal filter and drive impulse subtraction
    ## this is a tweaked version to make the paper plots. Should be combined with the above function eventually.

    coord_list = ['x', 'y', 'z']

    pulse_wind = 1
    pulse_time = step_params['x_time']

    ## function to take a second pass at reconstructing the steps    
    cdat, attr, _ = get_data(step_params['filename'])
    tvec = np.arange(len(cdat[:,0]))/attr['Fsamp'] - xrange[0]

    nyquist = attr['Fsamp']/2

    fig=plt.figure(figsize=(12,4.5))

    coord_data = np.zeros((3, len(cdat[:,0])))

    coord_labs_in = ['$A_x$ [MeV/c]', '$A_y$ [MeV/c]', '$A_z$ [MeV/c]', 'Charge [$e$]']
    renorm = 2 # rescale after lp filt

    for j, coord in enumerate(coord_list):
        if(coord in ['x', 'y']):
            fc_x = np.array([5, 70])/nyquist
        else:
            fc_x = np.array([100, 500])/nyquist           
        
        b_x, a_x = sp.butter(3, fc_x, btype='bandpass')
        if(do_bandpass):
            if(coord=='x'):
                bandpass = [50]
            elif(coord=='y'):
                bandpass = [200]
            elif(coord=='z'):
                bandpass = [500]
        else:
            bandpass = []

        filt_data = compute_opt_filt(cdat, coord, template_dict, noise_dict, attr['Fsamp'], bandpass=bandpass)
        filt_data_unfilt = compute_opt_filt(cdat, coord, template_dict, noise_dict, attr['Fsamp'], bandpass=[])        

        b_lp, a_lp = sp.butter(3, np.array([1,20])/nyquist, btype='bandpass') ## optional low pass for opt filt
        filt_data_filt = sp.filtfilt(b_lp, a_lp, np.abs(filt_data_unfilt))


        gpts = (tvec > pulse_time-pulse_wind) & (tvec < pulse_time+pulse_wind)
        norm = amp_cal_facs[1][j]

        plt.subplot(4,1,1+j)
        plt.plot(tvec, filt_data_filt*norm*renorm, 'k', lw=0.75, rasterized=True)
        ax1 = plt.gca()

        plt.gca().set_xticks([])
        if(ylims):
            plt.ylim(ylims[coord_list[j]])

        plt.xlim(0, np.diff(xrange))
        plt.ylabel(coord_labs_in[j])

        coord_data[j,:] = filt_data * norm * 1/(sigma[j])

        for c in charge_times:
            wind_size = 0.2
            yy = plt.ylim()
            ax1.fill_between([c-wind_size, c+wind_size], [yy[0], yy[0]], [yy[1], yy[1]], color='blue', alpha=0.1, zorder=0)

    plt.subplot(4,1,4)
    plt.plot(step_params['t_fine']-xrange[0], step_params['charge_fine'], 'gray', lw=0.75, rasterized=True)
    plt.plot(step_params['t_coarse']-xrange[0], step_params['charge_coarse'], 'r')
    plt.ylabel("Charge [$e$]")

    yy = plt.ylim()
    for c in charge_times:
        wind_size = 0.2
        plt.fill_between([c-wind_size, c+wind_size], [yy[0]-5, yy[0]-5], [yy[1]+5, yy[1]+5], color='blue', alpha=0.1) #, zorder=0)

    plt.ylim(yy)

    plt.xlabel("Time [s]")
    plt.xlim(0, np.diff(xrange))

    plt.subplots_adjust(hspace=0)
    fig.align_labels()

    return fig

def sine_subtraction_paper(step_params, res_params, template_dict, noise_dict, xrange, charge_times, amp_cal_facs = [1,1], do_bandpass=False, sigma=[1,1,1], ylims=None, xzoom=[]):

    ## function to plot the effect of the drive force subtraction for the paper

    coord_list = ['x',]

    pulse_wind = 1
    pulse_time = step_params['x_time']

    ## function to take a second pass at reconstructing the steps    
    cdat, attr, _ = get_data(step_params['filename'])
    tvec = np.arange(len(cdat[:,0]))/attr['Fsamp'] - xrange[0]

    nyquist = attr['Fsamp']/2

    fig=plt.figure(figsize=(12,3))

    coord_data = np.zeros((3, len(cdat[:,0])))

    coord_labs_in = ['$A_x$ [MeV/c]', '$A_y$ [MeV/c]', '$A_z$ [MeV/c]', 'Charge [$e$]']
    renorm = 2 # rescale after lp filt

    for j, coord in enumerate(coord_list):
        if(coord in ['x', 'y']):
            fc_x = np.array([5, 70])/nyquist
        else:
            fc_x = np.array([100, 500])/nyquist           
        
        b_x, a_x = sp.butter(3, fc_x, btype='bandpass')
        if(do_bandpass):
            if(coord=='x'):
                bandpass = [50]
            elif(coord=='y'):
                bandpass = [200]
            elif(coord=='z'):
                bandpass = [500]
        else:
            bandpass = []

        filt_data = compute_opt_filt(cdat, coord, template_dict, noise_dict, attr['Fsamp'], bandpass=bandpass)
        filt_data_unfilt = compute_opt_filt(cdat, coord, template_dict, noise_dict, attr['Fsamp'], bandpass=[])        

        b_lp, a_lp = sp.butter(3, np.array([1,20])/nyquist, btype='bandpass') ## optional low pass for opt filt
        filt_data_filt = sp.filtfilt(b_lp, a_lp, np.abs(filt_data_unfilt))


        if(coord in ['x', 'y']):
            fc_x = np.array([5, 120])/nyquist
        else:
            fc_x = np.array([100, 500])/nyquist           
        
        b_x, a_x = sp.butter(3, fc_x, btype='bandpass')
        x_filt = sp.filtfilt(b_x, a_x, cdat[:,x_idx+j])

        plt.subplot(2,1,1)

        ## now do the subtraction for x
        omega0, gamma = res_params['x']['x'][1], res_params['x']['x'][2]
        amp_before, amp_after = step_params['charge_before'], step_params['charge_after']
        svec, omega_vec = get_drive(cdat, attr['Fsamp'], drive_idx=drive_idx)
        Nvec = np.arange(len(cdat[:,drive_idx])) > np.argmax(np.abs(filt_data_filt))
        drive_resp = stepped_sine_response(amp_before, amp_after, svec, omega0, gamma, omega_vec, Nvec)
        edge_buff = 2000
        amp = np.sum(drive_resp[edge_buff:-edge_buff] * cdat[edge_buff:-edge_buff,x_idx])/np.sum(drive_resp[edge_buff:-edge_buff]**2)

        fc_dr = np.array([104, 118])/nyquist
        b_dr, a_dr = sp.butter(3, fc_dr, btype='bandpass')
        drive_data = sp.filtfilt(b_dr, a_dr, cdat[:,x_idx])

        sub_wf = x_filt/amp_cal_facs[0][coord] * 1e9 - amp*drive_resp/amp_cal_facs[0][coord] * 1e9

        plt.plot(tvec, x_filt/amp_cal_facs[0][coord] * 1e9, 'gray', lw=2)
        plt.plot(tvec, sub_wf, 'orange', lw=0.75)
        plt.plot(tvec, drive_data/amp_cal_facs[0][coord] * 1e9, 'tab:red')
        plt.plot(tvec, amp*drive_resp/amp_cal_facs[0][coord] * 1e9, 'tab:blue', lw=0.75)

        if(xzoom):
            plt.xlim(xzoom)

        gpts = (tvec > pulse_time-pulse_wind) & (tvec < pulse_time+pulse_wind)
        norm = amp_cal_facs[1][j]
        plt.gca().set_xticks([])

        plt.ylabel("$x$ position [nm]")

        plt.subplot(2,1,2)
        plt.plot(tvec, filt_data_filt*norm, 'gray')

        cdat[:,x_idx] -= amp*drive_resp
        filt_data = compute_opt_filt(cdat, coord, template_dict, noise_dict, attr['Fsamp'], bandpass=bandpass)
        filt_data_unfilt = compute_opt_filt(cdat, coord, template_dict, noise_dict, attr['Fsamp'], bandpass=[])        

        b_lp, a_lp = sp.butter(3, np.array([1,20])/nyquist, btype='bandpass') ## optional low pass for opt filt
        filt_data_filt_sub = sp.filtfilt(b_lp, a_lp, np.abs(filt_data_unfilt))
        gpts = (tvec > xzoom[0]) & (tvec < xzoom[1])

        plt.plot(tvec, filt_data_filt_sub*norm, 'k')

        plt.xlim(0, np.diff(xrange))
        plt.ylabel(coord_labs_in[j])

        if(xzoom):
            plt.xlim(xzoom)

        plt.ylim([-100,200])
        plt.xlabel("Time [s]")
        plt.gca().set_yticks([-100,0,100, 200])

    plt.subplots_adjust(hspace=0)
    fig.align_labels()

    return fig