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

def correlation_template(dat, attr, make_plots=False, lp_freq=20):
    ## first make the drive template (cut out the first pulse)
    #drive_dat = dat[:,10]
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
        #plt.plot(tvec[drive_start:drive_end], drive_dat[drive_start:drive_end])
        #plt.plot(tvec,thresh_cross)
        #plt.plot(tvec[:-1],thresh_cross)
        #plt.xlim(0.9*tvec[drive_start], 1.1*tvec[drive_end])
        #plt.plot(template)

    ## now correlate against original time stream
    #template = drive_dat[drive_start:drive_end]
    template = drive_dat[drive_start:(drive_start + 40*115)]
    #template = np.sin(2*np.pi*87*tvec[1:(10*115)])
    window = sp.windows.hamming(len(template))


    return template*window


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
        tvec = np.arange(len(dat[:,0]))/attr['Fsamp']
        plt.figure(figsize = (14, 3))
        plt.plot(tvec,corr_vec)
        plt.plot(tvec,corr_vec_rms)
        plt.plot(tvec,np.ones_like(corr_vec)*np.median(corr_vec_rms),'r:')
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
    
def chi(omega, omega0, gamma, A):
    return A / (omega0**2 - omega**2 - 1j*omega*gamma)

def lorentz(omega, omega0, gamma, A):
    return np.abs(chi(omega, omega0, gamma, A))**2

def fit_tf(dat, attr, nfft=2**19, frange=[20,80]):

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
    o1 = A1*np.sin(omega0*t + delta)
    o1[t > t0] = A2*np.sin(omega0*t[t>t0] + delta)
    return o1

def plot_impulse(dat, template, attr, res_pars, trange=[0, np.inf], skip_drive=False, lp_freq=20, make_plots=False, two_panel=True, fname=""):

    corr_vec = sp.correlate(dat[:,0], template, mode='same')

    ## filter for RMS calculation
    fc = lp_freq/(attr['Fsamp']/2) ## lowpass at 20 Hz
    b, a = sp.butter(3, fc, btype='lowpass')
    corr_vec_rms = sp.filtfilt(b,a,corr_vec**2)

    tvec = np.arange(len(dat[:,0]))/attr['Fsamp']

    ## deconvolve the data by the transfer function
    #fpts = (tvec > trange[0]) & (tvec < trange[1])
    fpts = (tvec > trange[0]) & (tvec < trange[1])
    sfunc_fixed = lambda t, A, d: sfunc(t, A, d, 2*np.pi*87)
    #sfunc_fixed = lambda t, A1, A2, d: sfunc_step(t, A1, A2, d, 2*np.pi*87, 37)
    sfit, scov = curve_fit(sfunc_fixed, tvec[fpts], dat[fpts,0], p0=[1, 0])
    fc = np.array([20,50])/(attr['Fsamp']/2)
    b, a = sp.butter(3, fc, btype='bandpass')
    xfilt_orig = dat[:,0] - sfunc_fixed(tvec, *sfit)
    xfilt2 = sp.filtfilt(b,a,xfilt_orig)
    
    tt = tvec[:4000]
    ttemp = np.exp(-tt*(5.7))*np.sin(2*np.pi*38.1*tt)
    ttemp = np.hstack((np.zeros_like(ttemp), ttemp))

    fc2 = 20/(attr['Fsamp']/2)
    b2, a2 = sp.butter(3, fc2, btype='lowpass')
    #xfilt3 = sp.filtfilt(b2,a2,xfilt_orig)
    xfilt3 = sp.correlate(xfilt_orig, ttemp, mode='same')
    xfilt3 = sp.filtfilt(b2, a2, xfilt3**2)
    xfilt3 *= np.max(xfilt2)/np.max(xfilt3) * 1/2.5

    xtilde = np.fft.rfft(xfilt_orig)
    f = np.fft.rfftfreq(len(dat[:,0]), 1/attr['Fsamp'])
    Ftilde = xtilde/chi(2*np.pi*f, res_pars[0], res_pars[1], 1e4)
    Fdecon = np.fft.irfft(Ftilde)
    fc = np.array([1,200])/(attr['Fsamp']/2)
    b, a = sp.butter(3, fc, btype='bandpass')
    Fdecon = sp.filtfilt(b,a,Fdecon)
    Fdecon *= np.max(dat[:,0])/np.max(Fdecon[50000:-50000])

    if(skip_drive):
        fc = 100/(attr['Fsamp']/2)
        b, a = sp.butter(3, fc, btype='lowpass')
        drive_dat = dat[:,10]
        drive_dat_filt = drive_dat**2
        drive_dat_filt = sp.filtfilt(b,a,drive_dat_filt)
        thresh = 0.1*np.max(drive_dat_filt) ## threshold at 50%
        bad_pts = drive_dat_filt > thresh
        ## pad around each crossing
        buff=10000
        upward_pts = np.where((bad_pts) & ~(np.roll(bad_pts,1)))[0]
        for up in upward_pts:
            bad_pts[(up-buff):up] = True
        upward_pts = np.where(~(bad_pts) & (np.roll(bad_pts,1)))[0]
        for up in upward_pts:
            bad_pts[up:(up+buff)] = True

        ## feedback issue
        bpts2 = (tvec > 31.5) & (tvec < 33.5)
        bad_pts = bad_pts | bpts2
        bpts2 = (tvec < 5) 
        bad_pts = bad_pts | bpts2

    else:
        bad_pts = np.zeros_like(dat[:,10]) > 1

    if(make_plots):

        plt.figure()
        plt.plot(tvec,dat[:,0])
        plt.plot(tvec[fpts], sfunc_fixed(tvec[fpts], *sfit), 'r')

        xx = trange
        plt.figure(figsize = (6,4.5))
        #corr_vec_z = sp.correlate(dat[:,2], template, mode='same')
        #corr_vec_z_rms = sp.filtfilt(b,a,corr_vec_z**2)
        impulse_time = 63.6
        wind_size = 0.75

        if(two_panel):
            plt.subplot(2,1,1)
            #plt.plot(tvec, dat[:,0]-sfunc_fixed(tvec, *sfit))
            #plt.plot(tvec, xfilt)
            pulse_wind_size = 0.010

            if(skip_drive):
                tvec = np.arange(0, len(tvec[~bad_pts]))/attr['Fsamp']

            bpts = (tvec < impulse_time-wind_size) | (tvec > impulse_time+wind_size)
            max_vec = xfilt3[~bad_pts]*1.0
            max_vec[bpts]=0

            #plt.ylim(-0.75,0.75)
            plt.ylim(-1,1)
            yy=plt.ylim()
            pulse_time = tvec[np.argmax(max_vec)]
            plt.plot([pulse_time, pulse_time], yy, 'k:')
            plt.plot(tvec,xfilt2[~bad_pts], label="Data")
            plt.plot(tvec,xfilt3[~bad_pts]*2.2, lw=2, label="Matched filt.")
            #plt.plot(tt+impulse_time, ttemp/2, 'r')
            plt.xlim(xx)

            #plt.plot(impulse_time*np.ones(2), yy, 'k:')      
            plt.ylim(yy)
            plt.xticks([])
            plt.ylabel('Position [arb units]')

            #plt.fill_between([pulse_time-pulse_wind_size, pulse_time+pulse_wind_size], [yy[0], yy[0]], [yy[1], yy[1]], color='red', alpha=0.3)
            plt.legend(loc='lower left')

            #plt.subplot(4,1,2)
            #plt.plot(tvec,dat[:,1])
            #plt.xlim(xx)
            #yy=plt.ylim()
            #plt.plot(impulse_time*np.ones(2), yy, 'k:')      
            #plt.ylim(yy)

            #plt.subplot(4,1,3)
            #plt.plot(tvec,dat[:,2])
            #plt.xlim(xx)
            #yy=plt.ylim()
            #plt.plot(impulse_time*np.ones(2), yy, 'k:')      
            #plt.ylim(yy)

            plt.subplot(2,1,2)
            cal_fac = 1.55e1*4.5
            plt.plot(tvec,corr_vec_rms[~bad_pts]/cal_fac, 'gray')
            b,a = sp.butter(3,0.00005)
            corr_filt = sp.filtfilt(b,a,corr_vec_rms[~bad_pts]/cal_fac)
            plt.plot(tvec,corr_filt, 'r')
            #plt.plot(tvec,drive_dat_filt)    
            #plt.plot(tvec[~bad_pts],drive_dat_filt[~bad_pts])    
            #plt.plot(tvec,corr_vec_z_rms)
            plt.xlim(xx)        
            plt.grid(True)
            #plt.ylim(74,80)
            yy=plt.ylim()

            plt.fill_between([impulse_time-wind_size, impulse_time+wind_size], [yy[0], yy[0]], [yy[1], yy[1]], color='orange', alpha=0.3)

            #window = [80, 88]
            #gpts = (tvec > window[0]) & (tvec < window[1])
            #plt.plot(window, np.median(corr_vec_rms[gpts])*np.ones(2), 'r')

            #plt.plot(impulse_time*np.ones(2), yy, 'k:')  
            plt.ylim(yy)
            #plt.ylim(77,85)
            plt.xlabel("Time [s]")
            plt.ylabel("Charge [$e$]")

            plt.subplots_adjust(hspace=0)

        else:
            if(skip_drive):
                tvec = np.arange(0, len(tvec[~bad_pts]))/attr['Fsamp']

            bpts = (tvec < impulse_time-wind_size) | (tvec > impulse_time+wind_size)
            max_vec = xfilt3[~bad_pts]*1.0
            max_vec[bpts]=0

            plt.ylim(-0.6,0.6)
            yy=plt.ylim()
            pulse_time = tvec[np.argmax(max_vec)]
            plt.plot([pulse_time, pulse_time], yy, 'k:')
            plt.plot(tvec,xfilt2[~bad_pts], label="Data")
            plt.plot(tvec,xfilt3[~bad_pts]*2.2, lw=2, label="Matched filt.")
            #plt.plot(tt+impulse_time, ttemp/2, 'r')
            plt.xlim(xx)

            #plt.plot(impulse_time*np.ones(2), yy, 'k:')      
            plt.ylim(yy)
            #plt.xticks([])
            plt.ylabel('Position [arb units]')
            plt.xlabel("Time [s]")
            #plt.fill_between([pulse_time-pulse_wind_size, pulse_time+pulse_wind_size], [yy[0], yy[0]], [yy[1], yy[1]], color='red', alpha=0.3)
            plt.legend(loc='lower left')

            #plt.figure(figsize=(6,4.5))
            #stp = len(ttemp)
            #h, b = np.histogram(2.2*xfilt3[~bad_pts][::int(stp/2)], bins=np.linspace(0,0.1,50))
            #bc = b[:-1] + np.diff(b)/2
            #plt.errorbar(bc, h, yerr=np.sqrt(h), fmt='ko')

        if(len(fname)> 0):
            plt.savefig(fname)
        plt.show()

    return np.median(corr_vec_rms)
