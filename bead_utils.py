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

## columns in the data files
x_idx, y_idx, z_idx = 0, 1, 2
drive_idx = 8 ## for data starting 9/27/2023


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

def get_noise_template(noise_files, nfft=-1, res_pars=[2*np.pi*30, 2*np.pi*5]):
    ## take noise files and find the average PSD for use in the optimal filter

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
    omega0, gamma = res_pars[0]/np.sqrt(1 - eta**2), 2*res_pars[1] ## factor of two by definition
    omega = 2*np.pi*cf
    sphere_tf = gamma/((omega0**2 - omega**2)**2 + omega**2*gamma**2)
    res_pos = np.argmin( np.abs(omega0-omega) )
    sphere_tf *= J[res_pos]/sphere_tf[res_pos]

    Jout = 1.0*J
    Jout[J/sphere_tf > 5] = 1e20

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

    noise_dict = {"freq": cf, "J": Jout}

    return noise_dict

def plot_charge_steps(charge_vec):
    dt = []
    for j,t in enumerate(charge_vec[:,0]):
        dt.append(labview_time_to_datetime(t-charge_vec[0,0]))

    plt.figure()
    #plt.plot_date(dt, charge_vec[:,1], 'k-')
    plt.plot(charge_vec[:,1], 'k.-')
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.show()

def signed_correlation_with_drive(dat, attr, nperseg=-1):
    xdat = dat[:,x_idx]
    ddat = dat[:,drive_idx]

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
    #print("Drive frequency is: %.2f Hz"%f[didx])

    corr_vec = []
    #print("Number of segments is: ", nwindows)
    for i in range(nwindows):
        st, en = i*nperseg, (i+1)*nperseg
        corr = -np.real(np.fft.rfft(xdat[st:en])/np.fft.rfft(ddat[st:en])) ## negative sign gives the charge with positve as excess protons
        corr_vec.append(corr[didx])

    return np.array(corr_vec)


def simple_correlation_with_drive(dat, attr, drive_freq, bw=1, decstages = -1, cal_fac=5e-6, make_plots=False):
    ### simple bandpass filter
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
        #for n in range(decstages):
        #    corr_vec_rms = sp.decimate(corr_vec_rms, 12)  
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

def plot_correlation_with_drive(dat, template, attr, skip_drive=False, lp_freq=20, make_plots=False):

    corr_vec = sp.correlate(dat[:,x_idx], template, mode='same')

    ## filter for RMS calculation
    fc = lp_freq/(attr['Fsamp']/2) ## lowpass at 20 Hz
    b, a = sp.butter(3, fc, btype='lowpass')


    ## special case for 20230803 when two drives were on
    ## due to the beating want to remove the times when the second drive was there
    ## this is the opposite of what we usually want
    if(skip_drive):
        drive_dat = dat[:,drive_idx]
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

    return np.median(corr_vec_rms), corr_vec_rms


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

def impulse_response(t, A, omega0, gamma, t0):
    outvec = np.zeros_like(t)
    gpts = t>t0
    outvec[gpts] = A*np.exp(-(t[gpts]-t0)*gamma)*np.sin(omega0*(t[gpts]-t0))
    return outvec

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

def parse_impulse_amplitude(input_string, label="MeV"):
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

def get_average_template(calib_dict, make_plots=False, fit_pars=[]):
    
    pulse_dict = {}
    fit_dict = {}
    fit_vals = []

    for impulse_amp in calib_dict.keys():

        curr_files = calib_dict[impulse_amp]

        curr_temp = 0
        for fname in curr_files:
            cdat, attr, _ = get_data(fname)

            ## find the impulse times
            drive_file = cdat[:,drive_idx]
            impulse_times, _ = find_crossing_indices(drive_file/np.max(drive_file), 0.5)
            window_length = 4000

            for time_idx in impulse_times:

                ## make sure the full impulse is contained
                if(time_idx + window_length/2 >= len(cdat[:,x_idx]) or 
                   time_idx < window_length/2):
                    continue
                sidx = int(time_idx-window_length/2)
                eidx = int(time_idx+window_length/2)
                curr_temp += cdat[sidx:eidx,x_idx]

        curr_temp -= np.median(curr_temp[:int(window_length/4)]) #baseline sub
        curr_temp /= np.max(curr_temp) ## normalize to unity amplitude
        pulse_dict[impulse_amp] = curr_temp

        ## also fit to a damped harmonic oscillator
        tvec = np.arange(len(curr_temp))/attr['Fsamp']
        if(len(fit_pars)>0):
            bp, bc = curve_fit(impulse_response, tvec, curr_temp, p0=fit_pars)
            fit_dict[impulse_amp] = impulse_response(tvec, *bp)
        else:
            bp = [0,0,0,0]
            bc = np.zeros((len(bp), len(bp)))

        ## save fit parameters as well
        fit_vals.append( [impulse_amp, bp[1], np.sqrt(bc[1,1]), bp[2], np.sqrt(bc[2,2])] )

        if(make_plots):
            
            plt.figure()
            plt.plot(tvec, curr_temp)
            plt.plot(tvec, impulse_response(tvec, *bp))
            plt.title("Impulse amplitude = %d MeV"%impulse_amp)
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized amplitude [arb units]")
            plt.show()

    return pulse_dict, fit_dict, np.array(fit_vals)

def bandpass_filt(calib_dict, template_dict, time_offset = 0, bandpass=[], notch_list = [], 
                  omega0 = 2*np.pi*40, gamma = 2*np.pi*4, subtract_sine_step=False, pulse_data=False, 
                  make_plots=False):
    ## simple time domain correlation between template and data
    filt_dict = {}

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

            filtconsts = np.array(bandpass)/nyquist # normalized to Nyquist
            b,a = sp.butter(3,filtconsts, btype='bandpass')
            xdata = sp.filtfilt(b,a,xdata)

            impulse_rise, impulse_fall = find_crossing_indices(cdat[:,drive_idx]/np.max(cdat[:,drive_idx]), 0.5)
            
            impulse_cent = get_impulse_cents(cdat, attr['Fsamp'], time_offset=time_offset, pulse_data=pulse_data, drive_freq = 120)

            filt_dict[impulse_amp] = np.hstack((filt_dict[impulse_amp], np.abs(xdata[impulse_cent])))

            if(make_plots):
                sfac =10
                plt.figure(figsize=(15,3))
                plt.plot(cdat[:,drive_idx]/np.max(cdat[:,drive_idx]))
                plt.plot(np.abs(xdata)*sfac)
                plt.plot(impulse_cent, np.abs(xdata[impulse_cent])*sfac, 'ro')
                #plt.xlim(0,2e5)
                plt.xlim(impulse_cent[0]-1000, impulse_cent[0]+1000)
                plt.ylim(0,2)
                plt.title(impulse_amp)

    return filt_dict

def correlation_filt(calib_dict, template_dict, f0=40, time_offset=0, bandpass=[], notch_list = [], 
                     omega0 = 2*np.pi*40, gamma = 2*np.pi*4, subtract_sine_step=False, pulse_data=True, 
                     make_plots=False):
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


            impulse_cent = get_impulse_cents(cdat, attr['Fsamp'], time_offset=time_offset, pulse_data=pulse_data, drive_freq = 120)

            filt_dict[impulse_amp] = np.hstack((filt_dict[impulse_amp], corr_data[impulse_cent]))

            if(make_plots):
                sfac = 1/5
                plt.figure(figsize=(15,3))
                impulse_times, _ = find_crossing_indices(cdat[:,drive_idx]/np.max(cdat[:,drive_idx]), 0.5)
                plt.plot(cdat[:,drive_idx]/np.max(cdat[:,drive_idx]))
                plt.plot(corr_data*sfac)
                plt.plot(impulse_cent, corr_data[impulse_cent]*sfac, 'ro')
                plt.xlim(0,3e5)
                #plt.xlim(0,2e4)
                plt.ylim(0,2)
                plt.title(impulse_amp)

                plt.figure(figsize=(15,3))
                fp, psd = sp.welch(cdat[:,x_idx], nperseg=2**16, fs=attr['Fsamp'])
                fp_filt, psd_filt = sp.welch(xdata, nperseg=2**16, fs=attr['Fsamp'])
                plt.semilogy(fp, psd)
                plt.semilogy(fp_filt, psd_filt)
                plt.xlim(0,100)
                mv = np.max(psd_filt)
                plt.ylim([1e-7*mv, 2*mv])
                plt.show()

    return filt_dict

def get_impulse_cents(cdat, fs, time_offset=0, pulse_data=True, drive_freq = 120):

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

def predict_step_impulse(cdat, fs, omega0, gamma, make_plots=False):
                xdata = cdat[:,x_idx]
                ## get the drive frequency (normalized template)
                ddata = cdat[:,drive_idx]
                ddata -= np.mean(ddata)
                bpf = [0.5/(fs/2), 1000/(fs/2)]
                b, a = sp.butter(1, bpf, btype='bandpass')
                ddata = sp.filtfilt(b, a, ddata)
                ddata = ddata/np.max(ddata)
                
                ddata_tilde = np.fft.rfft(ddata)
                drive_psd = np.abs(ddata_tilde)**2
                drive_freq_vec = np.fft.rfftfreq(len(ddata), d=1/fs)
                drive_freq = drive_freq_vec[np.argmax(drive_psd)]

                drive_wind = 2 ## window around drive frequency to keep
                bpf = [5/(fs/2), 200/(fs/2)]
                b, a = sp.butter(1, bpf, btype='bandpass')
                xdata_drive = sp.filtfilt(b,a,xdata)

                omega_vec = 2*np.pi*drive_freq_vec
                xtilde = ddata_tilde/(omega0**2 - omega_vec**2 + 1j*gamma*omega_vec)
                xdrive_inv = np.fft.irfft(xtilde)
                scale_func = lambda xdata, A: A*xdrive_inv

                best_scale, scale_err = curve_fit(scale_func, 0, xdata-np.median(xdata), p0=[1,])

                if(make_plots):
                    plt.figure(figsize=(12,5))
                    plt.plot(xdata_drive)
                    plt.plot(best_scale*xdrive_inv)
                    #plt.plot(xdata-np.median(xdata))
                    plt.xlim(0,2e5)
                    plt.show()
                
                return best_scale*xdrive_inv

def optimal_filt(calib_dict, template_dict, noise_dict, pulse_data=True, time_offset=0, 
                 omega0 = 2*np.pi*40, gamma = 2*np.pi*4, subtract_sine_step=False, make_plots=False):
    ## optimally filter including noise spectrum
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
        bpts = (sfreq < 5) | (sfreq > 100)
        J[bpts] = 1e20
        phi = stilde/J

        phi_t = np.fft.irfft(phi)
        phi_t = phi_t/np.max(phi_t)

        if(make_plots):
            plt.figure()
            plt.plot(phi_t)
            plt.plot(curr_template)
            plt.show()

        filt_dict[impulse_amp] = []
        for fname in curr_files:
            cdat, attr, _ = get_data(fname)
            fs = attr['Fsamp']
            xdata = cdat[:,x_idx]

            if(subtract_sine_step): ## remove the impulse caused by the sine wave step from the drive
                step_impulse = predict_step_impulse(cdat, fs, omega0, gamma, make_plots=False)     
                xdata -= step_impulse


            corr_data = np.abs(sp.correlate(xdata, phi_t, mode='same'))

            impulse_cent = get_impulse_cents(cdat, fs, time_offset=time_offset, pulse_data=pulse_data, drive_freq = 120)

            corr_vals = []
            corr_idx = []
            wind=30
            for ic in impulse_cent:
                current_search = corr_data[(ic-wind):(ic+wind)]
                corr_vals.append(np.max(current_search))
                corr_idx.append(ic-wind+np.argmax(current_search))
            filt_dict[impulse_amp] = np.hstack((filt_dict[impulse_amp], corr_vals))

            if(make_plots):
                fstr = str.split(fname,'/')[-1]
                sfac = 1/10
                plt.figure(figsize=(15,3))
                plt.plot(cdat[:,drive_idx]/np.max(cdat[:,drive_idx]))
                plt.plot(np.abs(corr_data*sfac))
                plt.plot(corr_idx, np.abs(corr_vals)*sfac, 'ro')
                #plt.xlim(0,3e5)
                #plt.xlim(0,2e4)
                plt.xlim(impulse_cent[0]-1000, impulse_cent[0]+1000)
                plt.ylim(0,2)
                plt.title("opt filt: " + str(impulse_amp) + ", " + fstr)

    return filt_dict