# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:39:39 2021

@author: Administrator
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
import glob
import pandas as pd
import datetime as dt

from skimage.feature import match_template
from scipy.signal import find_peaks
from matplotlib.path import Path
import pickle
#%% load kernel

kernel_csv=r"C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\specgram_corr\kernel_zcall_2.csv"


df=pd.read_csv(kernel_csv,index_col=0)
shape_t=df['Timestamp'].values - df['Timestamp'].min()
shape_f=df['Frequency'].values

#%% detection function


def spectrogram_correlation(f, t, Sxx,shape_t,shape_f,offset_t,offset_f):
       
    # kernel_csv=r"C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\specgram_corr\kernel_srw_1.csv"
    # df=pd.read_csv(kernel_csv,index_col=0)
    # shape_t=df['Timestamp'].values - df['Timestamp'].min()
    # shape_f=df['Frequency'].values
    f_lim=[ shape_f.min() - offset_f ,  shape_f.max() + offset_f ]

    # offset_f=50
    # offset_t=0.2
    k_length_seconds=shape_t.max()+offset_t*2
    shape_t=shape_t+offset_t
    
    # generate kernel  
    # f_step=np.diff(f)[0]
    time_step=np.diff(t)[0]
    
    k_t=np.linspace(0,k_length_seconds,int(k_length_seconds/time_step) )
    ix_f=np.where((f>=f_lim[0]) & (f<=f_lim[1]))[0]
    k_f=f[ix_f[0]:ix_f[-1]]
    # k_f=np.linspace(f_lim[0],f_lim[1], int( (f_lim[1]-f_lim[0]) /f_step)  )
    
    kk_t,kk_f=np.meshgrid(k_t,k_f)   
    kernel_background_db=0
    kernel_signal_db=1
    kernel=np.ones( [ k_f.shape[0] ,k_t.shape[0] ] ) * kernel_background_db
    
    x, y = kk_t.flatten(), kk_f.flatten()
    points = np.vstack((x,y)).T 
    p = Path(list(zip(shape_t, shape_f))) # make a polygon
    grid = p.contains_points(points)
    mask = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon  
    kernel[mask]=kernel_signal_db
    
    # fig=plt.figure(num=2)      
    # plt.clf()
    # plt.imshow(kernel,origin = 'lower',aspect='auto',extent=[k_t[0],k_t[-1],k_f[0],k_f[-1]])
    # breakpoint()
    ix_f=np.where((f>=f_lim[0]) & (f<=f_lim[1]))[0]
    spectrog =10*np.log10( Sxx[ ix_f[0]:ix_f[-1],: ] )

    result = match_template(spectrog, kernel)
    corr_score=result[0,:]
    t_score=np.linspace( t[int(kernel.shape[1]/2)] , t[-int(kernel.shape[1]/2)], corr_score.shape[0] )

    # fig=plt.figure(num=3)      
    # plt.clf()
    # plt.subplot(211)
    # plt.imshow(spectrog,aspect='auto',origin = 'lower')
    # plt.colorbar() 
    # plt.subplot(212)
    # plt.plot(t_score,corr_score)
    # plt.grid()
    # plt.colorbar()
    # plt.xlim( [t_score[0],t_score[-1]] )
    # plt.savefig(audiopath[:-4]+'_speccorr_zcall.jpg')
    
    return t_score,corr_score

# def detection_function(audiopath,fft_size):
#         # print(audiopath)    
#     time= dt.datetime.strptime( audiopath.split('\\')[-1], 'aural_%Y_%m_%d_%H_%M_%S.wav' )
     
#     fs, x = wav.read(audiopath)
#     x=x/32767    
#     dBFS=155
#     p =np.power(10,(dBFS/20))*x #convert data.signal to uPa    
#     # fft_size=8192
#     f, t, Sxx = signal.spectrogram(p, fs, window='hamming',nperseg=fft_size,noverlap=0.9*fft_size)

#     t_score,corr_score=spectrogram_correlation(f, t, Sxx,shape_t,shape_f,1,50)

#     peaks_indices = find_peaks(corr_score, height=threshold)[0]
#     # peaks_values = corr_score[peaks_indices]

#     # outputs
#     if peaks_indices.shape[0]>0:
#         detection_times=time +  pd.to_timedelta( t_score[peaks_indices]  , unit='s')
#     else:
#         detection_times=np.array([],dtype= 'datetime64')
#     recording_starttime=time
#     calls_per_min=detection_times.shape[0] / (t[-1]/60)

#     return detection_times,recording_starttime,calls_per_min




#%% loop over files


# audio_folder=r'C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\example_wav'

audio_folder=r'C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\detector_validation_subset_2016'
threshold_vec=np.arange(0.1,.8,.02)

d_dict={}
for threshold in  threshold_vec:     
    d_dict[threshold]=np.array([],dtype='datetime64')



for audiopath in glob.glob(audio_folder+'\*.wav'):         
    print(audiopath)    

    time= dt.datetime.strptime( audiopath.split('\\')[-1], 'aural_%Y_%m_%d_%H_%M_%S.wav' )
     
    fs, x = wav.read(audiopath)
    x=x/32767    
    dBFS=155
    p =np.power(10,(dBFS/20))*x #convert data.signal to uPa    
    fft_size=2**17
    f, t, Sxx = signal.spectrogram(p, fs, window='hamming',nperseg=fft_size,noverlap=0.9*fft_size)

    t_score,corr_score=spectrogram_correlation(f, t, Sxx,shape_t,shape_f,1,5)
    
    for threshold in  threshold_vec:     
        peaks_indices = find_peaks(corr_score, height=threshold)[0]
        if peaks_indices.shape[0]>0:
            detection_times=time +  pd.to_timedelta( t_score[peaks_indices]  , unit='s')
        else:
            detection_times=np.array([],dtype= 'datetime64')
        d_dict[threshold]=np.append(d_dict[threshold],detection_times)
pickle.dump( d_dict, open( "d_dict_bluewhale_zcall_2016.pkl", "wb" ) )

audio_folder=r'C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\detector_validation_subset_2017'
threshold_vec=np.arange(0.1,.8,.02)

d_dict={}
for threshold in  threshold_vec:     
    d_dict[threshold]=np.array([],dtype='datetime64')



for audiopath in glob.glob(audio_folder+'\*.wav'):         
    print(audiopath)    

    time= dt.datetime.strptime( audiopath.split('\\')[-1], 'aural_%Y_%m_%d_%H_%M_%S.wav' )
     
    fs, x = wav.read(audiopath)
    x=x/32767    
    dBFS=155
    p =np.power(10,(dBFS/20))*x #convert data.signal to uPa    
    fft_size=2**17
    f, t, Sxx = signal.spectrogram(p, fs, window='hamming',nperseg=fft_size,noverlap=0.9*fft_size)

    t_score,corr_score=spectrogram_correlation(f, t, Sxx,shape_t,shape_f,1,5)
    
    for threshold in  threshold_vec:     
        peaks_indices = find_peaks(corr_score, height=threshold)[0]
        if peaks_indices.shape[0]>0:
            detection_times=time +  pd.to_timedelta( t_score[peaks_indices]  , unit='s')
        else:
            detection_times=np.array([],dtype= 'datetime64')
        d_dict[threshold]=np.append(d_dict[threshold],detection_times)
pickle.dump( d_dict, open( "d_dict_bluewhale_zcall_2017.pkl", "wb" ) )


#%%
    
# import pickle
# pickle.dump( d_dict, open( "d_dict_bluewhale_zcall_2016.pkl", "wb" ) )




#%% get true clssifications

# df=pd.read_csv(r"C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\detector_validation_subset_2016_bwzcall_seb.csv")    
# detections_seb=pd.to_datetime(df['Timestamp'])


folder=r'C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\Linn_annotations\BW Z-call2016\*_log.csv'
detections=[]
csv_names=glob.glob(folder)
for path in csv_names:
    df=pd.read_csv(path)
    detections.append(df)
detections = pd.concat(detections,ignore_index=True)
detections=pd.to_datetime(detections['Timestamp'])


audio_folder=r'C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\detector_validation_subset_2016'
timevec=[]
for audiopath in glob.glob(audio_folder+'\*.wav',recursive=True): 
    starttime=pd.Timestamp( dt.datetime.strptime( audiopath.split('\\')[-1], 'aural_%Y_%m_%d_%H_%M_%S.wav' ) )
    endtime=starttime + pd.Timedelta('720s')
    dti = pd.Series( pd.date_range(start=starttime,end=endtime,freq='5S') )
    timevec.append(dti)
timevec=pd.concat(timevec,ignore_index=True)    

timediff_max=2.5
difmat= np.squeeze( timevec.values - detections.values[:, None] )
score_2016=np.sum( np.abs( difmat.astype(float)/1e9 ) < timediff_max , axis=0 )>0
timevec_2016=timevec.copy()


folder=r'C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\Linn_annotations\BW Z-call2017\*_log.csv'
detections=[]
csv_names=glob.glob(folder)
for path in csv_names:
    df=pd.read_csv(path)
    detections.append(df)
detections = pd.concat(detections,ignore_index=True)
detections=pd.to_datetime(detections['Timestamp'])


audio_folder=r'C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\detector_validation_subset_2017'
timevec=[]
for audiopath in glob.glob(audio_folder+'\*.wav',recursive=True): 
    starttime=pd.Timestamp( dt.datetime.strptime( audiopath.split('\\')[-1], 'aural_%Y_%m_%d_%H_%M_%S.wav' ) )
    endtime=starttime + pd.Timedelta('720s')
    dti = pd.Series( pd.date_range(start=starttime,end=endtime,freq='5S') )
    timevec.append(dti)
timevec=pd.concat(timevec,ignore_index=True)    

timediff_max=2.5
difmat= np.squeeze( timevec.values - detections.values[:, None] )
score_2017=np.sum( np.abs( difmat.astype(float)/1e9 ) < timediff_max , axis=0 )>0
timevec_2017=timevec.copy()


#%% compare

d_dict = pickle.load( open( "d_dict_bluewhale_zcall_2016.pkl", "rb" ) )

tpr_mat=np.ones(shape=[ threshold_vec.shape[0]  ]) * np.nan
fpr_mat=np.ones(shape=[ threshold_vec.shape[0]  ]) * np.nan

i1=0
for threshold in  threshold_vec:

        automaticdetections= pd.Series( d_dict[threshold] )

        if automaticdetections.shape[0]>1:
            # evaluate performance
            timediff_max=5
            difmat= np.squeeze( timevec_2016.values - automaticdetections.values[:, None] )
            score_auto=np.sum( np.abs( difmat.astype(float)/1e9 ) < timediff_max , axis=0 )>0
            
            tpr= np.sum( score_2016 & score_auto ) / np.sum(score_2016)
            fpr= np.sum( ~score_2016 & score_auto ) / np.sum(~score_2016)
            
            tpr_mat[i1]=tpr
            fpr_mat[i1]=fpr

            
        i1=i1+1  
        
fpr_mat_2016=fpr_mat.copy()
tpr_mat_2016=tpr_mat.copy()

d_dict = pickle.load( open( "d_dict_bluewhale_zcall_2017.pkl", "rb" ) )

tpr_mat=np.ones(shape=[ threshold_vec.shape[0]  ]) * np.nan
fpr_mat=np.ones(shape=[ threshold_vec.shape[0]  ]) * np.nan

i1=0
for threshold in  threshold_vec:

        automaticdetections= pd.Series( d_dict[threshold] )

        if automaticdetections.shape[0]>1:
            # evaluate performance
            timediff_max=5
            difmat= np.squeeze( timevec_2017.values - automaticdetections.values[:, None] )
            score_auto=np.sum( np.abs( difmat.astype(float)/1e9 ) < timediff_max , axis=0 )>0
            
            tpr= np.sum( score_2017 & score_auto ) / np.sum(score_2017)
            fpr= np.sum( ~score_2017 & score_auto ) / np.sum(~score_2017)
            
            tpr_mat[i1]=tpr
            fpr_mat[i1]=fpr

            
        i1=i1+1  
        
fpr_mat_2017=fpr_mat.copy()
tpr_mat_2017=tpr_mat.copy()

#%%


plt.figure(4)
plt.clf()

plt.plot(fpr_mat_2016,tpr_mat_2016,'.-r')
for i, txt in enumerate(threshold_vec):
    plt.annotate("{:.2f}".format(txt), (fpr_mat_2016[i], tpr_mat_2016[i]))

# plt.plot(fpr_mat_2017,tpr_mat_2017,'.-b')
# for i, txt in enumerate(threshold_vec):
#     plt.annotate("{:.2f}".format(txt), (fpr_mat_2017[i], tpr_mat_2017[i]))
    
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.grid()

# plt.savefig('ROC_curve_2016_BW_zcall')
# use 0.3 as threshold    
    
    
    
    
#%%

# automaticdetections=pd.DataFrame(dtct_detections)
# # automaticdetections.to_csv( 'automaticdetections_BW_zcall_full_2017.csv'  )

# df=pd.concat([ pd.DataFrame( dtct_time ), pd.DataFrame( dtct_n_calls_per_min )] ,axis=1 )
# # df.to_csv('detection_timeseries_experimental_srw.csv')

# plt.figure(5)
# plt.clf()
# plt.plot(df.iloc[:,0],df.iloc[:,1],'-k')



