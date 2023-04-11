# -*- coding: utf-8 -*-
"""
Preprocess Recorded tap data with tap detection and neonode mapping to key labels

The first arg.INPUT is the directory of the data. To work properly the folder must be:
    01.acoustic --- this is the folder where the original recordings are
    02.neonode --- this is where the raw neonode csv is

The current os.walk function requires the following folder structure:

    01.acoustic --- this is the folder where the original recordings are
    02.neonode --- this is where the raw neonode csv is


TO RUN... 

##########################  EXAMPLE ######################################

WHEN YOU WANT TO JUST PLOT WITHOUT SAMPLING - RECCOMENDED TO DO FIRST
python3 WavPreprocess.py --dir '/work/ajgeglio/Tap_Data/Tony_01_25_23_data' 
--idx 9 --latency -0.14 --peak_height 0.0025 --plot

WHEN YOU WANT TO SAMPLE AND STORE FILES
python3 WavPreprocess.py --dir '/work/ajgeglio/Tap_Data/Tony_01_25_23_data' 
--idx 9 --latency -0.14 --peak_height 0.0025 --plot --selectivity 0.134 --sample

#####################################################################

The OUTPUT for each recording sould be a .wav file for each tap and 1 csv file of label data

Created on Mon Oct 24 12:37:03 2022

@author: Anthony.Geglio
"""

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
matplotlib.style.use('default')
from scipy.io import wavfile
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier
import argparse
import re
from timeit import default_timer as stopwatch

def list_samples_labels(dir_):
    snd_files = []
    labels = []
    for dirs, subdirs, files in os.walk(dir_+'/data/'+set_):
        snd_files.extend(os.path.join(dirs, x) for x in files if x.endswith(".wav"))
    for l in snd_files:
        labels.append(re.findall(r"/([a-z]+)_\w+", l)[-1])
    print(len(labels)," labels counted")
    print(len(snd_files)," audio files counted")
    return snd_files, labels


def dir_to_array2(wav_file_list, idx):
    fs, snd = wavfile.read(wav_file_list[idx])
    time_x = np.linspace(0, len(snd) / fs, num=len(snd))
    init = wav_file_list[idx]
    init = re.findall(r'(\d+)',init)
    if args.set_ == 'set_a':
        init = f"{init[0][:4]}-{init[0][4:6]}-{init[0][6:8]} {init[0][8:10]}:{init[0][10:]}"
        init = datetime.datetime.strptime(init, "%Y-%m-%d %H:%M")
        ext = 0
        print(init)
    else:
        print(init[-1])
        ext = init[0]
        init = int(init[-1])/1000 # convert miliseconds to seconds
        dt_object = datetime.datetime.fromtimestamp(init)
        init = dt_object.strftime("%Y-%m-%d %H:%M")
        init = datetime.datetime.strptime(init, "%Y-%m-%d %H:%M")
        print(init)

    print("sound file start: ", init)
    print("Sample rate", fs)
    print("label", labels[idx])
    snd = (snd-snd.mean())/snd.std()
    return fs, snd, time_x, init, ext

def plot_avr_snd(   avr_signal, # input averaged sound time-domain matrix
                    linewidth):
    color1=tab20c(0)
    color2=tab20c(5)

    fig, ax = plt.subplots(1,1, sharex=True,sharey=True, figsize = (18,3), 
                           tight_layout=True, dpi=400)
    # plt.suptitle(f"File start: {str(name_time)}")
    plt.subplots_adjust(hspace=0.1)
    ax.set_title(f"{set_} start {name_time} label {label}", y=1.0, pad=-14)
    ax.plot(time_x, avr_signal, linewidth=linewidth)
    ax.vlines(peaks_times - behind/fs, -max_/4, max_/4, color=color1, linewidth=1)
    ax.vlines(peaks_times + forward/fs ,-max_/4 ,max_/4, color=color2, linewidth=1)
    ax.hlines(peak_height, start,stop, color='k', linewidth = 0.5)
    # ax.vlines(node_times, -max_/6, max_/6, color='red', linewidth = linewidth)
    ax.vlines(peaks_times, -max_/3, max_/3, color='k', linewidth = linewidth)
    sample_num = []
    for i in range(len(peaks)):
        ax.annotate(i+1, (peaks_times[i], 0),fontsize=8)
        ax.annotate(i+1, (peaks_times[i]- behind/fs, 0),fontsize=4)
        ax.annotate(i+1, (peaks_times[i]+ forward/fs, 0),fontsize=4)
        sample_num.append(i+1)
    # for j, txt in enumerate(node_labels):
    #     ax.annotate((j+1,txt), (node_times[j], np.random.uniform(0.003, max_/2)),fontsize=1.5)
    ax.set_xlim(start,stop)
    plt.savefig(f"./03.plots/{args.set_}/{name_time}_{label}.png")
    plt.close()    
    print(f"Plotted: {name_time} {label}")


def reshape3_(data): #takes input of average signal or 8 channels
    i = 1
    n = 0
    # This is how I control the selectivity (is the sample near a neonode label)
    for p, t in zip(peaks, peaks_times): #peaks is an index, p/fs is the time in decimal seconds
        try:
            b = np.take(data, np.arange(p-behind,p+forward), axis=0)
            wavfile.write(f'./05.wav_samples/{args.set_}/{name_time}_{ext}_{label}_{i:0>3}.wav', fs, b)
            n+=1
        except: 
            print(f'sample {i:0>3} not enough data in window')
            pass
        i+=1
    print(f"Sampled: {name_time} \n {n} files created, {i-1} peaks detected")
    return n

def plot_keys():
    fig, ax = plt.subplots()
    z = key_loc[:,0]
    y = key_loc[:,1]
    n = key_loc[:,2]
    ax.scatter(z, y)
    for i, txt in enumerate(n):
        ax.annotate(txt, (z[i], y[i]))
    plt.savefig(f"key_locs.png")

# Not used anymore
def label_encode(labels):
    encode_dic = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 
        'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 'r':17, 's':18,
        't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z':25, '|':26, '_1':27, 
        '_2':28, '_3':29, '_4':30, '_5':31, '_6':32, 'none':33}
    return [encode_dic[label] for label in labels]

####################### HIGH PASS FILTER ##############################
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a
    
# Currently taking in the average signal
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
######################################################################
if __name__ == "__main__":

    start_time = stopwatch()

    parser = argparse.ArgumentParser(description='Creates tap samples and labels from a wavfile of person recorded typing sentences\
                                                  and associated neonode data')
    parser.add_argument('--dir', help="directory of original wav files", dest="dir_", default =f'/work/ajgeglio/Tap_Data/Other_data/heartbeat_data - hub')
    parser.add_argument('--idx',type=int, help='index of a single recording', dest = "idx_")
    parser.add_argument('--set',type=str, help='set A or set B', dest = "set_", default='set_a')
    parser.add_argument('--plot', required=False, help='save plot to visually check labels', action="store_true")
    parser.add_argument('--latency',type=float, required=False, help='latency of the record start - used to line up with neonode labels', dest = "latency_", default=-0.15)
    parser.add_argument('--selectivity',type=float, required=False, help='only select samples this close to a node label (seconds)', dest = "selectivity", default=0)
    parser.add_argument('--peak_height',type=float, help='the amplitude height threshold for tap detection', dest = "peak_height", default=1)
    parser.add_argument('--sample', help="generate samples and labels", required=False, action="store_true")
    args = parser.parse_args()
    
    # Sort out the Files
    dir_ = args.dir_
    set_ = args.set_
    # Center locations of key taps
    if not os.path.exists(f"./03.plots/{set_}"):
        os.makedirs(f"./03.plots/{set_}")
    if not os.path.exists(f"./04.labels/{set_}"):
        os.makedirs(f"./04.labels/{set_}")
    if not os.path.exists(f"./05.wav_samples/{set_}"):
        os.makedirs(f"./05.wav_samples/{set_}")
    
    snd_files, labels = list_samples_labels(dir_)
    t = 0
    plot_num = 0
    tab20c = cm.get_cmap('tab20c')
    for idx_ in range(len(snd_files)):

        ##################### DEFINE RAW 8-CHANNEL MATRIX ###########################
        fs, snd_matrix, time_x, ini_record, ext = dir_to_array2( wav_file_list = snd_files,
                                                                 idx = idx_) # in the list of 01.acoustic

        name_time = str(ini_record).replace(':','.').replace(' ','.')
        label = labels[idx_]
        # ini_record = ini_record - datetime.timedelta(seconds = args.latency_)
        #################### Sound Attributes ##########################
        record_len = len(time_x)/fs
        end_record = ini_record + datetime.timedelta(seconds = record_len)
        # There is usually noise at the beginning which is why I clip at start
        start, stop = 0.05, record_len
        s = int(start*fs)
        e = int(stop*fs)
        ## How big is the sample window around detected tap?
        if args.set_ == 'set_a':
            behind = 19000
            forward = 5000
        else:
            behind = 1742
            forward = 458

        snd = snd_matrix[s:e]
        time_x = time_x[s:e]
        trace = snd
        # trace = butter_highpass_filter(data=snd, cutoff=1200, fs=fs, order=3)
        max_ = trace.max()
        peak_height =  max_/8

        ############## PEAK DETECTION ##############################
        peaks = signal.find_peaks(trace, 
                                #   threshold = max_/16,
                                distance=1000, # 4 taps per second is 0.25s*96000=24000 
                                height = peak_height,
                                #   prominence =max_/2
                                )[0]
        t += len(peaks)
        print(f'Detected {len(peaks)} peaks')
        peaks_times = time_x[peaks]
        ###################### PLOTTING ##################################
        
        if args.plot:
            if plot_num%10 == 0:
                plot_avr_snd(avr_signal = trace, linewidth = 0.3)
            plot_num += 1
        if args.sample:      
            
            n_samp = reshape3_(snd) # Saves out all of the tap samples
            # labels_ = [labels[idx_]]*n_samp
            # np.savetxt(f'./04.labels/{name_time}_{label}.csv', labels_, fmt='%s', delimiter=',')
            # print(n_samp, "labels created")
    print(f"Time Window being Sampled: {(forward+behind)/fs:0.2f} seconds")
    print(f"TOTAL PEAKS IN {args.set_}: {t}")
    print(f"TOTAL TIME: {stopwatch() - start_time:.2f}")