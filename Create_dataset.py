'''
Create a Hugging Face dataset from CSV files of Labels and .wav samples of captured taps already preprocessed
That were created with WavPreprocess.py. This program has more flexibility to allow for experimentation with 
different data shapes, such as reshape-c-style with np.reshape, stack, max-absulute value, average. These are defined in mapping 
functions. Also, you can specify the proportion of total data to select for experimentation.
The reshaping of the 8-channels were found to be the most effective represeentation, so this was incorperated into the preprocess
function in the training program 'Wav2vec2-tap2key'.

EXAMPLE

python create_dataset.py --all_data --sample_rate 24_000 --reshape_c_style --oversample

Note - In order to use a mapping function like reshape_c_style, you must specify a sample_rate, otherwise the dataset
only maps to the file location which is much more efficient, but does not contain the audio feature.

'''

import numpy as np
import random
import os
from datasets import Audio, Dataset, DatasetDict,  interleave_datasets
from transformers import AutoFeatureExtractor
import argparse
from timeit import default_timer as stopwatch
import matplotlib.pyplot as plt

def list_samples_labels(dir_):
    roots_ = []
    files_ = []
    for root, dirs, files in os.walk(dir_):
        roots_.append(root)
        files_.append(files)
    files_ = [item for sublist in files_ for item in sublist]
    label_files = [roots_[0]+'/'+x for x in files_ if 'label.csv' in x]
    label_files.sort()
    tap_files = [roots_[0]+'/'+x for x in files_ if '8chan.wav' in x]
    tap_files.sort()
    return label_files, tap_files

def concat_labels(label_files):
    label = np.empty([0,2], dtype=str)
    for i in range(len(label_files)):
        label1 =  np.loadtxt(label_files[i], delimiter=',', dtype=str)[:,:2]
        label = np.concatenate([label,label1])
    return label

def label_encoder(dataset):
    try:
        labels = dataset.features["label"].names
    except:
        labels = dataset['train'].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    return label2id, id2label

def create_dataset(data_dir, test_ds_dir, fs_, seed_):
    label_files, tap_files = list_samples_labels(data_dir)
    label = concat_labels(label_files)
    ds = Dataset.from_dict(
        {"audio": tap_files, 'label': label[:,1]}
        ).cast_column("audio", Audio(mono=False, sampling_rate = fs_))
    ds = ds.class_encode_column("label")      
    # split twice and combine
    train_set = ds.train_test_split(    shuffle = True, 
                                        seed = seed_, 
                                        stratify_by_column ='label',
                                        test_size=0.3)
    test_set = train_set['test'].train_test_split(  shuffle = True, 
                                                    seed = seed_, 
                                                    stratify_by_column ='label',
                                                    test_size=0.5)
    ds = DatasetDict({
        'train'         : train_set['train'],
        'validation'    : test_set['train'],
        'evaluation'    : test_set['test']})
    print(f"Saving evaluation dataset separately to: {test_ds_dir}")
    ds['evaluation'].save_to_disk(test_ds_dir)
    return ds
    
def oversample_interleave(tap_dataset, seed_):
        ds0 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==0)
        ds1 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==1)
        ds2 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==2)
        ds3 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==3)
        ds4 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==4)
        ds5 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==5)
        ds6 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==6)
        ds7 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==7)
        ds8 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==8)
        ds9 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==9)
        ds10 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==10)
        ds11 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==11)
        ds12 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==12)
        ds13 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==13)
        ds14 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==14)
        ds15 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==15)
        ds16 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==16)
        ds17 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==17)
        ds18 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==18)
        ds19 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==19)
        ds20 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==20)
        ds21 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==21)
        ds22 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==22)
        ds23 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==23)
        ds24 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==24)
        ds25 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==25)
        ds26 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==26)
        ds27 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==27)
        ds28 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==28)
        ds29 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==29)
        ds30 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==30)
        ds31 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==31)
        ds32 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==32)
        ds33 = tap_dataset['train'].filter(lambda tap_dataset: tap_dataset["label"]==33)

        tap_ds_oversample = interleave_datasets([ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9, ds10, ds11, ds12,
                                        ds13, ds14, ds15, ds16, ds17, ds18, ds19, ds20, ds21, ds22, ds23, ds24, ds25, 
                                        ds26, ds27, ds28, ds29, ds30, ds31, ds32, ds33], 
                                        probabilities=None, seed=seed_, stopping_strategy = 'all_exhausted')
        tap_ds_oversample = DatasetDict({   'train'         : tap_ds_oversample,
                                            'validation'    : tap_dataset['validation'],
                                            'evaluation'    : tap_dataset['evaluation']
                                            })
        return tap_ds_oversample


def preprocess_function(examples):
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    max_duration = 5  # seconds
    audio_arrays = [x["array"] for x in examples["audio"]]
    # audio_arrays = [np.reshape(x["array"], order='F', newshape=-1) for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True, 
    )
    return inputs


if __name__ == "__main__":
    start_time = stopwatch()

    parser = argparse.ArgumentParser(description='Creates a dataset of taps from a wavfile of recorded typed sentences and associated neonode file')
    # Logical operators for configuration of data
    parser.add_argument('--presplit', help='Used if samples are stored in separate directories for training and testing', action="store_true")
    parser.add_argument("--oversample", help="oversample class samples until 'all-exausted' to create class balance dataset", action="store_true")
    parser.add_argument('--test_size',type=float, help='validation/evaluation dataset split', dest = "ts_",default=0.5)
    parser.add_argument('--prop',type=float, help='proportion of total data to use', dest = "prop_",default=1.0)
    parser.add_argument('--reshape_c_style', help='reshapes the 8-channel data with numpy reshape, order = C', action="store_true")
    parser.add_argument('--reshape_c_style_top', help='reshapes the 8-channel data with numpy reshape, only top mics', action="store_true")
    parser.add_argument('--reshape_c_style_bottom', help='reshapes the 8-channel data with numpy reshape, only bottom mics', action="store_true")
    parser.add_argument("--reshape_stack", help="flattens 8-channels by stacking to 8x length", action="store_true")
    parser.add_argument('--max_absolute', help='maps 8-channel dataset to a 1 channel ds with the max absolute amplitude value', action="store_true")
    parser.add_argument('--encode', help='maps dataset with autofeature extractor (currently this is done in the training and inference programs)', action="store_true")
    parser.add_argument('--seed',type=int, help='set random seed', dest = "seed_",default=42)
    parser.add_argument('--sample_rate',type=int, help='sample rate to cast audio (khz)', dest = "fs_",default=None)
    parser.add_argument('--plot_average_channel', help='plot signal after average channel', action="store_true")
    # Taps and label files
    parser.add_argument('--data_dir', help="directory of all 96k samples and labels", required=False, dest="data_dir", default='/work/ajgeglio/Tap_Data/00.All_TapsLabels_96k')
    parser.add_argument('--train_dir96k', help="directory of train samples and labels", required=False, dest="train_dir96k", default='/work/ajgeglio/Tap_Data/01.Train_TapsLabels_96k')
    parser.add_argument('--test_dir96k', help="directory of train samples and labels", required=False, dest="test_dir96k", default='/work/ajgeglio/Tap_Data/02.Test_TapsLabels_96k')
    # Datastes Directories
    parser.add_argument('--oversampled_dir', help="directory of the oversampled dataset", dest="interleave_dir", default='/work/ajgeglio/Tap_Data/10.Oversampled_Dataset')
    parser.add_argument('--save_96k_all', help="directory of dictionary dataset with train+validation+evaluation sets", required=False, dest="save_96k_all", default = '/work/ajgeglio/Tap_Data/11.All_Dataset_96k')
    parser.add_argument('--save_tr96k', help="directory of a training dataset only", required=False, dest="save_tr96k", default = '/work/ajgeglio/Tap_Data/03.Train_Dataset_96k')
    parser.add_argument('--save_te96k', help="directory of a dictionary dataset with validation+evaluation sets", required=False, dest="save_te96k", default = '/work/ajgeglio/Tap_Data/04.Test_Dataset_96k')

    args = parser.parse_args()

    if not args.presplit:
        label_files, tap_files = list_samples_labels(args.data_dir)
        label = concat_labels(label_files)
        if args.prop_ < 1.0:
            idx = list(range(len(label)))
            random.seed(args.seed_)
            randIdx = random.sample(idx,int(args.prop_*len(idx)))
            tap_files = np.array(tap_files)[randIdx]
            label = np.array(label)[randIdx]
            print("new sample size: ", len(tap_files))
            print(len(label))
            
        tap_dataset = Dataset.from_dict({"audio": tap_files, 'label': label[:,1]})
        if args.fs_ != None:
            tap_dataset = tap_dataset.cast_column("audio", Audio(mono=False, sampling_rate = args.fs_))
        # To use the mapping functions
        try:
            if args.reshape_c_style:
                example = tap_dataset[64]
                tmp = reshape_c_style(example)
                plot_sample(tmp, 'sample_reshape-interleave')
                # quit()
                tap_dataset = tap_dataset.map(reshape_c_style)
                sample = tap_dataset[0]['audio']['array']
                print("New Shape: ", sample.shape)
            if args.reshape_c_style_top:
                example = tap_dataset[64]
                tmp = reshape_c_style_top(example)
                plot_sample(tmp, 'sample_reshape-interleave-top')
                tap_dataset = tap_dataset.map(reshape_c_style_top)
                sample = tap_dataset[0]['audio']['array']
                print("New Shape: ", sample.shape)
            if args.reshape_c_style_bottom:
                example = tap_dataset[64]
                tmp = reshape_c_style_bottom(example)
                plot_sample(tmp, 'sample_reshape-interleave-bottom')
                tap_dataset = tap_dataset.map(reshape_c_style_top)
                sample = tap_dataset[0]['audio']['array']
                print("New Shape: ", sample.shape)
            if args.reshape_stack:
                example = tap_dataset[64]
                tmp = reshape_stack(example)
                plot_sample(tmp, 'sample_reshape-stack')
                tap_dataset = tap_dataset.map(reshape_stack)
                sample = tap_dataset[0]['audio']['array']
                print("New Shape: ", sample.shape)
            if args.max_absolute:
                example = tap_dataset[64]
                tmp = channel_maxabs(example)
                plot_sample(tmp, 'sample_max_absolute')
                tap_dataset = tap_dataset.map(channel_maxabs)
                new_sample = tap_dataset[0]['audio']['array']
                print("New Shape: ", new_sample.shape)
            if args.plot_average_channel:
                example = tap_dataset[64]["audio"]['array']
                new_x = np.average(example, axis=0)
                print(new_x.shape)
                plt.plot(new_x)
                plt.savefig(f"/home/ajgeglio/FutureGroup/sample_avr_channel.png")
                quit()
        except: 
            print('Dataset not created, you probably need to specify a sampling rate, ie, 16000 to use the mapping function')
            quit()

        tap_dataset = tap_dataset.class_encode_column("label")      
        # split twice and combine
        train_set = tap_dataset.train_test_split(   shuffle = True, 
                                                    seed = args.seed_, 
                                                    stratify_by_column ='label',
                                                    test_size=0.3)
        test_set = train_set['test'].train_test_split(  shuffle = True, 
                                                        seed = args.seed_, 
                                                        stratify_by_column ='label',
                                                        test_size=args.ts_)
        tap_dataset = DatasetDict({
            'train'         : train_set['train'],
            'validation'    : test_set['train'],
            'evaluation'    : test_set['test']})
        
        print(f"Saving 3-split 96k dataset to disk, directory: {args.save_96k_all}")
        tap_dataset.save_to_disk(args.save_96k_all)
        print(f"Saving evaluation dataset separately to: {args.save_te96k}")
        tap_dataset['evaluation'].save_to_disk(args.save_te96k)
        
        
    if args.presplit:
        label_files, tap_files = list_samples_labels(args.train_dir96k)
        label = concat_labels(label_files)
        label_files_test, tap_files_test = list_samples_labels(args.test_dir96k)
        label_test = concat_labels(label_files_test)
        tap_dataset = Dataset.from_dict(   {"audio": tap_files, 
                                            'label': label[:,1]})
        tap_dataset_test = Dataset.from_dict(   {"audio": tap_files_test, 
                                                'label': label_test[:,1]})
        tap_dataset = tap_dataset.class_encode_column("label").shuffle(seed=args.seed_)
        tap_dataset_test = tap_dataset_test.class_encode_column("label")
        tap_dataset_test = tap_dataset_test.train_test_split(   test_size = args.ts_, 
                                                                stratify_by_column ='label', 
                                                                shuffle = True, 
                                                                seed = args.seed_)
        print(f"Saving training dataset to disk, directory: {args.save_tr96k}")
        tap_dataset.save_to_disk(args.save_tr96k)
        print(f"Saving test dataset to disk, directory: {args.save_te96k}")
        tap_dataset_test.save_to_disk(args.save_te96k)

    if args.oversample:
        tap_dataset = oversample_interleave()
        print(f'Saving oversampled dataset to disk, {args.interleave_dir}')
        tap_dataset.save_to_disk(args.interleave_dir)

    if args.encode:
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        tap_dataset = tap_dataset.cast_column("audio", Audio(mono=True))
        encoded_dataset = tap_dataset.map(preprocess_function, remove_columns=["audio"], batched=True)
        print(f"Saving encoded dataset to disk, /work/ajgeglio/Tap_Data/09.Encoded_Dataset")
        encoded_dataset.save_to_disk('/work/ajgeglio/Tap_Data/09.Encoded_Dataset')
   
    print('DATASET DESCRIPTION #################################')
    print(tap_dataset)
    print('Sound Features #######################################')
    try:
        fs = tap_dataset['train'].features['audio'].sampling_rate
        print("Sampling Rate: ", fs)
        wavform = tap_dataset['train'][0]['audio']
        print("Waveform", wavform)
        print("Waveform shape", wavform['array'].shape)
    except:
        try:
            fs = args.fs_
            print("Sampling Rate: ", fs)
            wavform = tap_dataset['train'][0]['audio']
            print("Waveform", wavform)
        except:
            fs = args.fs_
            print("Sampling Rate: ", fs)
            wavform = tap_dataset[0]['audio']
            print("Waveform", wavform)

    print('\n')

    print(f"TOTAL TIME: {stopwatch() - start_time:.2f}")