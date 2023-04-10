'''
Wav2vec2_tap2key is a transformer based on pretrained facebook Wav2vec2 model for audio processing and audio sample classification

'''
from huggingface_hub import notebook_login
import evaluate
from datasets import Audio, Dataset, load_from_disk, DatasetDict,  interleave_datasets
# from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import Wav2Vec2Model, Wav2Vec2Config
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy
from timeit import default_timer as stopwatch
import time
import argparse
import os
import random
import re
import matplotlib.pyplot as plt
import numpy as np
# from huggingface_hub import HfApi
# from Create_dataset import label_encoder, concat_labels, oversample_interleave, create_dataset

'''
Wav2vec preprocess step to create features from audio samples...

Examples are fed to the feature_extractor with the argument truncation=True, 
as well as the maximum sample length. This will ensure that very long inputs 
like the ones in the _silence_ class can be safely batched.

https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/audio_classification.ipynb#scrollTo=qUtxmoMvqml1

The feature extractor will return a list of numpy arays for each example:

'''
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
    
def preprocess_function(examples):
    max_duration = 0.55  # seconds
    # audio_arrays = [x["array"] for x in examples["audio"]]
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True, 
    )
    return inputs

def list_samples_labels(dir_):
    snd_files = []
    labels = []
    for dirs, subdirs, files in os.walk(dir_):
        snd_files.extend(os.path.join(dirs, x) for x in files if x.endswith(".wav"))
    try:
        for l in snd_files:
            labels.append(re.findall(r"_([a-z]+)_\w+", l)[-1])
    except:
        for l in snd_files:
            labels.append(re.findall(r"([a-z]+)__\w+", l)[0])
    print(len(labels)," labels counted")
    print(len(snd_files)," audio files counted")
    # print(labels[:5])
    return snd_files, labels

def create_dataset(data_dir, test_ds_dir, fs_, seed_):
    snd_files, labels  = list_samples_labels(data_dir)
    # label = concat_labels(label_files)
    ds = Dataset.from_dict(
        {"audio": snd_files, 'label': labels}
        ).cast_column("audio", Audio(mono=True, sampling_rate = fs_))
    ds = ds.class_encode_column("label")      
    # split twice and combine
    train_set = ds.train_test_split(    shuffle = True, 
                                        seed = seed_, 
                                        stratify_by_column ='label',
                                        test_size=(0.3))
    test_set = train_set['test'].train_test_split(  shuffle = True, 
                                                    seed = seed_, 
                                                    stratify_by_column ='label',
                                                    test_size=0.5)
    ds = DatasetDict({
        'train'         : train_set['train'],
        'validation'    : test_set['train'],
        'test'    : test_set['test']})
    print(f"Saving test dataset separately to: {test_ds_dir}")
    ds['test'].save_to_disk(test_ds_dir)
    return ds

def oversample_interleave(dataset, seed_):
    if args.set_ == 'set_a':
        ds0 = dataset['train'].filter(lambda dataset: dataset["label"]==0)
        ds1 = dataset['train'].filter(lambda dataset: dataset["label"]==1)
        ds2 = dataset['train'].filter(lambda dataset: dataset["label"]==2)
        ds3 = dataset['train'].filter(lambda dataset: dataset["label"]==3)
        ds_oversample = interleave_datasets([ds0, ds1, ds2, ds3], 
                                        probabilities=None, seed=seed_, stopping_strategy = 'all_exhausted')
        ds_oversample = DatasetDict({   'train'         : ds_oversample,
                                        'validation'    : dataset['validation'],
                                        'test'    : dataset['test']
                                            })
    else:
        ds0 = dataset['train'].filter(lambda dataset: dataset["label"]==0)
        ds1 = dataset['train'].filter(lambda dataset: dataset["label"]==1)
        ds2 = dataset['train'].filter(lambda dataset: dataset["label"]==2)
        ds_oversample = interleave_datasets([ds0, ds1, ds2], 
                                        probabilities=None, seed=seed_, stopping_strategy = 'all_exhausted')
        ds_oversample = DatasetDict({   'train'         : ds_oversample,
                                            'validation'    : dataset['validation'],
                                            'test'    : dataset['test']
                                            })
    return ds_oversample

def chunk_examples(examples):
    chunks = []
    for sound in examples["audio"]:
        chunks += [sound[i:i + 50] for i in range(0, len(sound), 50)]
    return {"chunks": chunks}



def plot_data_grid():
    fig, ax = plt.subplots(3,3, figsize=(10,10), tight_layout=True)
    for r in range(ax.shape[0]):
        for c in range(ax.shape[1]):
            n = random.randint(0,85)
            wavform = beat_dataset_tmp[n]['audio']['array']
            time_x = np.linspace(0, len(wavform) / fs, num=len(wavform))
            label_ = id2label[str(beat_dataset_tmp[n]['label'])]
            ax[r][c].plot(time_x,wavform)
            ax[r][c].set_title(f"{label_}")
            ax[r][c].set_xlabel('time (s)')
            ax[r][c].set_ylabel('amplitude')
    plt.savefig('Heartbeat.png')

def concat_labels(label_files):
    label = np.empty([0,2], dtype=str)
    for i in range(len(label_files)):
        label1 =  np.loadtxt(label_files[i], delimiter=',', dtype=str)[:]
        label = np.concatenate([label,label1])
    return label


'''
Here, we need to define a function for how to compute the metrics from the predictions, which will just use 
the metric we loaded earlier. The only preprocessing we have to do is to take the argmax of our predicted logits:
'''
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="weighted")

if __name__== '__main__':
    
    start = stopwatch()
    t = time.localtime()
    current_time = time.strftime("%b-%d-%Y-%H:%M", t)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser(description='Wav2vec2_tap2key does training and evaluaiton of the tap sample\
            dataset that has already been created or can create dataset. The model is a transformer based on Wav2vec2 architecture with\
            Wav2Vec2-base weight initialialization. The fine tuning is done on 34 classes based on the virtual\
            table top keyboard and audio sample classification is supervised based on the labels created with\
            WavePreprocess.py')
    parser.add_argument('--dataset_dir', help="directory of the hugging face dataset", dest="dataset_dir", default='/work/ajgeglio/Tap_Data/Other_data/dataset')
    parser.add_argument('--data_dir', help="directory of the sampled beats", dest="data_dir", default='/home/ajgeglio/OtherProjects/05.wav_samples')
    parser.add_argument('--seed',type=int, help='set random seed', dest = "seed_",default=42)
    parser.add_argument('--set',type=str, help='set A or set B', dest = "set_", default='set_a')
    parser.add_argument("--load_dataset", help="load hugging face dataset saved to disk", action="store_true")
    parser.add_argument('--save_te96k', help="directory to save evaluation dataset", dest="save_te96k", default = '/work/ajgeglio/Tap_Data/Other_data/test_dataset')
    parser.add_argument("--oversample", help="oversample hugging face dataset", action="store_true")
    parser.add_argument('--sample_rate',type=int, help='sample rate to cast audio (khz)', dest = "fs_",default=16_000)
    parser.add_argument("--early_stop", help="early stopping using evaluation loss", action="store_true")
    parser.add_argument("--plot", help="plot some data", action="store_true")
    parser.add_argument("--train_model", help="plot some data", action="store_true")
    parser.add_argument("--early_patience", type=int, help="number of worse evals before early stopping", default=16)
    parser.add_argument("--epochs", type=int, help="number of epochs to run", default=150)
    args = parser.parse_args()

    ### For set a original: --data_dir '/work/ajgeglio/Tap_Data/Other_data/heartbeat_data - hub/data/set_a'


    print(f"Creating Dataset {args.set_}")
    s2 = stopwatch()
    beat_dataset = create_dataset(args.data_dir+'/'+args.set_,args.save_te96k, args.fs_, args.seed_)
    print(beat_dataset)
    print(f"DATASET CREATION TIME: {stopwatch() - s2:.2f}")

    if args.oversample:
        s4 = stopwatch()
        os_ds = oversample_interleave(beat_dataset, args.seed_)
        beat_dataset = os_ds
        print(beat_dataset)
        print(f"OVERSAMPLING TIME: {stopwatch() - s4:.2f}")
    

    # Label Dictionary
    label2id, id2label = label_encoder(beat_dataset)
    num_labels = len(id2label)
    print(id2label)
    ############# PRINT BASIC DATA PARAMETERS ####################
    print("############# WAVEFORM ###################")
    # reshape_wavform = np.reshape(wavform, order='F', newshape=-1)
    beat_dataset_tmp = beat_dataset['train']
    fs = beat_dataset_tmp.features['audio'].sampling_rate
    n = random.randint(0,85)
    wavform = beat_dataset_tmp[n]['audio']['array']
    time_x = np.linspace(0, len(wavform) / fs, num=len(wavform))
    sample_time = wavform.shape[0]/fs

    print("Sample rate:", fs,
        "\nReal sample time(s):", sample_time, 
        # "\nReshaped sample time(s):", reshape_time, 
        "\nwaveform shapes:","original-->", wavform.shape)
    print("###########################################")   

    if args.plot:
        plot_data_grid()

    if args.train_model:
    ############################################# MODEL ############################################################
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        '''
        To apply the preprocess function on all samples in our dataset, we just use the map method of our dataset object we created earlier. 
        * This will apply the function on all the elements of all the splits in dataset, so our training, validation and 
        * testing data will be preprocessed in one single command.
        '''
        s3 = stopwatch()
        encoded_dataset = beat_dataset.map(preprocess_function, remove_columns=["audio"], batched=True)
        print(f"Feature Creation Time: {stopwatch() - s3:.2f}")
        f1_metric = evaluate.load("f1")

        '''
        ######################## Training the model ###################################

        Now that our data is ready, we can download the pretrained model and fine-tune it. 
        For classification we use the AutoModelForAudioClassification class. 
        Like with the feature extractor, the from_pretrained method will download and cache the model for us. 
        As the label ids and the number of labels are dataset dependent, we pass num_labels, label2id, 
        and id2label alongside the model_checkpoint here:
        '''
        model_checkpoint = "facebook/wav2vec2-base"
        batch_size = 32
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            )

        '''
        In some cases, you might be interested in keeping the weights of the pre-trained encoder frozen 
        and optimizing only the weights of the head layers. To do so, simply set the requires_grad attribute 
        to False on the encoder parameters, which can be accessed with the base_model submodule on any 
        task-specific model in the library:
        '''
        # for param in model.base_model.parameters():
        #     param.requires_grad = False

        '''
        To instantiate a Trainer, we will need to define the training configuration and the evaluation metric. 
        The most important is the TrainingArguments, which is a class that contains all the attributes to customize the training. 
        It requires one folder name, which will be used to save the checkpoints of the model, and all other arguments are optional:
        '''
        model_name = f"{model_checkpoint.split('/')[-1]}"

        ######## Training Strategy #############3
        # Set up arguments for early stopping
        callbacks = None
        if args.early_stop:
            callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_patience)]

        args = TrainingArguments(
            f"/work/ajgeglio/other_models/{model_name}_{current_time}_heartbeat",
            logging_steps=10,
            evaluation_strategy = IntervalStrategy.EPOCH,
            save_strategy = IntervalStrategy.EPOCH,
            learning_rate=3e-5,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=args.epochs,
            warmup_ratio=0.1,
            save_total_limit = 3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            gradient_checkpointing=True,
            fp16_full_eval=True,
            fp16=True
            # push_to_hub=True,
        )

        '''
        Here we set the evaluation to be done at the end of each epoch, tweak the learning rate, use the batch_size defined at the top 
        of the notebook and customize the number of epochs for training, as well as the weight decay. Since the best model might not 
        be the one at the end of training, we ask the Trainer to load the best model it saved (according to metric_name) at the end of training.

        The last argument push_to_hub allows the Trainer to push the model to the Hub regularly during training. 
        Remove it if you didn't follow the installation steps at the top of the notebook. If you want to save your model 
        locally with a name that is different from the name of the repository, or if you want to push your model under an organization 
        and not your name space, use the hub_model_id argument to set the repo name (it needs to be the full name, including your namespace: 
        for instance "anton-l/wav2vec2-finetuned-ks" or "huggingface/anton-l/wav2vec2-finetuned-ks").
    
        Then we just need to pass all of this along with our datasets to the Trainer:
        '''

        trainer = Trainer(
            model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            tokenizer=feature_extractor,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            )

        trainer.train()
        # We can check with the evaluate method that our Trainer did reload the best model properly (if it was not the last one):
        trainer.evaluate()
        # You can now upload the result of the training to the Hub, just execute this instruction:
        
    # trainer.push_to_hub()
    print(f"TOTAL TIME: {stopwatch() - start:.2f}")

