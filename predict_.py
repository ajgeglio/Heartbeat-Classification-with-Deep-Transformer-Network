from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from transformers import AutoModelForAudioClassification, Trainer, TrainingArguments
from transformers import AutoFeatureExtractor
from datasets import Audio, Dataset, load_from_disk
from timeit import default_timer as stopwatch
import time
import evaluate
import matplotlib.pyplot as plt
import os
import argparse
import torch
import numpy as np


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="weighted")

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
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    max_duration = 0.3  # seconds
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True, 
    )
    return inputs


if __name__== '__main__':
    start = stopwatch()
    t = time.localtime()
    current_time = time.strftime("%b-%d-%Y-%H:%M", t)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser(description='Run Evaluation prediction on trained Wav2Vec2 model and output\
                                        a performance report and confusion matrix plot of the results')
    parser.add_argument('--dataset_dir', help="directory of the evaluation dataset", dest="dataset_dir", default='/work/ajgeglio/Tap_Data/Other_data/test_dataset')
    args = parser.parse_args()

     ####### LOAD DATASET ##############
    try:
        tap_dataset_test = load_from_disk(args.dataset_dir)['evaluation']
    except:
        tap_dataset_test = load_from_disk(args.dataset_dir)
    ############# PRINT BASIC DATA PARAMETERS ####################
    print("############# WAVEFORM ###################")    
    fs = tap_dataset_test.features['audio'].sampling_rate
    wavform = tap_dataset_test[0]['audio']['array']
    reshape_wavform = np.reshape(wavform, order='F', newshape=-1)
    sample_time = wavform.shape[0]/fs
    try: np.isclose(sample_time, 0.085375)
    except: 
        print("Data casted incorrectly to not reflect actual sample time window")
        quit()
    reshape_time = len(reshape_wavform)/fs
    print("Sample rate:", fs,
        "\nReal sample time(s):", sample_time, 
        "\nReshaped sample time(s):", reshape_time, 
        "\nwaveform shapes:","original-->", wavform.shape, "reshaped-->", reshape_wavform.shape)
    print("###########################################") 
    label2id, id2label = label_encoder(tap_dataset_test)
    num_labels = len(id2label)
    labels_ = label2id.keys()

    # max_eval_size = 2456
    # prop = num_rows/max_eval_size
    # print(f"{prop*100:0.0f}p of largest eval set")

    ############## Define the path to the saved checkpoint file for the best models ###############################
    ##          Set A
    # model_checkpoint = "/work/ajgeglio/other_models/wav2vec2-base_Mar-27-2023-10:51_heartbeat/checkpoint-30"
    # model_checkpoint = "/work/ajgeglio/other_models/wav2vec2-base_Mar-28-2023-10:41_heartbeat/checkpoint-12"
    # model_checkpoint = "/work/ajgeglio/other_models/wav2vec2-base_Apr-04-2023-19:10_heartbeat/checkpoint-132"
    ##          Set B
    # model_checkpoint = '/work/ajgeglio/other_models/wav2vec2-base_Apr-05-2023-13:43_heartbeat/checkpoint-2640'
    # model_checkpoint = '/work/ajgeglio/other_models/wav2vec2-base_Apr-05-2023-17:29_heartbeat/checkpoint-782'
    # model_checkpoint = '/work/ajgeglio/other_models/wav2vec2-base_Apr-06-2023-10:31_heartbeat/checkpoint-1566'
    model_checkpoint = '/work/ajgeglio/other_models/wav2vec2-base_Apr-06-2023-12:00_heartbeat/checkpoint-1890'
    batch_size = 32

    encoded_test_dataset = tap_dataset_test.map(preprocess_function, remove_columns=["audio"], batched=True)
    # print(encoded_test_dataset)
    model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label)

    f1_metric = evaluate.load("f1")
    # Load the training args used to train the checkpoint
    args = torch.load(model_checkpoint+"/training_args.bin")
    # print(args)
    # Load the trainer used to train the checkpoint
    trainer = Trainer(model=model, args=args)

    logits, y_test , metrics = trainer.predict(encoded_test_dataset)
    y_pred = logits.argmax(-1)
    print(f"TOTAL TIME: {stopwatch() - start:.2f}")
    # predicted_labels = [model.config.id2label[id] for id in y_pred.squeeze().tolist()]
    # print(predicted_labels)
    cm = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred, target_names=labels_))
    print(cm)

    disp = ConfusionMatrixDisplay.from_predictions( y_test, y_pred, display_labels=labels_)
    fig = disp.ax_.get_figure() 
    fig.set_figwidth(14)
    fig.set_figheight(14)  
    plt.savefig(f"confusion_matrix_{model_checkpoint.split('/')[-1]}.png")