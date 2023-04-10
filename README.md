# The performance of statistical machine learning methods in the classification and diagnosis of time-series heart sounds
### Tony Geglio | M.S. Data Science
### Michigan Technological University
### https://github.com/ajgeglio/Audio-signal-classificat-with-GridsearchCV-and-k-fold-cross-validation-for-multiclass-target

# 1. Introduction to the project
In my research, I am focused on audio processing for classification and localization of sound sources. One way this could be extended to the health community is in the diagnosis of cardiovascular abnormalities through analyzing the heart sounds. 

## Problems to Solve
Currently, patients with cardiovascular disease have tools for monitoring their heart, however early diagnosis of a heart problem is still difficult. I think through very inexpensive sensors connected to a human body and communicating data with a smartphone could provide continuous analysis to people with heart problems, and send alerts, rather than requiring the patient to do the measurement and then react.

## Implementation Plan and Data Description
Previously, I explored statistical machine learning methods to classify heart sounds. Now, I am using transfer learning, using a deep transformer network called Wav2Vec2, originally designed for natural language processing. For data, currently I am looking at a past competition for classifying heart sounds [1]. This data has 2 challenges, the heart sound segmentation and the heart sound classification for 2 datasets with different classes. The challenge was to correctly classify the unlabeled parts of the dataset. I do not have the labels for the test sets, therefore I will use the training sets for validation and testing.

## Data
Data was collected for a 2011 challenge proposed by Bentley et al. The challenge included 2 data sets: data set A)  with heart sounds from the general public via the iStethoscope Pro iPhone app; and, data set B) with heart sounds from a clinic trial in hospitals using the digital stethoscope DigiScope. Combined, there are a total of 585 samples, each being a short clip in .wav format. The class balance is heavily biased to “normal” for both data sets.

4 categories for Dataset A:

* Normal
* Murmur
* Extra Heart Sound
* Artifact

3 classes contained in Dataset B:

* Normal
* Murmur
* Extrasystole

### Experiments and Results
4 Class Classification for Dataset A - No Oversampling
SET A
Loading best model from /work/ajgeglio/other_models/wav2vec2-base_Mar-27-2023-10:51_heartbeat/checkpoint-30 (score: 0.7304625199362041).
{'train_runtime': 109.7388, 'train_samples_per_second': 117.552, 'train_steps_per_second': 1.367, 'train_loss': 0.7688572458598925, 'epoch': 46.0}                       
 31%|████████████████████████████████████████▏                                                                                          | 46/150 [01:49<04:08,  2.39s/it]
***** Running Evaluation *****
  Num examples = 19
  Batch size = 32
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 226.05it/s]
TOTAL TIME: 142.68
              precision    recall  f1-score   support

    artifact       0.71      0.83      0.77         6
    extrahls       0.00      0.00      0.00         3
      murmur       0.80      0.80      0.80         5
      normal       0.57      0.80      0.67         5

    accuracy                           0.68        19
   macro avg       0.52      0.61      0.56        19
weighted avg       0.59      0.68      0.63        19

## This is after using findpeaks to chop up the the dataset by clipping out 0.55 second clips

Loading best model from /work/ajgeglio/other_models/wav2vec2-base_Apr-04-2023-19:10_heartbeat/checkpoint-132 (score: 0.6692898435989222).
{'train_runtime': 299.3673, 'train_samples_per_second': 296.125, 'train_steps_per_second': 2.004, 'train_loss': 1.0096153507427292, 'epoch': 48.84} 
 33%|███████████████████████████████████▌                                                                         | 196/600 [04:59<10:17,  1.53s/it]
***** Running Evaluation *****
  Num examples = 215
  Batch size = 32
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 14.57it/s]
TOTAL TIME: 332.70

############# WAVEFORM ###################
Sample rate: 16000 
Real sample time(s): 0.5226875 
Reshaped sample time(s): 0.5226875 
waveform shapes: original--> (8363,) reshaped--> (8363,)
###########################################
Using amp half precision backend                                                                                                                    
***** Running Prediction *****
  Num examples = 270
  Batch size = 32
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 14.55it/s]TOTAL TIME: 15.30
              precision    recall  f1-score   support

    artifact       0.82      0.95      0.88        88
    extrahls       0.17      0.03      0.05        37
      murmur       0.88      0.87      0.88        79
      normal       0.58      0.73      0.64        66

    accuracy                           0.75       270
   macro avg       0.61      0.65      0.61       270
weighted avg       0.69      0.75      0.71       270

[[84  0  2  2]
 [11  1  0 25]
 [ 2  0 69  8]
 [ 6  5  7 48]]

## I Reduced height of the peak detector to oversample the dataset
 Loading best model from /work/ajgeglio/other_models/wav2vec2-base_Apr-05-2023-13:43_heartbeat/checkpoint-2640 (score: 0.943282005982124).
{'train_runtime': 1973.16, 'train_samples_per_second': 536.018, 'train_steps_per_second': 4.181, 'train_loss': 0.24108446513587709, 'epoch': 64.0} 
 43%|█████████████████████████████████████████████▏                                                            | 3520/8250 [32:53<44:11,  1.78it/s]
***** Running Evaluation *****
  Num examples = 1511
  Batch size = 32
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:03<00:00, 15.56it/s]
TOTAL TIME: 2116.96
***** Running Prediction *****
  Num examples = 1511
  Batch size = 32
 96%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▍    | 46/48 [00:02<00:00, 15.71it/s]TOTAL TIME: 6.99
              precision    recall  f1-score   support

    artifact       0.99      0.99      0.99       852
    extrahls       0.76      0.83      0.80        96
      murmur       0.91      0.92      0.92       337
      normal       0.84      0.79      0.81       226

    accuracy                           0.94      1511
   macro avg       0.88      0.88      0.88      1511
weighted avg       0.94      0.94      0.94      1511

[[847   2   2   1]
 [  2  80   1  13]
 [  0   5 311  21]
 [  4  18  26 178]]

# SET B
## 
Loading best model from /work/ajgeglio/other_models/wav2vec2-base_Mar-28-2023-09:49_heartbeat/checkpoint-10 (score: 0.6046583850931676).
{'train_runtime': 101.0611, 'train_samples_per_second': 477.929, 'train_steps_per_second': 2.969, 'train_loss': 1.243794214157831, 'epoch': 20.73}                       
 14%|██████████████████▎                                                                                                                | 42/300 [01:41<10:20,  2.41s/it]
***** Running Evaluation *****
  Num examples = 69
  Batch size = 32
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 11.81it/s]
TOTAL TIME: 127.83
Using amp half precision backend                                                                                                                                         
***** Running Prediction *****
  Num examples = 70
  Batch size = 32
 67%|█████████████████████████████████████████████████████████████████████████████████████████▎                                            | 2/3 [00:00<00:00, 10.61it/s]TOTAL TIME: 8.25
              precision    recall  f1-score   support

  extrastole       0.00      0.00      0.00         7
      murmur       0.67      0.13      0.22        15
      normal       0.72      0.98      0.83        48

    accuracy                           0.70        70
   macro avg       0.46      0.37      0.35        70
weighted avg       0.64      0.70      0.62        70

## After sampling 0.55 s time windows

## Oversampling to class balance
DATASET CREATION TIME: 2.35
DatasetDict({                                                                                                                                      
    train: Dataset({
        features: ['audio', 'label'],
        num_rows: 6990
    })
    validation: Dataset({
        features: ['audio', 'label'],
        num_rows: 772
    })
    test: Dataset({
        features: ['audio', 'label'],
        num_rows: 773
    })
})
OVERSAMPLING TIME: 76.74
{'0': 'extrastole', '1': 'murmur', '2': 'normal'}
Loading best model from /work/ajgeglio/other_models/wav2vec2-base_Apr-06-2023-10:31_heartbeat/checkpoint-1566 (score: 0.6754090794282015).
{'train_runtime': 1691.601, 'train_samples_per_second': 619.827, 'train_steps_per_second': 4.788, 'train_loss': 0.4135295678888267, 'epoch': 44.99}
 30%|███████████████████████████████▏                                                                        | 2430/8100 [28:11<1:05:47,  1.44it/s]
***** Running Evaluation *****
  Num examples = 772
  Batch size = 32
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:01<00:00, 13.22it/s]
TOTAL TIME: 1835.25
***** Running Prediction *****
  Num examples = 773
  Batch size = 32
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:01<00:00, 14.25it/s]TOTAL TIME: 15.37
              precision    recall  f1-score   support

  extrastole       0.33      0.28      0.30       111
      murmur       0.86      0.47      0.61       162
      normal       0.73      0.86      0.79       500

    accuracy                           0.69       773
   macro avg       0.64      0.54      0.57       773
weighted avg       0.70      0.69      0.68       773

[[ 31   3  77]
 [  2  76  84]
 [ 61   9 430]]

## Increased feature extractor sample time
DATASET CREATION TIME: 0.44
DatasetDict({                                                                                                                                      
    train: Dataset({
        features: ['audio', 'label'],
        num_rows: 6990
    })
    validation: Dataset({
        features: ['audio', 'label'],
        num_rows: 772
    })
    test: Dataset({
        features: ['audio', 'label'],
        num_rows: 773
    })
})
OVERSAMPLING TIME: 74.74
{'0': 'extrastole', '1': 'murmur', '2': 'normal'}
Loading best model from /work/ajgeglio/other_models/wav2vec2-base_Apr-06-2023-12:00_heartbeat/checkpoint-1890 (score: 0.8131090830779825).
{'train_runtime': 2678.6646, 'train_samples_per_second': 391.426, 'train_steps_per_second': 3.024, 'train_loss': 0.3415688495425617, 'epoch': 50.99}
 34%|███████████████████████████████████▎                                                                    | 2754/8100 [44:38<1:26:39,  1.03it/s]
***** Running Evaluation *****
  Num examples = 772
  Batch size = 32
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:04<00:00,  6.04it/s]
TOTAL TIME: 2816.33
***** Running Prediction *****
  Num examples = 773
  Batch size = 32
 92%|█████████████████████████████████████████████████████████████████████████████████████████████████████▏        | 23/25 [00:01<00:00, 15.43it/s]TOTAL TIME: 15.43
              precision    recall  f1-score   support

  extrastole       0.43      0.39      0.41       111
      murmur       0.71      0.69      0.70       162
      normal       0.80      0.82      0.81       500

    accuracy                           0.73       773
   macro avg       0.65      0.63      0.64       773
weighted avg       0.73      0.73      0.73       773

[[ 43   7  61]
 [  6 112  44]
 [ 50  38 412]]


############## Random Forests ####################
SET A
RandomForestClassifier(max_depth=3, max_features='sqrt', n_estimators=10,
                       random_state=42)
{'criterion': 'gini', 'max_depth': 3, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 10}

[x] performance for RF classifier
Test:
accuracy:0.710
recall:0.710
precision:0.801
Classification Report
               precision    recall  f1-score   support

    artifact       0.90      0.90      0.90        10
    extrahls       0.36      0.80      0.50         5
      murmur       1.00      0.75      0.86         8
      normal       0.75      0.38      0.50         8

    accuracy                           0.71        31
   macro avg       0.75      0.71      0.69        31
weighted avg       0.80      0.71      0.72        31

SET B (3-class)
accuracy:0.733
recall:0.733
precision:0.664
Classification Report
               precision    recall  f1-score   support

  extrastole       0.00      0.00      0.00        12
      murmur       0.78      0.29      0.42        24
      normal       0.73      0.97      0.83        80

    accuracy                           0.73       116
   macro avg       0.50      0.42      0.42       116
weighted avg       0.66      0.73      0.66       116

The second modification was to explore a 5-class classification problem after combining the two data sets. After the combination, the data contained 5 unique classes:

* Normal
* Murmur
* Extra Heart Sound
* Artifact
* Extrasystole

# 2. Systematic Review with help from ChatGPT
### Chat GPT:
Cardiac sound signal classification is an important area of research in cardiology that aims to accurately identify abnormal heart sounds, which can be indicative of various cardiac disorders. Accurate classification of cardiac sound signals is essential for making timely and accurate diagnoses, and for developing effective treatment plans. However, current methods for analyzing and classifying cardiac sound signals have several limitations and challenges.

One potential solution to these challenges is the use of artificial intelligence (AI), machine learning (ML), and deep learning (DL) techniques, which have shown promise in improving the accuracy and reliability of sound analysis for cardiac sound signal classification.
The limitations and challenges of current methods for cardiac sound signal classification include the subjectivity of human interpretation, which can lead to variability in diagnoses, and the lack of standardization in terminology and classification schemes. In addition, traditional signal processing techniques may not be effective for capturing subtle differences in sound patterns that are indicative of cardiac disorders.
The purpose of this review is to evaluate the effectiveness of AI/ML/DL methods for cardiac sound signal classification, with a focus on the potential for these techniques to improve the accuracy and reliability of sound analysis. The review will include a systematic search and analysis of relevant studies, including a critical evaluation of the strengths and limitations of AI/ML/DL methods for cardiac sound signal classification. The ultimate goal of the review is to provide insights into the potential of these techniques for improving clinical outcomes and advancing our understanding of cardiac disorders.

Here are five publications on the current state of the art in cardiac sound signal classification published between 2020 and 2023:

1. "Deep learning-based detection of abnormal heart sounds using a time-frequency representation of phonocardiogram" (2020) by Li et al. This paper proposes a deep learning-based approach for detecting abnormal heart sounds using a time-frequency representation of phonocardiogram signals.
2. "Automatic classification of heart sounds using a convolutional neural network" (2020) by Badawy et al. This paper presents a convolutional neural network-based approach for classifying heart sounds, which achieved high accuracy on a publicly available dataset.
3. "A novel approach for automatic heart sound classification using deep convolutional neural network with transfer learning" (2021) by Singh et al. This paper presents a novel approach for heart sound classification using a deep convolutional neural network with transfer learning, which achieved high accuracy on a dataset of over 3,000 heart sound recordings.
4. "A deep learning approach for automated detection of mitral regurgitation from heart sound signals" (2021) by Vasconcelos et al. This paper proposes a deep learning-based approach for automated detection of mitral regurgitation from heart sound signals, which achieved high sensitivity and specificity on a dataset of over 1,000 heart sound recordings.
5. "A systematic review of automatic classification of heart sounds using machine learning techniques" (2022) by Pandit et al. This review article provides a comprehensive overview of the current state of the art in automatic classification of heart sounds using machine learning techniques, including a critical evaluation of the strengths and limitations of existing approaches and a discussion of future research directions.

### Not Chat GPT
The fifth citation generated by Chat GPT looks relevant, however I was unable to access the article. Another relevant systematic review is titled “Heart sound classification using signal processing and machine learning algorithms” (2022) by Zeinali et. al[2]. In this study, they explored extracting signal features such as amplitude, dominant frequencies, and the discrete wavelet transforms, and information theory features. The authors explored using the support vector machines classifier (SVC), gradient boosting classifier (GBC), and random forest classifier (RFC). The accuracy ranged from 75 - 87% with gradient boosting performing the best.

# 3. Method
One common method for statistical machine learning classification of audio data is explained in multiple speaker identification studies[3] [4]. Their method is to extract features, in this case, extracting the Mel Frequency Cepstral Coefficients (MFCC) from audio signals. Statistical classification can be done on the MFCC features by representing the signal features in a tabular format. I will test out bagging and boosting methods, such as Random Forest and Adaboost, as well as a Support Vector Machine and Multi-Layer Perceptron and compare their performance on the MFCC data. Later, I may explore deep learning on the raw signals because 1D-CNN on audio has shown good performance in other studies and requires less processing time.

We use a train-test-split of 75/25, gridsearchCV for parameter tuning, 5-fold cross-validation for resampling, and the weighted average F1-score is used to optimize the models.

# 4. Results

## Random Forests

Best Model:

RandomForestClassifier(max_depth=4, n_estimators=15, random_state=42)
{'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 15}

* accuracy:0.714
* recall:0.714
* precision:0.716

Classification Report

|                 | precision | recall | f1-score | support |
|-----------------|-----------|--------|----------|---------|
| artifact        | 1.00      | 0.90   | 0.95     | 10      |
| extrahls        | 0.44      | 0.80   | 0.57     | 5       |
| extrastole      | 0.00      | 0.00   | 0.00     | 12      |
| murmur          | 1.00      | 0.25   | 0.40     | 32      |
| normal          | 0.69      | 0.95   | 0.80     | 88      |
|-----------------|-----------|--------|----------|---------|
| accuracy        |           |        | 0.71     | 147     |
| macro avg       | 0.63      | 0.58   | 0.54     | 147     |
| weighted avg    | 0.72      | 0.71   | 0.65     | 147     |

## Multi-Layer Perceptron

Best Model:

MLPClassifier(activation='logistic', alpha=0.1, hidden_layer_sizes=(40, 20), max_iter=500)
              
{'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': (40, 20)}

* accuracy:0.728
* recall:0.728
* precision:0.672

Classification Report

|                 | precision | recall | f1-score | support |
|-----------------|-----------|--------|----------|---------|
| artifact        | 1.00      | 0.90   | 0.95     | 10      |
| extrahls        | 0.50      | 1.00   | 0.67     | 5       |
| extrastole      | 0.00      | 0.00   | 0.00     | 12      |
| murmur          | 0.64      | 0.50   | 0.56     | 32      |
| normal          | 0.75      | 0.88   | 0.81     | 88      |
|-----------------|-----------|--------|----------|---------|
| accuracy        |           |        | 0.73     | 147     |
| macro avg       | 0.58      | 0.66   | 0.60     | 147     |
| weighted avg    | 0.67      | 0.73   | 0.69     | 147     |

## Support Vector Machine

SVC(C=0.1, gamma=0.01, kernel='linear', max_iter=10000, probability=True)

{'C': 0.1, 'gamma': 0.01, 'kernel': 'linear'}

* accuracy:0.673
* recall:0.673
* precision:0.625

Classification Report

|                |precision  | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| artifact       | 1.00      | 0.70   | 0.82     | 10      |
| extrahls       | 0.29      | 0.40   | 0.33     | 5       |
| extrastole     | 0.00      | 0.00   | 0.00     | 12      |
| murmur         | 0.64      | 0.28   | 0.39     | 32      |
| normal         | 0.68      | 0.92   | 0.78     | 88      |
|-----------------|-----------|--------|----------|---------|
| accuracy       |           |        | 0.67     | 147     |
| macro avg      | 0.52      | 0.46   | 0.47     | 147     |
| weighted avg   | 0.63      | 0.67   | 0.62     | 147     |

## Gradient Boosting Classifier

GradientBoostingClassifier(max_depth=1, n_estimators=50, random_state=42)

{'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 50}

* accuracy:0.701
* recall:0.701
* precision:0.658

Classification Report

|              	| precision 	| recall 	| f1-score 	| support 	|
|--------------	|-----------	|--------	|----------	|---------	|
| artifact     	| 1.00      	| 0.70   	| 0.82     	| 10      	|
| extrahls     	| 0.57      	| 0.80   	| 0.67     	| 5       	|
| extrastole   	| 0.00      	| 0.00   	| 0.00     	| 12      	|
| murmur       	| 0.73      	| 0.25   	| 0.37     	| 32      	|
| normal       	| 0.69      	| 0.95   	| 0.80     	| 88      	|
|-----------------|-----------|--------|----------|---------|
| accuracy     	|           	|        	| 0.70     	| 147     	|
| macro avg    	| 0.60      	| 0.54   	| 0.53     	| 147     	|
| weighted avg 	| 0.66      	| 0.70   	| 0.64     	| 147     	|



# Works Cited


[1]	“Classifying Heart Sounds Challenge.” http://www.peterjbentley.com/heartchallenge/ (accessed Feb. 12, 2023).

[2]	Y. Zeinali and S. T. A. Niaki, “Heart sound classification using signal processing and machine learning algorithms,” Mach. Learn. Appl., vol. 7, p. 100206, Mar. 2022, doi: 10.1016/j.mlwa.2021.100206.

[3]	S. Nakagawa, L. Wang, and S. Ohtsuka, “Speaker Identification and Verification by Combining MFCC and Phase Information,” IEEE Trans. Audio Speech Lang. Process., vol. 20, no. 4, pp. 1085–1095, May 2012, doi: 10.1109/TASL.2011.2172422.

[4]	M. Hasan, M. Jamil, G. Rabbani, and Md. S. Rahman, “Speaker Identification Using Mel Frequency Cepstral Coefficients,” Proc. 3rd Int. Conf. Electr. Comput. Eng. ICECE 2004, Dec. 2004.






# Heartbeat-Classification-with-Deep-Transformer-Network
