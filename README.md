# The performance of machine learning methods in the classification and diagnosis of time-series heart sounds
### Tony Geglio | M.S. Data Science
### Michigan Technological University


# 1. Introduction to the project
Cardiac sound signal classification is an important area of research in cardiology that aims to accurately identify abnormal heart sounds, which can be indicative of various cardiac disorders. Accurate classification of cardiac sound signals is essential for making timely and accurate diagnoses, and for developing effective treatment plans. However, current methods for analyzing and classifying cardiac sound signals have several limitations and challenges.

One potential solution to these challenges is the use of artificial intelligence (AI), machine learning (ML), and deep learning (DL) techniques, which have shown promise in improving the accuracy and reliability of sound analysis for cardiac sound signal classification. The limitations and challenges of current methods for cardiac sound signal classification include the subjectivity of human interpretation, which can lead to variability in diagnoses, and the lack of standardization in terminology and classification schemes. In addition, traditional signal processing techniques may not be effective for capturing subtle differences in sound patterns that are indicative of cardiac disorders.


## Problems to Solve
Currently, patients with cardiovascular disease have tools for monitoring their heart, however early diagnosis of heart problems is still difficult. The proposal here is: through very inexpensive microphone recorders connected to a human body and communicating data with a smartphone could provide continuous analysis to people with heart problems, and send alerts, rather than requiring the patient to perform a measurement and then react. This project goal is to test machine learning approaches that could be implemented in heart sound classification tools.

## Implementation Plan and Data Description
Previously, I explored statistical machine learning methods to classify heart sounds. Now, I am proposing a transfer learning approach, using Wav2Vec 2.0[1]. Wav2vec is a deep transformer network originally designed for natural language processing. This classification challenge uses data from a past competition for classifying heart sounds[2]. There are two heart sound data sets: Set A with 4-classes, and Set B with 3-classes. The challenge was to submit a model to correctly classify an unlabeled set. I do not have the labels for the unlabeled sets, therefore I will use the training sets for validation and testing of the Wav2Vec 2.0 transformer.

## Data
Data was collected for a 2011 challenge proposed by Bentley et al. The challenge included 2 data sets: data set A)  with heart sounds from the general public via the iStethoscope Pro iPhone app; and, data set B) with heart sounds from a clinic trial in hospitals using the digital stethoscope DigiScope. Combined, there are a total of 585 samples, each being a short clip in .wav format ranging anywhere from 3 to 30 seconds. The class balance is relatively balanced in Set A while Set B is unbalanced with many more “normal” samples. Set B are generally around 3 seconds, and Set A contains longer recordings.

Set A has a total of 124 recordings

4 categories for Set A:
* Normal (31)
* Murmur (34)
* Extra Heart Sound (19)
* Artifact (40)

Set B has a total of 464 recordings

3 classes contained in Set B:
* Normal (320)
* Murmur (95)
* Extrasystole (46)

## Data Augmentation and Sub-sampling

Data Augmentation is performed in the python file called “WavPreprocess-Hearbeat.py”. The augmentation method generates 0.55 second samples from the recordings. Each sample’s “origin” was located using the numpy “find peaks” function. Note there is sample overlap which creates one form of oversampling. I also oversampled using built-in functions for hugging face datasets called “interleave” which generates a class balance. See the figure visualizing the samples 5 and 6 taken from a set B recording. A total of ten beats were detected in this file, so in this case, a 3.5 second recording generates 10 samples totaling 5.5 seconds of data. The augmentation outputs an audio (.wav) file for each sample.




# 4. Results

## Set A with Transfer Learning

|                 |precision  | recall | f1-score | support |
|-----------------|-----------|--------|----------|---------|
|    artifact     |  0.99     | 0.99   |   0.99   |    852  |
|    extrahls     |  0.76     | 0.83   |   0.80   |     96  |
|      murmur     |  0.91     | 0.92   |   0.92   |    337  |
|      normal     |  0.84     | 0.79   |   0.81   |    226  |
|-----------------|-----------|--------|----------|---------|
|    accuracy     |           |        |   0.94   |   1511  |
|   macro avg     |  0.88     | 0.88   |   0.88   |   1511  |
| weighted avg    |  0.94     | 0.94   |   0.94   |   1511  |

|          | artifact | extrahls | murmur | normal |     
|----------|----------|----------|--------|--------|
|artifact  |	847  | 2 |  2 |  1|
|extrahls	 |    2  |80 |  1 | 13|
|murmur	   |    0  | 5 |311 | 21|
|normal	   |    4  |18 | 26 |178|

## Set B with Transfer Learning

|                 |precision  |  recall|  f1-score|  support|
|-----------------|-----------|--------|----------|---------|
|  extrastole     |  0.43     | 0.39   |   0.41   |    111  |
|      murmur     |  0.71     | 0.69   |   0.70   |    162  |
|      normal     |  0.80     | 0.82   |   0.81   |    500  |
|-----------------|-----------|--------|----------|---------|
|    accuracy     |           |        |   0.73   |    773  |
|   macro avg     |  0.65     | 0.63   |   0.64   |    773  |
| weighted avg    |  0.73     | 0.73   |   0.73   |    773  |

|          | extrastole | murmur | normal |   
|----------|------------|----------|--------|
|extrastole| 43 |  7 | 61 |
|murmur 	 |  6 | 112| 44 |
|normal	   | 50 | 38 |412 |

## Set A with MLP
MLPClassifier(activation='logistic', alpha=0.01, hidden_layer_sizes=(40, 20),
              max_iter=500)

[x] performance for MLP classifier
|                 |precision  |  recall|  f1-score| support |
|-----------------|-----------|--------|----------|---------|
|    artifact     |  1.00     | 1.00   |   1.00   |    767  |
|    extrahls     |  0.91     | 0.99   |   0.95   |     86  |
|      murmur     |  0.99     | 0.96   |   0.97   |    303  |
|      normal     |  0.96     | 0.96   |   0.96   |    204  |
|-----------------|-----------|--------|----------|---------|
|    accuracy     |           |        |   0.98   |   1360  |
|   macro avg     |  0.96     | 0.98   |   0.97   |   1360  |
| weighted avg    |  0.98     | 0.98   |   0.98   |   1360  |

|           |artifact   |extrahls|  murmur | normal  | 
|-----------|-----------|--------|---------|---------|  
|artifact   | 767       | 0      | 0       |0        |
| extrahls  | 0         | 85     | 0       |1        |
|murmur     | 0         | 3      | 292     |8        |
|normal     | 0         | 5      | 4       |195      |

TOTAL TIME: 1155.32

## Set B with MLP
MLPClassifier(activation='logistic', alpha=0.01, hidden_layer_sizes=(40, 20),
              max_iter=500)

[x] performance for MLP classifier
|                 |precision  |  recall|  f1-score| support |
|-----------------|-----------|--------|----------|---------|
|  extrastole     |  0.51     | 0.65   |   0.57   |    100  |
|      murmur     |  0.61     | 0.75   |   0.67   |    146  |
|      normal     |  0.87     | 0.76   |   0.81   |    450  |
|-----------------|-----------|--------|----------|---------|
|    accuracy     |           |        |   0.74   |    696  |
|   macro avg     |  0.66     | 0.72   |   0.68   |    696  |
| weighted avg    |  0.77     | 0.74   |   0.75   |    696  |

|                 |extrastole | murmur |  normal  | 
|-----------------|-----------|--------|----------|
|extrastole       |65         |  13    | 22       |
|murmur           |10         | 109    | 27       | 
|normal           |53         | 57     | 340      |

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


[1]	A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.” 2020.

[2]	“Classifying Heart Sounds Challenge.” http://www.peterjbentley.com/heartchallenge/ (accessed Feb. 21, 2023).

[3]	Y. Zeinali and S. T. A. Niaki, “Heart sound classification using signal processing and machine learning algorithms,” Mach. Learn. Appl., vol. 7, p. 100206, Mar. 2022, doi: 10.1016/j.mlwa.2021.100206.

[4]	S. Nakagawa, L. Wang, and S. Ohtsuka, “Speaker Identification and Verification by Combining MFCC and Phase Information,” IEEE Trans. Audio Speech Lang. Process., vol. 20, no. 4, pp. 1085–1095, May 2012, doi: 10.1109/TASL.2011.2172422.

[5]	M. Hasan, M. Jamil, G. Rabbani, and Md. S. Rahman, “Speaker Identification Using Mel Frequency Cepstral Coefficients,” Proc. 3rd Int. Conf. Electr. Comput. Eng. ICECE 2004, Dec. 2004.
