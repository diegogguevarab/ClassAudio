import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import minmax_scale, MinMaxScaler, LabelEncoder, StandardScaler
import IPython
import warnings
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import seaborn as sns
import datetime
import joblib
import pprint
sns.set_style('whitegrid')
warnings.filterwarnings("ignore")

seed = 12
np.random.seed(seed)
path_audio_files = "./Data/genres_original/"
hop_length = 512
n_fft = 2048
best_estimator = []
scaler = StandardScaler()
le = LabelEncoder()
thirty_df = pd.read_csv('./features_30_sec.csv')
three_df = pd.read_csv('./features_3_sec.csv')
og_df = pd.concat([thirty_df, three_df])
genres = og_df['label'].unique()
work_df = og_df.drop(columns=['filename', 'length'])
labels = work_df.label
Y = le.fit_transform(labels)
X = scaler.fit_transform(np.array(work_df.iloc[:, :-1], dtype=float))
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)


def descriptionGraph(path):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))
    # Waveshow
    data, sampling_rate = librosa.load(path)
    librosa.display.waveshow(y=data, sr=sampling_rate,
                             color="#A300F9", ax=axes[0][0])
    axes[0][0].set_title('Waveshow')
    #print("Data:", data)

    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=data, hop_length=hop_length)[0]
    axes[0][1].plot(zero_crossing_rate, color="#A300F9")
    axes[0][1].set_title("Zero Crossing Rate")
    #print("Zero Crossing Rate:", zero_crossing_rate)

    # Short Time Fourier Transforms
    stft_data = np.abs(librosa.stft(
        y=data, n_fft=n_fft, hop_length=hop_length))
    axes[0][1].plot(stft_data, color="#A300F9")
    axes[0][1].set_title('STFT')
    #print("STFT:", stft_data)

    # Spectogram
    DB = librosa.amplitude_to_db(stft_data, ref=np.max)
    img = librosa.display.specshow(DB, sr=sampling_rate, hop_length=hop_length,
                                   x_axis='time', y_axis='log', cmap='cool', ax=axes[0][2])
    #print("DB:", DB)

    # Mel Spectogram
    mel_spec = librosa.feature.melspectrogram(
        data, sr=sampling_rate, hop_length=hop_length)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sampling_rate, hop_length=hop_length,
                                   x_axis='time', y_axis='log', cmap='cool', ax=axes[1][0])
    fig.colorbar(img, ax=axes[1][0])
    axes[1][0].set_title("Mel Spectogram")
    #print("Mel Spectogram:", mel_spec_db)

    # Mel Frequency Cepstral Coefficients
    mfcc_data = np.abs(librosa.feature.mfcc(data, sr=sampling_rate))
    img = librosa.display.specshow(
        mfcc_data, sr=sampling_rate, x_axis='time', y_axis='log', cmap='cool', ax=axes[1][1])
    fig.colorbar(img, ax=axes[1][1])
    axes[1][1].set_title("Mel Frequency Cepstral Coefficients")
    #print("Mel Frequency Cepstral Coefficients:", mfcc_data)

    # Chromagram
    chromagram = librosa.feature.chroma_stft(
        data, sr=sampling_rate, hop_length=hop_length)
    img = librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma',
                                   hop_length=hop_length, cmap='coolwarm', ax=axes[1][2])
    fig.colorbar(img, ax=axes[1][2])
    axes[1][2].set_title("Chromagram")
    #print("Chromagram:", chromagram)

    # Spectral Bandwidth
    spectral_centroids = librosa.feature.spectral_centroid(
        y=data, sr=sampling_rate)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=data, sr=sampling_rate)[0]
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    axes[2][0].plot(t, sklearn.preprocessing.minmax_scale(
        spectral_centroids, axis=0), color='b', alpha=0.7)
    axes[2][0].plot(t, sklearn.preprocessing.minmax_scale(
        spectral_centroids + np.sqrt(spec_bw), axis=0), color='r', alpha=0.3)
    axes[2][0].plot(t, sklearn.preprocessing.minmax_scale(
        spectral_centroids - np.sqrt(spec_bw), axis=0), color='r', alpha=0.3)
    axes[2][0].set_title("Spectral Bandwidth")
    #print("Spectral Bandwidth:", spec_bw)

    # Tempo and Beats
    tempo, beat_times = librosa.beat.beat_track(
        y=data, sr=sampling_rate, units='time')
    axes[2][1].vlines(beat_times, np.min(data), np.max(data), color='r')
    axes[2][1].set_title("(Tempo - " + str(round(tempo, 2)) + ")")
    #print("Tempo:", tempo)
    #print("Beats:", beat_times)

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        data, sr=sampling_rate)[0]
    frames = range(len(spectral_rolloff))
    t = librosa.frames_to_time(frames)
    axes[2][2].plot(t, sklearn.preprocessing.minmax_scale(
        spectral_rolloff, axis=0), color='#FFB100')
    axes[2][2].set_title("Spectral Rolloff")
    #print("Spectral Rolloff:", spectral_rolloff)

    # Harmonic and Percussive Components
    y_harm, y_perc = librosa.effects.hpss(data)
    axes[3][0].plot(y_harm, color='#A300F9')
    axes[3][0].plot(y_perc, color='#FFB100')
    axes[3][0].set_title("Harmonic and Percussive Components")
    #print("Harmonic:", y_harm)
    #print("Percussive:", y_perc)

    # Root-Mean-Square
    rms = librosa.feature.rms(y=data)
    axes[3][1].plot(rms, color='#A300F9')
    axes[3][1].set_title("Root-Mean-Square")
    #print("Root-Mean-Square", rms)

    plt.tight_layout(pad=5)
    plt.savefig("Graphs.png")


def training():
    # print(og_df.head(10))
    random_state = 42
    classifier = [
        MLPClassifier(),
        RandomForestClassifier(random_state=random_state),
        SVC(random_state=random_state, probability=True),
        KNeighborsClassifier(),
    ]
    mlp_param_grid = {
        "hidden_layer_sizes": (32, 32),
        "solver": ["adam"],
        "batch_size": [32],
        "early_stopping": [True],
        "validation_fraction": [0.2],
        "n_iter_no_change": [10],
        "learning_rate_init": [0.01],
        "max_iter": [1000]
    }
    rf_param_grid = {
        'max_features': [1, 3, 10],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [False],
        'n_estimators': [100, 300],
        'criterion': ['gini']
    }
    svc_param_grid = {
        'kernel': ['rbf'],
        'gamma': [0.001, 0.01, 0.1, 1],
        'C': [1, 10, 50, 100, 200, 300, 1000]
    }
    knn_param_grid = {
        'n_neighbors': np.linspace(1, 20, 10, dtype=int).tolist(),
        'weights': ['uniform', 'distance'],
        'metric': ['eulidean', 'manhattan']
    }
    classifier_params = [
        mlp_param_grid,
        rf_param_grid,
        svc_param_grid,
        knn_param_grid
    ]
    cv_result = []
    result_name = [
        'MLP Classifier accuracy: ',
        'Random Forest Classifier accuracy: ',
        'Support Vector Machine Classifier accuracy: ',
        'KNN Classifier accuracy: ',
    ]

    for i in range(len(classifier)):
        print(datetime.datetime.now(), f'fitting {result_name[i][:-11]}...')
        clf = GridSearchCV(classifier[i], param_grid=classifier_params[i], cv=StratifiedKFold(
            n_splits=10), scoring='accuracy', n_jobs=-1, verbose=1)
        clf.fit(X_train, Y_train)
        print(datetime.datetime.now(),
              f'Done fitting {result_name[i][:-11]}...')
        y_pred = clf.predict(X_test)
        cv_result.append(clf.best_score_)
        best_estimator.append(clf.best_estimator_)
        print(f'{result_name[i]}{cv_result[i]}\n')
        plot_cm(result_name[i][:-11], Y_test, y_pred)

    voting_c = VotingClassifier(estimators=[('mlp', best_estimator[0]),
                                            ('rfc', best_estimator[1]),
                                            ('svm', best_estimator[2]),
                                            ('knn', best_estimator[3])],
                                voting='soft', n_jobs=-1)
    print(datetime.datetime.now(), f'fitting {result_name[i][:-11]}...')
    voting_c = voting_c.fit(X_train, Y_train)
    print(datetime.datetime.now(), f'Done fitting {result_name[i][:-11]}...')
    y_prediction = voting_c.predict(X_test)
    best_estimator.append(voting_c)
    plot_cm('Ensemble Learning', Y_test, y_prediction)
    plt.figure(figsize=(15, 20))
    cv_results = pd.DataFrame({'Cross Validation Means': cv_result, 'ML Models': [
        'MLP Classifier', 'RandomForestClassifier', 'SVM', 'KNeighborsClassifier']})
    g = sns.barplot(y='Cross Validation Means', x='ML Models', data=cv_results)
    g.set_xlabel('Mean Accuracy')
    g.set_title('Cross Validation Scores')
    plt.savefig("Results.png")


def plot_cm(model, y_true, y_pred):
    conf = confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(
        conf, index=[i for i in genres],
        columns=[i for i in genres]
    )
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(cm, annot=True, fmt='d')
    ax.set_title(f'Confusion Matrix for {model}')
    plt.savefig(f'Confusion Matrix for {model}.png')
    print(classification_report(y_true, y_pred, target_names=genres))


def predictions(path):
    predict_df = pd.DataFrame(data={"chroma_stft_mean": [], "chroma_stft_var": [], "rms_mean": [], "rms_var": [], "spectral_centroid_mean": [], "spectral_centroid_var": [], "spectral_bandwidth_mean": [], "spectral_bandwidth_var": [], "rolloff_mean": [], "rolloff_var": [], "zero_crossing_rate_mean": [], "zero_crossing_rate_var": [], "harmony_mean": [], "harmony_var": [], "perceptr_mean": [], "perceptr_var": [], "tempo": [], "mfcc1_mean": [], "mfcc1_var": [], "mfcc2_mean": [], "mfcc2_var": [], "mfcc3_mean": [], "mfcc3_var": [], "mfcc4_mean": [], "mfcc4_var": [], "mfcc5_mean": [
    ], "mfcc5_var": [], "mfcc6_mean": [], "mfcc6_var": [], "mfcc7_mean": [], "mfcc7_var": [], "mfcc8_mean": [], "mfcc8_var": [], "mfcc9_mean": [], "mfcc9_var": [], "mfcc10_mean": [], "mfcc10_var": [], "mfcc11_mean": [], "mfcc11_var": [], "mfcc12_mean": [], "mfcc12_var": [], "mfcc13_mean": [], "mfcc13_var": [], "mfcc14_mean": [], "mfcc14_var": [], "mfcc15_mean": [], "mfcc15_var": [], "mfcc16_mean": [], "mfcc16_var": [], "mfcc17_mean": [], "mfcc17_var": [], "mfcc18_mean": [], "mfcc18_var": [], "mfcc19_mean": [], "mfcc19_var": [], "mfcc20_mean": [], "mfcc20_var": []})
    data_predict, sampling_rate_predict = librosa.load(path)
    stft_predict = np.abs(librosa.stft(
        y=data_predict, n_fft=n_fft, hop_length=hop_length))
    rms_predict = librosa.feature.rms(y=data_predict)
    spectral_centroids_predict = librosa.feature.spectral_centroid(
        y=data_predict, sr=sampling_rate_predict)[0]
    spec_bw_predict = librosa.feature.spectral_bandwidth(
        y=data_predict, sr=sampling_rate_predict)[0]
    spectral_rolloff_predict = librosa.feature.spectral_rolloff(
        data_predict, sr=sampling_rate_predict)[0]
    zero_crossing_rate_predict = librosa.feature.zero_crossing_rate(
        y=data_predict, hop_length=hop_length)[0]
    y_harm_predict, y_perc_predict = librosa.effects.hpss(data_predict)
    tempo_predict, beat_times_predict = librosa.beat.beat_track(
        y=data_predict, sr=sampling_rate_predict, units='time')
    mfcc_predict = np.abs(librosa.feature.mfcc(
        data_predict, sr=sampling_rate_predict))
    predict_df.loc[0] = [np.mean(stft_predict), np.var(stft_predict), np.mean(rms_predict), np.var(rms_predict), np.mean(spectral_centroids_predict), np.var(spectral_centroids_predict), np.mean(spec_bw_predict), np.var(spec_bw_predict), np.mean(spectral_rolloff_predict), np.var(spectral_rolloff_predict), np.mean(zero_crossing_rate_predict), np.var(zero_crossing_rate_predict), np.mean(y_harm_predict), np.var(y_harm_predict), np.mean(y_perc_predict), np.var(y_perc_predict), tempo_predict, np.mean(mfcc_predict[0]), np.var(mfcc_predict[0]), np.mean(mfcc_predict[1]), np.var(mfcc_predict[1]), np.mean(mfcc_predict[2]), np.var(mfcc_predict[2]), np.mean(mfcc_predict[3]), np.var(mfcc_predict[3]), np.mean(mfcc_predict[4]), np.var(mfcc_predict[4]), np.mean(mfcc_predict[5]), np.var(
        mfcc_predict[5]), np.mean(mfcc_predict[6]), np.var(mfcc_predict[6]), np.mean(mfcc_predict[7]), np.var(mfcc_predict[7]), np.mean(mfcc_predict[8]), np.var(mfcc_predict[8]), np.mean(mfcc_predict[9]), np.var(mfcc_predict[9]), np.mean(mfcc_predict[10]), np.var(mfcc_predict[10]), np.mean(mfcc_predict[11]), np.var(mfcc_predict[11]), np.mean(mfcc_predict[12]), np.var(mfcc_predict[12]), np.mean(mfcc_predict[13]), np.var(mfcc_predict[13]), np.mean(mfcc_predict[14]), np.var(mfcc_predict[14]), np.mean(mfcc_predict[15]), np.var(mfcc_predict[15]), np.mean(mfcc_predict[16]), np.var(mfcc_predict[16]), np.mean(mfcc_predict[17]), np.var(mfcc_predict[17]), np.mean(mfcc_predict[18]), np.var(mfcc_predict[18]), np.mean(mfcc_predict[19]), np.var(mfcc_predict[19])]
    # print(predict_df)
    X_predict = scaler.transform(np.array(predict_df.iloc[:,:], dtype=float))
    # print(X_predict)
    print(datetime.datetime.now(), f'Start predicting {path}...')
    mlp_predict = best_estimator[0].predict(X_predict)
    rfc_predict = best_estimator[1].predict(X_predict)
    svm_predict = best_estimator[2].predict(X_predict)
    knn_predict = best_estimator[3].predict(X_predict)
    voting_predict = best_estimator[4].predict(X_predict)
    print("MLP:", le.inverse_transform(mlp_predict))
    print("Random Forest:", le.inverse_transform(rfc_predict))
    print("SVM:", le.inverse_transform(svm_predict))
    print("KNN:", le.inverse_transform(knn_predict))
    print("Todos:", le.inverse_transform(voting_predict))
    print(datetime.datetime.now(), f'Done predicting {path}...')


def pack_up():
    joblib.dump(best_estimator[0], 'MLP.pkl', compress=1)
    joblib.dump(best_estimator[1], 'Random Forest.pkl', compress=1)
    joblib.dump(best_estimator[2], 'SVM.pkl', compress=1)
    joblib.dump(best_estimator[3], 'KNN.pkl', compress=1)
    joblib.dump(best_estimator[4], 'Todos.pkl', compress=1)


def unpack():
    best_estimator.append(joblib.load('./MLP.pkl'))
    best_estimator.append(joblib.load('./Random Forest.pkl'))
    best_estimator.append(joblib.load('./SVM.pkl'))
    best_estimator.append(joblib.load('./KNN.pkl'))
    best_estimator.append(joblib.load('./Todos.pkl'))

if not os.path.exists('Todos.pkl'):
    print(datetime.datetime.now(), "Start training...")
    training()
    pack_up()
    print(datetime.datetime.now(), "Done training...")
else:
    unpack()

predictions(f'./Data/genres_original/blues/blues.000{np.random.randint(low=0,high=99)}.wav')
predictions(f'./Data/genres_original/classical/classical.000{np.random.randint(low=0,high=99)}.wav')
predictions(f'./Data/genres_original/country/country.000{np.random.randint(low=0,high=99)}.wav')
predictions(f'./Data/genres_original/disco/disco.000{np.random.randint(low=0,high=99)}.wav')
predictions(f'./Data/genres_original/hiphop/hiphop.000{np.random.randint(low=0,high=99)}.wav')
predictions(f'./Data/genres_original/jazz/jazz.000{np.random.randint(low=0,high=99)}.wav')
predictions(f'./Data/genres_original/metal/metal.000{np.random.randint(low=0,high=99)}.wav')
predictions(f'./Data/genres_original/pop/pop.000{np.random.randint(low=0,high=99)}.wav')
predictions(f'./Data/genres_original/reggae/reggae.000{np.random.randint(low=0,high=99)}.wav')
predictions(f'./Data/genres_original/rock/rock.000{np.random.randint(low=0,high=99)}.wav')
