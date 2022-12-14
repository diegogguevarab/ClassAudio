# Librerías utilizadas en el modelo
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import minmax_scale, MinMaxScaler, LabelEncoder, StandardScaler
import sklearn
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import seaborn as sns
import datetime
import joblib

# Estilo de Seaborn
sns.set_style('whitegrid')
warnings.filterwarnings("ignore")

# Variables globales utilizadas
seed = 12
np.random.seed(seed)
path_audio_files = "./Data/genres_original/"
hop_length = 512
n_fft = 2048
best_estimator = []
# Accuracies de training (24-11)
all_accuracies = ['91.6%', '90.2%', '80.1%', '94.3%', '92.3%'] 
scaler = StandardScaler()
le = LabelEncoder()

# Archivos para entrenar los modelos
thirty_df = pd.read_csv('./features_30_sec.csv')
three_df = pd.read_csv('./features_3_sec.csv')
og_df = pd.concat([thirty_df, three_df])
genres = og_df['label'].unique()
work_df = og_df.drop(columns=['filename', 'length'])
labels = work_df.label

# Repartición del set de entrenamiento y pruebas con Scikit-learn
Y = le.fit_transform(labels)
X = scaler.fit_transform(np.array(work_df.iloc[:, :-1], dtype=float))
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)


# Función donde se calculan las propiedades del audio para generar un gráfico
def descriptionGraph(path):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))
    # Onda
    data, sampling_rate = librosa.load(path)
    librosa.display.waveshow(y=data, sr=sampling_rate,
                             color="#A300F9", ax=axes[0][0])
    axes[0][0].set_title('Waveshow')

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=data, hop_length=hop_length)[0]
    axes[0][1].plot(zero_crossing_rate, color="#A300F9")
    axes[0][1].set_title("Zero Crossing Rate")

    # Transformada de Fourier
    stft_data = np.abs(librosa.stft(
        y=data, n_fft=n_fft, hop_length=hop_length))
    axes[0][1].plot(stft_data, color="#A300F9")
    axes[0][1].set_title('STFT')
    #print("STFT:", stft_data)

    # Espectograma
    DB = librosa.amplitude_to_db(stft_data, ref=np.max)
    img = librosa.display.specshow(DB, sr=sampling_rate, hop_length=hop_length,
                                   x_axis='time', y_axis='log', cmap='cool', ax=axes[0][2])

    # Espectograma de Mel
    mel_spec = librosa.feature.melspectrogram(
        data, sr=sampling_rate, hop_length=hop_length)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sampling_rate, hop_length=hop_length,
                                   x_axis='time', y_axis='log', cmap='cool', ax=axes[1][0])
    fig.colorbar(img, ax=axes[1][0])
    axes[1][0].set_title("Mel Spectogram")

    # Coeficientes de las frecuencias de Mel
    mfcc_data = np.abs(librosa.feature.mfcc(data, sr=sampling_rate))
    img = librosa.display.specshow(
        mfcc_data, sr=sampling_rate, x_axis='time', y_axis='log', cmap='cool', ax=axes[1][1])
    fig.colorbar(img, ax=axes[1][1])
    axes[1][1].set_title("Mel Frequency Cepstral Coefficients")

    # Chromagram
    chromagram = librosa.feature.chroma_stft(
        data, sr=sampling_rate, hop_length=hop_length)
    img = librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma',
                                   hop_length=hop_length, cmap='coolwarm', ax=axes[1][2])
    fig.colorbar(img, ax=axes[1][2])
    axes[1][2].set_title("Chromagram")

    # Bando de ancha espectral
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

    # Tempo y Beats
    tempo, beat_times = librosa.beat.beat_track(
        y=data, sr=sampling_rate, units='time')
    axes[2][1].vlines(beat_times, np.min(data), np.max(data), color='r')
    axes[2][1].set_title("(Tempo - " + str(round(tempo, 2)) + ")")

    # Rolloff espectral
    spectral_rolloff = librosa.feature.spectral_rolloff(
        data, sr=sampling_rate)[0]
    frames = range(len(spectral_rolloff))
    t = librosa.frames_to_time(frames)
    axes[2][2].plot(t, sklearn.preprocessing.minmax_scale(
        spectral_rolloff, axis=0), color='#FFB100')
    axes[2][2].set_title("Spectral Rolloff")

    # Componentes de percusión y armónicos
    y_harm, y_perc = librosa.effects.hpss(data)
    axes[3][0].plot(y_harm, color='#A300F9')
    axes[3][0].plot(y_perc, color='#FFB100')
    axes[3][0].set_title("Harmonic and Percussive Components")

    # Root-Mean-Square
    rms = librosa.feature.rms(y=data)
    axes[3][1].plot(rms, color='#A300F9')
    axes[3][1].set_title("Root-Mean-Square")

    plt.tight_layout(pad=5)
    plt.savefig("Graphs.png")

# Función de entrenamiento del modelo
def training():
    random_state = 42

    # Clasificadores a utilizar
    classifier = [
        MLPClassifier(),
        RandomForestClassifier(random_state=random_state),
        SVC(random_state=random_state, probability=True),
        KNeighborsClassifier(),
    ]

    # Parámetros del multiperceptrón
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

    # Parámetros de Random Forest
    rf_param_grid = {
        'max_features': [1, 3, 10],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [False],
        'n_estimators': [100, 300],
        'criterion': ['gini']
    }

    # Parámetros de la máquina de Soporte Vectorial
    svc_param_grid = {
        'kernel': ['rbf'],
        'gamma': [0.001, 0.01, 0.1, 1],
        'C': [1, 10, 50, 100, 200, 300, 1000]
    }

    # Parámetros de KNN
    knn_param_grid = {
        'n_neighbors': np.linspace(1, 20, 10, dtype=int).tolist(),
        'weights': ['uniform', 'distance'],
        'metric': ['eulidean', 'manhattan']
    }

    # Unión
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
    
    # Para cada clasificador empieza el entrenamiento
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

    # Clasificador que reúne a todos los anteriores
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
    all_accuracies.append(Y_test, y_prediction)
    plot_cm('Ensemble Learning', Y_test, y_prediction)
    plt.figure(figsize=(15, 20))
    cv_results = pd.DataFrame({'Cross Validation Means': cv_result, 'ML Models': [
        'MLP Classifier', 'RandomForestClassifier', 'SVM', 'KNeighborsClassifier']})
    g = sns.barplot(y='Cross Validation Means', x='ML Models', data=cv_results)
    g.set_xlabel('Mean Accuracy')
    g.set_title('Cross Validation Scores')
    # Resultados del entrenamiento
    plt.savefig("Results.png")


# Función para graficar las matrices de confusión y mostrar por consola el reporte de clasificación
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

# Predict de los modelos, recibe la ruta del audio seleccionado y retorna el género por modelo
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
    X_predict = scaler.transform(np.array(predict_df.iloc[:,:], dtype=float))
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
    results_genre = [str(le.inverse_transform(svm_predict)), str(le.inverse_transform(rfc_predict)), str(le.inverse_transform(mlp_predict)), str(le.inverse_transform(knn_predict)), str(le.inverse_transform(voting_predict),
    ), str(all_accuracies[0]), str(all_accuracies[1]), str(all_accuracies[2]), str(all_accuracies[3]), str(all_accuracies[4])]
    return results_genre

# Función que genera archivos Pickle (.pkl) para el entrenamiento del modelo y facilitar las pruebas
def pack_up():
    joblib.dump(best_estimator[0], './Pickle/MLP.pkl', compress=1)
    joblib.dump(best_estimator[1], './Pickle/Random Forest.pkl', compress=1)
    joblib.dump(best_estimator[2], './Pickle/SVM.pkl', compress=1)
    joblib.dump(best_estimator[3], './Pickle/KNN.pkl', compress=1)
    joblib.dump(best_estimator[4], './Pickle/Todos.pkl', compress=1)

# Función que utiliza los archivos Pickle (.pkl) generados para hacer las predicciones y no ejecutar el entrenamiento cada vez que se corra la aplicación
def unpack():
    best_estimator.append(joblib.load('./Pickle/MLP.pkl'))
    best_estimator.append(joblib.load('./Pickle/Random Forest.pkl'))
    best_estimator.append(joblib.load('./Pickle/SVM.pkl'))
    best_estimator.append(joblib.load('./Pickle/KNN.pkl'))
    best_estimator.append(joblib.load('./Pickle/Todos.pkl'))

# Verificación de los archivos Pickle
# Si no existen, inicia el entrenamiento y se generan para hacer las predicciones
# Si existen, se utilizan para correr la aplicación y hacer la predicción
if not os.path.exists('./Pickle/Todos.pkl'):
    print(datetime.datetime.now(), "Start training...")
    training()
    pack_up()
    print(datetime.datetime.now(), "Done training...")
else:
    unpack()

