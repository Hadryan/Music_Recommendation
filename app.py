from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
from bisect import bisect_left, bisect_right
import sklearn
import os
import scipy
import math
import librosa
from pydub import AudioSegment
from scipy.io import wavfile
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

app = Flask(__name__)

# model=pickle.load(open('model.pkl','rb'))
def cancel_zeros(str_array):
    ret_arr = []
    for i in range(0, len(str_array)):
        ret_arr.append(str(int(str_array[i])))
    return ret_arr

def get_usr_pref(temp_arr, all_feature_array):
    usr_pref_array = []
    for i in range(0,len(temp_arr)):
        for j in range(0, len(all_feature_array)):
            if (temp_arr[i] == str(all_feature_array[j][0])):
                usr_pref_array.append(all_feature_array[j])    
    return usr_pref_array

def get_recommendation(usr_pref_array, all_feature_array):
    print("USR", usr_pref_array)
    X = np.array(usr_pref_array)
    print("X=", X)
    X = X[:, 1:4].astype(float)
    X = np.array(X)
    RADIUS_THRESH = 2
    max_radius = RADIUS_THRESH + 1
    n_clusters = 0
    # Increase the number of clusters till the largest cluster radius < RADIUS_THRESH
    while (max_radius > RADIUS_THRESH):
      n_clusters = n_clusters + 1
      # Use the k-means classifier
      kmeans = KMeans(n_clusters, random_state=0).fit(X)
      max_radius = 0
      # Compute the max_radius
      for i in range(0, len(X)):
        centre_no = kmeans.labels_[i]
        centre_coor = kmeans.cluster_centers_[centre_no]
        dist = (X[i][0]-centre_coor[0])**2 + (X[i][1]-centre_coor[1])**2 + (X[i][2]-centre_coor[2])**2
        dist = math.sqrt(dist)  # Euclidean distance between data-point in X and its cluster centre  
        if dist > max_radius:
          max_radius = dist  
    print("Final number of clusters formed are: ", n_clusters)
    print("Cluster allocation is: ", kmeans.labels_, "respectively")
    BOX_SIZE_FACTOR = 50 # Length of side of sq = factor*RADIUS_THRESH
    start_index = []
    end_index = []
    for i in range(0, n_clusters):
        x_min = kmeans.cluster_centers_[i][0]-((BOX_SIZE_FACTOR/2)*RADIUS_THRESH)
        x_max = kmeans.cluster_centers_[i][0]+((BOX_SIZE_FACTOR/2)*RADIUS_THRESH)
        start_index.append(bisect_left(all_feature_array[:,1].astype(float), x_min))
        end_index.append(bisect_right(all_feature_array[:,1].astype(float), x_max))
    print(start_index)
    print(end_index)
    accepted_song = []
    for i in range(0, n_clusters):
        y_min = kmeans.cluster_centers_[i][1]-((BOX_SIZE_FACTOR/2)*RADIUS_THRESH)
        y_max = kmeans.cluster_centers_[i][1]+((BOX_SIZE_FACTOR/2)*RADIUS_THRESH)
        z_min = kmeans.cluster_centers_[i][2]-((BOX_SIZE_FACTOR/2)*RADIUS_THRESH)
        z_max = kmeans.cluster_centers_[i][2]+((BOX_SIZE_FACTOR/2)*RADIUS_THRESH)
        for j in range(start_index[i], end_index[i]):
            if(all_feature_array[j][2]<= y_max and all_feature_array[j][2] >= y_min):
                if(all_feature_array[j][3] <= z_max and all_feature_array[j][3]>= z_min):
                    accepted_song.append(all_feature_array[j])
    new_recommendation = []
    for i in range(0, len(accepted_song)):
        usr_pref = False
        for j in range(0, len(usr_pref_array)):
            if(accepted_song[i][0]==usr_pref_array[j][0]):
                usr_pref = True
        if (usr_pref == False):
            new_recommendation.append(accepted_song[i])
    print("No of new recommendations found=", len(new_recommendation))
    score = []
    new_recommendation_with_score = []
    for i in range(0, len(new_recommendation)):
        score.append(0)
        for j in range(0, n_clusters):
            centre_coor = kmeans.cluster_centers_[j]
            dist = (new_recommendation[i][1]-centre_coor[0])**2 + (new_recommendation[i][2]-centre_coor[1])**2 + (new_recommendation[i][3]-centre_coor[2])**2
            dist = math.sqrt(dist)  # Euclidean distance between song in new_recommendation and its cluster centre  
            temp = kmeans.labels_.tolist()
            n_cluster_points = temp.count(j)
            score[i] = score[i] + (1/(dist+0.000001))*(n_cluster_points/len(usr_pref_array))
        new_recommendation_with_score.append([new_recommendation[i][0], score[i]])
    new_recommendation_with_score = sorted(new_recommendation_with_score, key=lambda a:a[1], reverse=True)
    print("Your top recommendations are:")
    recommended_songs = []
    for i in range(0, len(new_recommendation_with_score)):
        print(i+1, ") Song Name.", new_recommendation_with_score[i][0])
        recommended_songs.append(str(i+1) + ") Song Name: " + str(new_recommendation_with_score[i][0]) + "_____________" +new_recommendation[i][4])
    return recommended_songs

@app.route('/get_recommendation/predict',methods=['POST','GET'])
def predict_recommendation():
    df = pd.read_csv('k_mean_feat.csv')
    all_feature_array = df.to_numpy()
    print(all_feature_array)
    temp = str([(x) for x in request.form.values()][0])
    temp_arr = cancel_zeros((temp.split(",")))
    usr_pref_array = get_usr_pref(temp_arr, all_feature_array)
    recommended_songs = get_recommendation(usr_pref_array, all_feature_array)
    print(temp)
    return render_template('result.html', ans1 = '{}'.format(recommended_songs[0]),
        ans2 = '{}'.format(recommended_songs[1]),
        ans3 = '{}'.format(recommended_songs[2]),
        ans4 = '{}'.format(recommended_songs[3]),
        ans5 = '{}'.format(recommended_songs[4]),
        ans6 = '{}'.format(recommended_songs[5]),
        ans7 = '{}'.format(recommended_songs[6]),
        ans8 = '{}'.format(recommended_songs[7]),
        ans9 = '{}'.format(recommended_songs[8]),
        ans10 = '{}'.format(recommended_songs[9]))
    # int_features=[int(x) for x in request.form.values()]
    # final=[np.array(int_features)]
    # print(int_features)
    # print(final)
    # prediction=model.predict_proba(final)
    # output='{0:.{1}f}'.format(prediction[0][1], 2)

    # if output>str(0.5):
    #     return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    # else:
    #     return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")

def spectral_flux(music_wave_data):
    # obtain the stft of the music_wave_data
    spectrum = librosa.core.stft(music_wave_data)
    N = spectrum.shape[0]

    # calculating the spectral flux
    sf = np.sqrt(np.sum((np.diff(np.abs(spectrum)))**2, axis=0)) / N

    return sf

def feature_extraction(audio_data):
    feature_list_all = []
    for i in range(0, len(audio_data)):
        feature_list = [audio_data[i][2]]
        y = audio_data[i][0]
        sr = audio_data[i][1]
        feature_list.append(np.mean(abs(y)))
        feature_list.append(np.std(y))
        feature_list.append(scipy.stats.skew(abs(y)))
        feature_list.append(scipy.stats.kurtosis(y))
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y + 0.0001, frame_length=2048, hop_length=512)[0]
        feature_list.append(np.mean(zcr))
        feature_list.append(np.std(zcr))    
        
        # RMSE
        rmse = librosa.feature.rmse(y + 0.0001)[0]
        feature_list.append(np.mean(rmse))
        feature_list.append(np.std(rmse))

        # Tempo
        tempo = librosa.beat.tempo(y, sr=sr)
        feature_list.extend(tempo)
        
        # Spectral Centroids
        spectral_centroids = librosa.feature.spectral_centroid(y+0.01, sr=sr)[0]
        feature_list.append(np.mean(spectral_centroids))
        feature_list.append(np.std(spectral_centroids))

        # Spectral Bandwidth
        spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr, p=2)[0]
        spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr, p=3)[0]
        spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr, p=4)[0]
        feature_list.append(np.mean(spectral_bandwidth_2))
        feature_list.append(np.std(spectral_bandwidth_2))
        feature_list.append(np.mean(spectral_bandwidth_3))
        feature_list.append(np.std(spectral_bandwidth_3))
        feature_list.append(np.mean(spectral_bandwidth_3))
        feature_list.append(np.std(spectral_bandwidth_3))
            
        # Spectral Flux
        sf = spectral_flux(y)
        sf_num = np.mean(sf)
        feature_list.append(sf_num) 

        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr, n_bands = 6, fmin = 200.0)
        feature_list.extend(np.mean(spectral_contrast, axis=1))
        feature_list.extend(np.std(spectral_contrast, axis=1))

        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr, roll_percent = 0.85)[0]
        feature_list.append(np.mean(spectral_rolloff))
        feature_list.append(np.std(spectral_rolloff))

        # MFCC
        mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20)
        feature_list.extend(np.mean(mfccs, axis=1))
        feature_list.extend(np.std(mfccs, axis=1))

        # STFT
        chroma_stft = librosa.feature.chroma_stft(y, sr=sr, hop_length=1024)
        feature_list.extend(np.mean(chroma_stft, axis=1))
        feature_list.extend(np.std(chroma_stft, axis=1))

        # Round off
        feature_list[1:] = np.round(feature_list[1:], decimals=3)
    feature_list_all.append(feature_list)
    return feature_list_all

def pca_extraction(X_data):
    X_data = StandardScaler().fit_transform(X_data)
    pca = pickle.load(open("models/pca.pkl",'rb'))
    principalComponents = pca.transform(X_data)
    return principalComponents


def load_model(filename):
    path = "models"
    path = os.path.join(path,filename)
    loaded_model = pickle.load(open(path,"rb"))
    return loaded_model

def save_song(song):
    song.save(os.path.join("./songs/", str(song.filename)))

def load_song():
    audio_data = []
    path = "songs"
    count = 0
    audio_data = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith('.mp3'):
                filepath = str(r)+ '/' + str(file)
                count += 1
                file_name = str(r)+ '/' + str(count) + ".wav"
                print(file_name)
                sound = AudioSegment.from_mp3(filepath)
                sound.export(file_name, format="wav")
                fs, data = wavfile.read(file_name)
                print (data)
                if (data.shape[1] == 2):
                    data = data[:,0]
                data = data.astype(float)
                audio_data.append([data, fs, filepath])
    # for r, d, f in os.walk(path):
    #     for file in f:
    #         if file.endswith('.mp3'):
    #             filepath = str(r)+ '/' + str(file)
    #             print(filepath)
    #             # try:
    #             y, sr = librosa.load(filepath, sr = 22050)
    #             audio_data.append([y, sr, filepath])
    #             # except:
    #             #     continue
    return audio_data

def get_genre(k):
    thisdict =  {
      0: "International",
      1: "Instrumental",
      2: "Pop",
      3: "Folk",
      4: "Hip-Hop",
      5: "Experimental",
      6: "Rock",
      7: "Electronic"
    }
    x = thisdict.get(k)
    return x

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/get_recommendation', methods=['POST','GET'])
def get_recommendation():
    return render_template("get_recommendation.html")

@app.route('/genre_prediction', methods=['POST','GET'])
def genre_prediction():
    if request.method == 'POST':
      f = request.files['file']
      print(f)
      save_song(f)
    return render_template("genre_prediction.html")

@app.route('/rf', methods=['POST','GET'])
def rf():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("rf.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all[:,1:])
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/rf_pca', methods=['POST','GET'])
def rf_pca():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("rf_pca.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all = pca_extraction(feature__list_all[:,1:])
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all)
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/rf_rfe', methods=['POST','GET'])
def rf_rfe():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("rf_rfe.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all[:,1:])
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/rf_pca_rfe', methods=['POST','GET'])
def rf_pca_rfe():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("rf_pca_rfe.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all = pca_extraction(feature__list_all[:,1:])
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all)
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/xgb', methods=['POST','GET'])
def xgb():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("xgb.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all[:,1:])
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/xgb_pca', methods=['POST','GET'])
def xgb_pca():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("xgb_pca.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all = pca_extraction(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all)
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/xgb_rfe', methods=['POST','GET'])
def xgb_rfe():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("xgb_rfe.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all[:,1:])
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/xgb_pca_rfe', methods=['POST','GET'])
def xgb_pca_rfe():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("xgb_pca_rfe.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all = pca_extraction(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all[:,1:])
        print (pred_probs)
    else:
        return "Could not open the song"
    #give output

@app.route('/svm', methods=['POST','GET'])
def svm():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("svm.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all[:,1:])
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/svm_pca', methods=['POST','GET'])
def svm_pca():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("svm_pca.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all = pca_extraction(feature__list_all[:,1:])
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all)
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/nn', methods=['POST','GET'])
def nn():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("nn.pickle")
        feature__list_all = np.array(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all[:,1:])
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/nn_pca', methods=['POST','GET'])
def nn_pca():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("nn_pca.pickle")
        feature__list_all = np.array(feature__list_all)
        feature__list_all = pca_extraction(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all[:,1:])
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/cnn', methods=['POST','GET'])
def cnn():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("cnn.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all[:,1:])
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/cnn_pca', methods=['POST','GET'])
def cnn_pca():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("cnn_pca.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all = pca_extraction(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all[:,1:])
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"

@app.route('/en', methods=['POST','GET'])
def en():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("cnn.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all[:,1:])
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"
    #give output

@app.route('/en_pca', methods=['POST','GET'])
def en_pca():
    audio_songs = load_song()
    if (audio_songs != []):
        feature__list_all = feature_extraction(audio_songs)
        loaded_model = load_model("en_pca.pkl")
        feature__list_all = np.array(feature__list_all)
        feature__list_all = pca_extraction(feature__list_all)
        feature__list_all.reshape(-1,1)
        print(feature__list_all)
        pred_probs = loaded_model.predict_proba(feature__list_all[:,1:])
        print (pred_probs)
        pred_probs = np.array(pred_probs[0])
        result = np.where(pred_probs == np.amax(pred_probs))
        print(int(result[0]))
        genre = get_genre(int(result[0])) 
        print("G=",genre)
        return render_template('output.html', genre = '{}'.format(genre))
    else:
        return "Could not open the song"

@app.route('/upload', methods=['POST','GET'])
def upload():
   return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)