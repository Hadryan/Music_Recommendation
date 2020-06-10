from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
from bisect import bisect_left, bisect_right
import sklearn
import math
from sklearn.cluster import KMeans

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

def feature_extraction(music_data)
    col_names = ['file_name', 'signal_mean', 'signal_std', 'signal_skew', 'signal_kurtosis', 
                'zcr_mean', 'zcr_std', 'rmse_mean', 'rmse_std', 'tempo',
                'spectral_centroid_mean', 'spectral_centroid_std',
                'spectral_bandwidth_2_mean', 'spectral_bandwidth_2_std',
                'spectral_bandwidth_3_mean', 'spectral_bandwidth_3_std',
                'spectral_bandwidth_4_mean', 'spectral_bandwidth_4_std', 'spectral_flux'] + \
                ['spectral_contrast_' + str(i+1) + '_mean' for i in range(7)] + \
                ['spectral_contrast_' + str(i+1) + '_std' for i in range(7)] + \
                ['spectral_rolloff_mean', 'spectral_rolloff_std'] + \
                ['mfccs_' + str(i+1) + '_mean' for i in range(20)] + \
                ['mfccs_' + str(i+1) + '_std' for i in range(20)] + \
                ['chroma_stft_' + str(i+1) + '_mean' for i in range(12)] + \
                ['chroma_stft_' + str(i+1) + '_std' for i in range(12)]
     
    df = pd.DataFrame(columns=col_names)
    count = 0
    for i in range(0, len(audio_data)):
        feature_list = [audio_data[i][2]]
        y = audio_data[i][0]
        
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

        # Append feature list to the feature data
        df = df.append(pd.DataFrame(feature_list, index=col_names).transpose(), ignore_index=True)

        count +=1
        print(count)
    #print(df)

def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))

def store_song(song):
    #save the file locally
    pass

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
      store_genre(f)
    return render_template("genre_prediction.html")

@app.route('/result', methods=['POST','GET'])
def output():
    #somehow get which button is clicked
    #and call feature extraction accordingly
    #then the saved model to be loaded
    #and compute the genre
    pass


@app.route('/upload', methods=['POST','GET'])
def upload():
   return render_template('upload.html')



if __name__ == '__main__':
    app.run(debug=True)