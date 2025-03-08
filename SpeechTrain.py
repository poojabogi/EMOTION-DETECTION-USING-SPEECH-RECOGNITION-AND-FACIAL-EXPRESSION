import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
import librosa
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import soundfile

path = 'SpeechEmotionDataset'

labels = []
X_train = []
Y_train = []
error = ['Actor_01/03-01-02-01-01-02-01.wav','Actor_05/03-01-02-01-02-02-05.wav','Actor_20/03-01-03-01-02-01-20.wav','Actor_20/03-01-06-01-01-02-20.wav']

def extract_feature(file_name, mfcc, chroma, mel): #extract features function e=to extract MFCC data from audio file
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32") #read sound data
        sample_rate=sound_file.samplerate #identifying sample rate from audio
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0) #extracting mfcc data from audio files
            result=np.hstack((result, mfccs)) #add extracted features to result variable
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    sound_file.close()        
    return result #return result to caller function

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name+"/"+directory[j] not in error: #looping all audio files from dataset
            mfcc = extract_feature(root+"/"+directory[j], mfcc=True, chroma=True, mel=True)#calling extract features method to extract MFCC data from all audio files
            X_train.append(mfcc) #adding emotion features data to train array variable
            arr = directory[j].split("-")
            Y_train.append(int(arr[2])) #finding emotion values from dataset file name from position 2
            print(name+" "+root+"/"+directory[j]+" "+str(mfcc.shape)+" "+str(int(arr[2])))
       
        
        
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
print(Y_train)

X_train = X_train.astype('float32')
X_train = X_train/255

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1,1))
print(X_train.shape)
    
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
Y_train = to_categorical(Y_train)
#np.save('model/speechX.txt',X_train)
#np.save('model/speechY.txt',Y_train)

X_train = np.load('model/speechX.txt.npy')
Y_train = np.load('model/speechY.txt.npy')
print(X_train.shape)
print(Y_train.shape)
    
if os.path.exists('model/speechmodel.json'):
    with open('model/speechmodel.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/speech_weights.h5")
    classifier._make_predict_function()   
    print(classifier.summary())
    f = open('model/speechhistory.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))
else:
    classifier = Sequential() #creating sequential object as classifier
    #creating CNN layer with 32 neurons or filters and giving input shape size and this data will be filtered by cnn 32 times
    classifier.add(Convolution2D(32, 1, 1, input_shape = (180, 1, 1), activation = 'relu'))
    #defining max pooling layer to extract important features from dataset
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    #creating another layer with 32 filters
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    #defining max pooling layer to extract important features from dataset
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    #converting multidimensional data to single dimensional data
    classifier.add(Flatten())
    #defining output layer
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    #output layer has to predict values as per given in Y data
    classifier.add(Dense(output_dim = Y_train.shape[1], activation = 'softmax'))
    #print summary of CNN
    print(classifier.summary())
    #compiling CNN model
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #start training CNN with given X and Y data
    hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=100, shuffle=True, verbose=2)
    '''
    classifier.save_weights('model/speech_weights.h5')
    model_json = classifier.to_json()
    with open("model/speechmodel.json", "w") as jsonFile:
        jsonFile.write(model_json)
    jsonFile.close() 
    f = open('model/speechhistory.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    f = open('model/speechhistory.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))
    '''
