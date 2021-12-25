
# Importing all the required libraries

import streamlit as st

import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import pickle

from keras.models import load_model
from sklearn.metrics import accuracy_score

# Working on Front-End

page_bg_img = '''
<style>
.main{
background-image: url("https://cdn.pixabay.com/photo/2017/10/30/16/38/music-2902891_960_720.jpg");
background-size: cover;
font-family: verdana;
}
</style>
'''

title = '<h1 style="font-family: verdana; text-align: center; color: #525252; margin: 0px; padding: 0px;">INSTRUMENT RECOGNISER</h1>'

txt1 = '<hr><br>'

txt2 = '<h3 style="font-family: verdana; font-size: 1rem; color: #713f12; text-align: center;">Music is part of our daily life.<br>Therefore, use this multi-class classifier to identify different instruments.</h3>'

txt3 = '<h2 style="font-family: verdana; color: #525252; font-size: 1.2rem; text-align: center;">TRY IT OUT!</h2>'

st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(title, unsafe_allow_html=True)
st.markdown(txt1, unsafe_allow_html=True)
st.markdown(txt2, unsafe_allow_html=True)
st.markdown(txt3, unsafe_allow_html=True)

# Defining the functions

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')


def build_predictions(audio):
    y_pred = []

    rate, wav = wavfile.read(audio)
    y_prob = []

    for i in range(0, wav.shape[0]-config.step, config.step):
        sample = wav[i:i+config.step]
        x = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        x = (x - config.min) / (config.max - config.min)

        if config.mode == 'conv':
            x = x.reshape(1, x.shape[0], x.shape[1], 1)
        elif config.mode == 'time':
            x = np.expand_dims(x, axis=0)
        y_hat = model.predict(x)
        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat))

    fn_prob = np.mean(y_prob, axis=0).flatten()

    return y_pred, fn_prob

# Prediction

df = pd.read_csv('instruments.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles', 'conv.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

model = load_model(config.model_path)


# Uploading the audio file and printing the output

uploaded_file = st.file_uploader("Choose an audio file")
if uploaded_file is not None:
    y_pred, fn_prob = build_predictions(uploaded_file)
    y_prob = fn_prob
    y_pred = classes[np.argmax(y_prob)]
    ans = '<h3 style="font-family: verdana; font-size: 1.2rem; color: #713f12; text-align: center;">The instrument is: <u>' + y_pred + '</u></h3>'
    st.markdown(ans, unsafe_allow_html=True)