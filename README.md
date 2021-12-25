# Instrument Recognition Web App
This is a machine learning project that aims to implement a multi-class classifier that identifies instruments in music streams. Our model consists of a CNN which’s input is an audio stream that we pre-process to extract the mel-spectogram, and outputs the dominance or non-dominance of pre-selected instruments. 

Our focus was to study on 10 instruments, and thus classify audio streams into one of those classes.

The instruments are as follows:-
Acoustic guitar, Bass drum, Cello, Clarinet, Double bass, Flute, Hi hat, Saxophone, Snare drum and Voilin.

I have used 'streamlit' to build the front-end of the application. It is an open-source app framework for Machine Learning and Data Science teams. It creates beautiful data apps in hours. All in pure Python.

Sequence of execution of files:
1. eda.py
2. model.py
3. webpage.py
 
I have built this project with the help of a YouTube tutorial by Seth Adams. Here is the [link](https://youtube.com/playlist?list=PLhA3b2k8R3t2Ng1WW_7MiXeh1pfQJQi_P).

