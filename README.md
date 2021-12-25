# Instrument Recognition Web App
This is a machine learning project that aims to implement a multi-class classifier that identifies instruments in music streams. Our model consists of a CNN which’s input is an audio stream that we pre-process to extract the mel-spectogram, and outputs the dominance or non-dominance of pre-selected instruments. 

Our focus was to study on 10 instruments, and thus classify audio streams into one of those classes.

The instruments are as follows:-
Acoustic guitar, Bass drum, Cello, Clarinet, Double bass, Flute, Hi hat, Saxophone, Snare drum and Voilin.

I have used 'streamlit' to build the front-end of the application.

Sequence of files:
1. eda.py
2. model.py
3. webpage.py

 All the resources are available here:
 https://drive.google.com/drive/folders/1aETBD_J5hV9q-N4HDF91muRa-z2VMIw_?usp=sharing
