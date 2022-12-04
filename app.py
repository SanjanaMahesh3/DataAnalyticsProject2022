import streamlit as st
import os
from predict import pred_feature_ext
from predict import predictor
from playsound import playsound
st.title('DA project')
up=st.file_uploader(label="choose an audio file",type=['.mp3'])

if up is not None:
    print(up)
    st.write(up.name)
    f = pred_feature_ext(up.name)
    k = predictor(f)
    st.write('Output class is ')
    st.write(k)
    playsound(os.path.abspath(up.name))

