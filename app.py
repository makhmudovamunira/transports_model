import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath



#title
st.title('Transportlarni klassifikatsiya qiluvchi model')

file=st.file_uploader('rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])

if file:
  st.image(file)
  #PIL convert
  img=PILImage.create(file)
  #model
  model=load_learner('transport_model.pkl')

  #prediction
  pred, pred_id, probs=model.predict(img)
  st.success(f"bashorat {pred}")
  st.info(f"Ehtimollik {probs[pred_id]*100:.2f}%")                   

px.bar(x=probs*100, y=model.dls.vocab)
st.plotly_chart(fig)
