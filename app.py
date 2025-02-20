import streamlit as st
from fastai.vision.all import *

#title
st.title('Transportlarni klassifikatsiya qiluvchi model')

file=st.file_uploader('rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])

if file:
  st.image(file)
  #PIL convert
                      
