import streamlit as st
import torch
import torch.nn as nn
from helper_CNN.machine_vision_model1 import MachineVisionModel #Import machinevisionmodel

import warnings
warnings.filterwarnings('ignore') #Ignore warnings

print("torch version:", torch.__version__) #Used for test verification

model = MachineVisionModel(num_classes = 6, data_dir = r"C:\Users\willi\Desktop\MP2\magTileDefectData")

st.set_page_config(page_icon = ":fox_face:", layout="wide") #Set page config with a fox icon and a wide layout
st.title("Tile Defect Detector")

Tiles, Detect = st.columns([1, 1], gap = "small", border = True) #Creates 2 columns; Tiles and Detect

with Tiles: #Inputs column 
    st.header("Tiles") #Column headers 
    img = st.file_uploader(label = "Input tile images here") #File upload which saves under a variable
    but = st.button(label = "Run") #Button with the label "Run"
    sel = ["resnet", "vgg", "densenet"] #Multiselect options dont work outside of list
    sb = st.selectbox(label = 'Select Model to Use', options = sel) #Selects model based on sel list

with Detect: #Outputs columnm
    st.header("Outputs") #Column headers
    if but: #If button is pressed then run 
        if img: #If there is an image while button is pressed
            if sb == "resnet": #Quick if tree to load different models based on different selections
                model = torch.load(r"C:\Users\willi\Desktop\MP2\resnet_full.pth", weights_only = False)
            elif sb == "vgg":
                model = torch.load(r"C:\Users\willi\Desktop\MP2\vgg_full.pth", weights_only = False)
            elif sb == "densenet":
                model = torch.load(r"C:\Users\willi\Desktop\MP2\dense_full.pth", weights_only = False)
            else:
                st.write("Please select a machine vision model to use.")

            predicted_class, confidence = model.predict(img)

            st.write(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}')
            if confidence > 0.5:
                img = st.image(img, caption="Tile is defective") #Returns the image uploaded in the input column
                st.write("Using: ", {sb})
            else:
                img = st.image(img, caption="Tile is not defective") #If tile is ok
                st.write("Using: ", {sb})