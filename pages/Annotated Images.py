import streamlit as st
import os

# Specify the folder path
folder_path = "./annotated_images"

# Get a list of image files
image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")]

# Display the images in columns
cols = st.columns(3)
for i, image_file in enumerate(image_files):
    with cols[i % 3]:
        st.image(os.path.join(folder_path, image_file), caption=image_file)