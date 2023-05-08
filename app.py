import torch
import pandas as pd
import xgboost as xgb
import streamlit as st
import numpy as np

from PIL import Image
from constants import transform
from models import ConvVAE

model_ld = ConvVAE(512)

# Load the checkpoint file
checkpoint = torch.load('vae_insulet.ckpt')

# Load the model state_dict from the checkpoint
model_ld.load_state_dict(checkpoint)
model_ld.eval()

# Load the saved xgboost from file
xgbmodel = xgb.Booster(model_file='xgbmodel.bin')

test_metadata = pd.read_csv('test.csv')
test_metadata['date'] = pd.to_datetime(test_metadata['date'])
test_metadata['year'] = test_metadata['date'].dt.year
test_metadata['month'] = test_metadata['date'].dt.month


# Set page title
st.set_page_config(page_title="Insulet inference app")

# Set app title
st.title("Insulet inference app")

files_list = test_metadata['image'].tolist()

selected_file = st.selectbox('Select a file', files_list)
# Create file uploader
if selected_file:
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If file uploaded
#if uploaded_file is not None:
    # Load image
    image = Image.open(selected_file)

    # Display image
    st.image(image, caption="Uploaded Image", use_column_width=True)


    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        code = model_ld.encoder(img_tensor).squeeze().numpy()

    st.write(code)


    df = pd.DataFrame(data=np.vstack(code))
    test_dataset = test_metadata[test_metadata['image']==selected_file]
    st.write(test_dataset.shape)
    test_dataset = test_dataset[['year','month','bar', 'baz', 'xgt', 'qgg', 'lux',
       'wsg', 'yyz', 'drt', 'gox', 'foo', 'boz', 'fyt', 'lgh', 'hrt',
      'juu']].copy().values
    codes_dataset = np.concatenate([np.vstack(code), test_dataset], axis=1)
    st.write(codes_dataset.shape)

    if 'target' in codes_dataset.columns:
        X = codes_dataset.drop('target', 1)
    else:
        X = codes_dataset

    print("X shape: ", X.shape)
    data_dmatrix = xgb.DMatrix(X)
    preds_test = xgbmodel.predict(data_dmatrix)
    st.write("prediction: ", preds_test)
    # TODO: Perform inference on image_np with your ML model and display the results
    # prediction = model.predict(image_np)
    # st.write("Prediction:", prediction)

