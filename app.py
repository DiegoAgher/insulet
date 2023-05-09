import torch
import pandas as pd
import xgboost as xgb
import streamlit as st
import numpy as np

from PIL import Image
from constants import transform, xgb_feature_names
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
    path_selected_file = selected_file
    image = Image.open(path_selected_file)

    # Display image
    st.image(image, caption="Uploaded Image", use_column_width=True)


    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        code = model_ld.encoder(img_tensor).squeeze().numpy()

    st.write(code)

    df = pd.DataFrame(data=np.vstack(code).reshape(-1, 1024),
                      columns=[f'code_{i}' for i in range(1024)])
    test_dataset = test_metadata[test_metadata['image']==selected_file]
    st.write(test_dataset.shape)
    test_dataset = test_dataset[['year','month','bar','day_of_week',
     'baz', 'xgt', 'qgg', 'lux',
    'wsg', 'yyz', 'drt', 'gox', 'foo', 'boz', 'fyt', 'lgh', 'hrt',
    'juu']].copy()
    codes_dataset = pd.concat([df, test_dataset], axis=1)
    codes_dataset = codes_dataset[xgb_feature_names]
    st.write(codes_dataset.shape)

    X = codes_dataset
    preds_test = xgbmodel.predict(X)
    st.write("Model's prediction: ", preds_test[0])
    # TODO: Perform inference on image_np with your ML model and display the results
    # prediction = model.predict(image_np)
    # st.write("Prediction:", prediction)

