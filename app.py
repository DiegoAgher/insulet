import torch
import pandas as pd
import xgboost as xgb
import streamlit as st
import numpy as np
import torchvision.utils as vutils

from PIL import Image
from constants import transform, xgb_feature_names, get_interactions
from models import ConvVAE

model_ld = ConvVAE(512)

# Load the checkpoint file
checkpoint = torch.load('vae_insulet_80epochs.ckpt')

# Load the model state_dict from the checkpoint
model_ld.load_state_dict(checkpoint)
model_ld.eval()

# Load the saved xgboost from file
xgbmodel = xgb.XGBRegressor()
xgbmodel.load_model("xgbmodel.bin")

# preprocess data
test_metadata = pd.read_csv('test.csv')
test_metadata['date'] = pd.to_datetime(test_metadata['date'])
test_metadata['year'] = test_metadata['date'].dt.year
test_metadata['month'] = test_metadata['date'].dt.month
test_metadata['day_of_week'] = test_metadata['date'].dt.day_of_week
test_metadata, interactions = get_interactions(test_metadata)


# Set page title
st.set_page_config(page_title="Insulet inference app")

# Set app title
st.title("Insulet inference app")

files_list = test_metadata['image'].tolist()
selected_file = st.selectbox('Select a file', files_list)

if selected_file:
    # Load image
    image = Image.open(selected_file)

    # Display image
    st.image(image, caption="Selected Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        code = model_ld.encoder(img_tensor).squeeze().numpy()
        recon = model_ld(img_tensor).squeeze()

    df = pd.DataFrame(data=np.vstack(code).reshape(-1, 1024),
                      columns=[f'code_{i}' for i in range(1024)])
    test_dataset = test_metadata[test_metadata['image']==selected_file]
    test_dataset = test_dataset[['year','month','bar','day_of_week',
                                 'baz', 'xgt', 'qgg', 'lux',
                                 'wsg', 'yyz', 'drt', 'gox', 'foo', 'boz',
                                  'fyt', 'lgh', 'hrt', 'juu'] + interactions].copy()
    codes_dataset = pd.concat([df, test_dataset.reset_index(drop=True)], axis=1)
    codes_dataset = codes_dataset[xgb_feature_names]

    X = codes_dataset
    preds_test = xgbmodel.predict(X)
    st.write("image latents")
    st.write(df)
    st.write("feats")
    st.write(X)
    st.write("Model's prediction: ", preds_test[0])

    button = st.button("Show Recon")
    image = st.empty()
    if button:
        with torch.no_grad():
            recon = np.transpose(recon, (1,2,0))

        st.image(vutils.make_grid(recon, normalize=True).cpu().numpy(), caption="Recon Image")
