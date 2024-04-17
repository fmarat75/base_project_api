from fastapi import FastAPI
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import base64
from my_model.model_main import get_predictions, get_all_trees
import pandas as pd
import numpy as np

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}


## http://127.0.0.1:8000/predict?day_of_week=5&time=5
@app.get('/predict')
def predict(day_of_week, time):

    wait_prediction = get_predictions(day_of_week, time)

    return {'wait': wait_prediction}


## http://127.0.0.1:8000/counttrees?lat=40.4248077&long=-3.6938166&step=1&api_key=XXXXXXXXXXXXXXXXXXXXx
@app.get('/counttrees')
def count_trees(lat, long, step, api_key):

    trees_df, tree_img = get_all_trees(lat, long, step, api_key)

    ## change DF into JSON
    trees_json = trees_df.to_json(orient='records')

    ## change numpy array image to a PIL Image, then to bytes in PNG format
    ##tree_img_rgb = tree_img[:, :, ::-1]
    pil_img = Image.fromarray(tree_img.astype(np.uint8))  # Ensure the image array is of type uint8
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    byte_data = buf.getvalue()
    img_base64 = base64.b64encode(byte_data).decode('utf-8')

    return JSONResponse(content={"trees": trees_json, "image": img_base64})
