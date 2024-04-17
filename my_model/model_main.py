import numpy as np
import pandas as pd
import requests
import io
from PIL import Image
from deepforest import main
from deepforest import visualize
import math

IMG_ZOOM = 20 # Zoom level (0 to 21+), adjust as necessary for the desired detail
TILE_SIZE = 600 # Size of the image (max 640x640 for non-premium users)

def get_predictions(day_of_week, time):

    my_array = np.array([float(day_of_week), float(time)])
    my_df = pd.DataFrame(my_array, columns = ["time"])

    final_predict = my_df["time"][0] * my_df["time"][1]

    return final_predict


def download_satellite_image(lat, long, api_key):

    zoom = IMG_ZOOM
    size = str(TILE_SIZE)+'x'+ str(TILE_SIZE)

    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{long}&zoom={zoom}&size={size}&maptype=satellite&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:

        image = Image.open(io.BytesIO(response.content))

        # Convert the image to RGB format
        rgb_image = image.convert('RGB')

        # Convert the image to a numpy array
        np_image = np.array(rgb_image)

        # Change the type to float32
        np_image = np_image.astype(np.float32)

        return np_image

    else:
        raise Exception(f"Failed to download image: {response.status_code}")


def run_deepforest(image_data):

    model = main.deepforest()
    model.use_release()
    trees_df = model.predict_image(image=image_data, return_plot=False)
    tree_img = visualize.plot_predictions(image_data[:,:,::-1], trees_df)

    return trees_df, tree_img


def get_local_image(): ## TEMP ###

    image_data = Image.open('11.png')

    return image_data


def get_all_trees(lat, long, step, api_key):
    trees_all_df = pd.DataFrame()
    tree_all_img = np.empty(shape=(TILE_SIZE, TILE_SIZE*(2*step+1)), dtype=int)

    for y_step in np.arange(-step, step+1, 1):

        tree_all_x_img = np.empty(shape=(TILE_SIZE, TILE_SIZE), dtype=int)
        for x_step in np.arange(-step, step+1, 1):
            lat_step, lon_step = tile_coordinates(lat, long, IMG_ZOOM, x_step, y_step, TILE_SIZE)
            trees_df_step, tree_img_step = get_trees(lat_step, lon_step, api_key)

            tree_img_step_rgb = tree_img_step[:, :, ::-1]
            tree_all_x_img = np.concatenate(tree_all_x_img, tree_img_step_rgb, axis=1)

            trees_all_df = pd.concat([trees_all_df, trees_df_step], ignore_index=True)

        tree_all_img = np.concatenate(tree_all_img, tree_all_x_img, axis=0)


    return trees_all_df, tree_all_img



def get_trees(lat, long, api_key):


    image_data = download_satellite_image(lat, long, api_key)
    #image_data = get_local_image()

    trees_df, tree_img = run_deepforest(image_data)

    return trees_df, tree_img


def tile_coordinates(lat, lon, zoom, x_step, y_step, tile_size):
    # Convert lat/lon to pixels
    lat_rad = math.radians(lat)
    sin_lat = math.sin(lat_rad)
    pixels_x = (lon + 180) / 360 * (2 ** zoom * 256)
    pixels_y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * (2 ** zoom * 256)

    # Calculate the pixel shift
    new_pixels_x = pixels_x + (x_step * tile_size)
    new_pixels_y = pixels_y + (y_step * tile_size)

    # Convert the new pixel values back to lat/lon
    new_lon = new_pixels_x / (2 ** zoom * 256) * 360 - 180
    new_lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * new_pixels_y / (2 ** zoom * 256))))
    new_lat = math.degrees(new_lat_rad)

    return new_lat, new_lon
