from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
import config

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

# Process the uploaded image
def process_image(filepath):
    img = Image.open(filepath).convert('L')
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# Save plots to files
def save_plot(plot_func, plot_name, history_or_data, *args):
    plt.figure(figsize=(10, 5))
    plot_func(history_or_data, *args)
    plot_path = os.path.join('static', 'plots', plot_name)
    plt.savefig(plot_path)
    plt.close()
    return plot_path
