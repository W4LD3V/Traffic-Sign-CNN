from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from flask import Flask, request, render_template, redirect, url_for, flash
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for generating plots
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import os
import cv2
import shutil
from PIL import Image
import warnings
import re

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)

print("All imports successful")

import tensorflow as tf
print("TensorFlow Version:", tf.__version__)

# Create the Flask application instance
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Secret key for managing sessions and flash messages

# Load the pre-trained model without loading the optimizer
model = load_model('traffic_sign_model.keras', compile=False)
# Compile the model with the specified loss function and optimizer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the traffic sign classes (adjust based on your dataset)
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (90km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing zone',
    10: 'No passing for vehicles with a total weight of over 3.5 t.',
    11: 'Upcoming intersection or crossing',
    12: 'Priority Road starts',
    13: 'Right of Way',
    14: 'Stop and yield',
    15: 'No entry for any type of Vehicle',
    16: 'No entry for motor vehicles with a maximum authorized mass of more than 3.5 t.',
    17: 'Do not enter',
    18: 'General danger or warning sign',
    19: 'A single curve is approaching in the left direction',
    20: 'A single curve is approaching in the right direction',
    21: 'Indicates an approaching double curve - first to the left.',
    22: 'Warning of a rough road ahead.',
    23: 'Danger of skidding or slipping',
    24: 'The road narrows from the right side.',
    25: 'Work in process.',
    26: 'Indicates the traffic signal ahead.',
    27: 'Pedestrians may cross the road - installation on the right side of the road.',
    28: 'Pay attention to children - installation on the right side of the road.',
    29: 'Be aware of cyclists, - installation on the right side of the road.',
    30: 'Beware of an icy road ahead.',
    31: 'Indicates wild animals may cross the road.',
    32: 'End of all previously set passing and speed restrictions.',
    33: 'Indicates that traffic must turn right (after the signboard).',
    34: 'Indicates that traffic must turn left (after the signboard).',
    35: 'The mandatory direction of travel is straight ahead. No turns are permitted.',
    36: 'Mandatory directions of travel, straight ahead or right (after the signboard).',
    37: 'Mandatory directions of travel, straight ahead or left (after the signboard).',
    38: 'Prescribed drive direction around the obstacle. Drive from the right of the obstacle.',
    39: 'Prescribed drive direction around the obstacle. Drive from the left of the obstacle.',
    40: 'Indicates entrance to a traffic circle (roundabout).',
    41: 'End of the no-passing zone.',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Configure the upload folder for saving uploaded files
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create the folder if it doesn't exist

USER_DATA_FOLDER = 'static/uploads/user_data'  # Base path for user data folders

# Allowed file extensions for uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'ppm'}

# Configure the Flask app with the upload folder and other settings
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['USER_DATA_FOLDER'] = USER_DATA_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max file size

# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# SQLAlchemy setup for managing the database
Base = declarative_base()

# Define a database model for storing user-uploaded training images
class UserTrafficSign(Base):
    __tablename__ = 'user_train_data'
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    class_id = Column(Integer)
    image_path = Column(String)

# Define a database model for storing training metrics and related plots
class TrainingMetrics(Base):
    __tablename__ = 'training_metrics'
    id = Column(Integer, primary_key=True)
    model_name = Column(String)
    accuracy = Column(Float)
    val_accuracy = Column(Float)
    loss = Column(Float)
    val_loss = Column(Float)
    date_trained = Column(DateTime, default=datetime.utcnow)
    accuracy_plot_path = Column(String)
    loss_plot_path = Column(String)
    confusion_matrix_plot_path = Column(String)

# Function to save a plot to a file
def save_plot(plot_func, plot_name, history_or_data, *args):
    plt.figure(figsize=(10, 5))
    plot_func(history_or_data, *args)
    
    # Save the plot to the static/plots directory
    plot_path = os.path.join('static', 'plots', plot_name)
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

# Function to plot training and validation accuracy
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

# Function to plot training and validation loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

# Setup the database connection and session
engine = create_engine('sqlite:///traffic_signs.db')
Base.metadata.create_all(engine)  # Ensure the tables are created
Session = sessionmaker(bind=engine)
session = Session()

@app.route('/', methods=['GET', 'POST'])
def index():
    # Fetch available models from the models directory
    model_files = [f for f in os.listdir('models') if f.endswith('.keras')]

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')  # Show a message if no file is uploaded
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')  # Show a message if no file is selected
            return redirect(request.url)
        
        selected_model = request.form.get('model_select')  # Get selected model name from form
        if selected_model:
            model_path = os.path.join('models', selected_model)
            model = load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            flash('No model selected')  # Show a message if no model is selected
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)  # Save the uploaded file

            try:
                img = Image.open(filepath)
                img.verify()  # Verify that the image is valid
            except (IOError, SyntaxError):
                flash('Invalid image file')  # Show a message if the file is not a valid image
                os.remove(filepath)  # Remove the invalid file
                return redirect(request.url)

            img = Image.open(filepath).convert('L')  # Convert the image to grayscale
            img = img.resize((32, 32))  # Resize the image to 32x32 pixels
            img = np.array(img) / 255.0  # Normalize the image data
            img = np.expand_dims(img, axis=0)  # Add a batch dimension
            img = np.expand_dims(img, axis=-1)  # Add a channel dimension

            predictions = model.predict(img)  # Get predictions from the model
            predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class
            prediction_text = classes.get(predicted_class, "Unknown Class")  # Get the class name

            return render_template('index.html', filepath=filepath, prediction=prediction_text, model_files=model_files)

    return render_template('index.html', model_files=model_files, classes=classes)

@app.route('/upload_train', methods=['POST'])
def upload_train():
    # Clear the user_train_data table before each new session
    session.query(UserTrafficSign).delete()
    session.commit()

    if 'train_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('train_file')
    class_id = request.form.get('class_id')

    # Create a new user_data_X folder for this session
    user_data_folders = [folder for folder in os.listdir('static/uploads/') if re.match(r'user_data_\d+', folder)]
    new_session_folder = f"user_data_{len(user_data_folders)}"
    new_session_path = os.path.join(app.config['UPLOAD_FOLDER'], new_session_folder)
    os.makedirs(new_session_path, exist_ok=True)

    if len(files) > 50:
        flash('You can only upload up to 50 files at once.')
        return redirect(request.url)
    
    if not class_id:
        flash('Class ID is required')
        return redirect(request.url)
    
    class_id = int(class_id)  # Convert class ID to an integer
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            class_folder = os.path.join(new_session_path, f'class_{class_id}')
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filepath = os.path.join(class_folder, filename)
            file.save(filepath)

            try:
                img = Image.open(filepath)
                img.verify()  # Verify the image is valid
            except (IOError, SyntaxError):
                flash('Invalid image file')
                os.remove(filepath)
                continue  # Skip to the next file
            
            # Save the image details in the database
            new_sign = UserTrafficSign(
                filename=filename,
                class_id=class_id,
                image_path=filepath
            )
            session.add(new_sign)
            session.commit()
    
    flash('Training images uploaded successfully!')
    return redirect(url_for('index'))

@app.route('/clear_data', methods=['POST'])
def clear_data():
    # Find all user_data_X folders
    user_data_folders = [folder for folder in os.listdir('static/uploads/') if re.match(r'user_data_\d+', folder)]

    if not user_data_folders:
        flash('No uploaded data to clear.')
        return redirect(url_for('index'))

    # Sort folders to identify the most recent one
    user_data_folders.sort(key=lambda x: int(re.search(r'\d+', x).group()), reverse=True)
    
    # Get the most recent folder
    most_recent_folder = user_data_folders[0]
    most_recent_folder_path = os.path.join('static/uploads/', most_recent_folder)

    # Delete the most recent user_data_X folder
    shutil.rmtree(most_recent_folder_path)

    # Delete the corresponding records from the database
    session.query(UserTrafficSign).filter(UserTrafficSign.image_path.like(f'%{most_recent_folder}%')).delete(synchronize_session=False)
    session.commit()

    flash(f'Deleted the most recent uploaded data: {most_recent_folder}')
    return redirect(url_for('index'))

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    # Get the model name from the form
    model_name = request.form.get('model_name', 'traffic_sign_model')
    print(f"Model selected for fine-tuning: {model_name}")

    # Ensure the model name has the correct extension
    if not model_name.endswith('.keras'):
        model_name += '.keras'

    # Build the model path
    model_path = os.path.join('models', model_name)
    print(f"Attempting to load model from: {model_path}")

    # Check if the model file exists before loading
    if not os.path.exists(model_path):
        print(f"Model file does not exist at: {model_path}")
        flash(f"Model file does not exist at: {model_path}")
        return redirect(url_for('index'))

    # Load the pre-trained model
    try:
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        print(f"Error loading model: {e}")
        flash(f"Error loading model: {e}")
        return redirect(url_for('index'))

    # Load original training data from the database
    df_train = pd.read_sql('SELECT * FROM train_data', engine)

    # Fetch all user-uploaded training data
    df_user_train = pd.read_sql('SELECT * FROM user_train_data', engine)

    if df_user_train.empty:
        flash('No user-uploaded data available to retrain the model.')
        return redirect(url_for('index'))

    # Preprocess images
    def preprocess_image(image_path, target_size=(32, 32)):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, target_size)
            image = image / 255.0
            return image
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    # Combine original and user-uploaded data
    X_train, y_train = [], []

    for _, row in df_train.iterrows():
        img = preprocess_image(row['ImagePath'])
        if img is not None:
            X_train.append(img)
            y_train.append(row['ClassId'])

    for _, row in df_user_train.iterrows():
        img = preprocess_image(row['image_path'])
        if img is not None:
            X_train.append(img)
            y_train.append(row['class_id'])

    X_train = np.array(X_train).reshape(-1, 32, 32, 1)
    y_train = np.array(y_train)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
    )
    datagen.fit(X_train)

    # Implement learning rate reduction on plateau
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    # Split the data manually into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Use flow for training with data augmentation
    train_generator = datagen.flow(X_train, y_train, batch_size=32)

    print("Starting model training...")
    history = model.fit(train_generator,
                        validation_data=(X_val, y_val),
                        epochs=2,
                        callbacks=[lr_reduction])
    print("Model training finished.")

    # Get the new model name from the form (must be done before saving plots)
    new_model_name = request.form['new_model_name']

    # Ensure the plots directory exists
    plots_dir = os.path.join('static', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Save the accuracy, loss, and confusion matrix plots
    accuracy_plot_path = save_plot(plot_accuracy, f"{new_model_name}_accuracy.png", history)
    loss_plot_path = save_plot(plot_loss, f"{new_model_name}_loss.png", history)
    
    y_pred = model.predict(X_val).argmax(axis=1)
    confusion_matrix_plot_path = save_plot(plot_confusion_matrix, f"{new_model_name}_confusion_matrix.png", y_val, y_pred, list(classes.values()))

    # Save the fine-tuned model
    new_model_path = os.path.join('models', f"{new_model_name}_finetuned.keras")
    model.save(new_model_path)

    # Insert training metrics and plot paths into the database
    metrics = TrainingMetrics(
        model_name=new_model_name,
        accuracy=history.history['accuracy'][-1],
        val_accuracy=history.history['val_accuracy'][-1],
        loss=history.history['loss'][-1],
        val_loss=history.history['val_loss'][-1],
        date_trained=datetime.utcnow(),
        accuracy_plot_path=accuracy_plot_path,
        loss_plot_path=loss_plot_path,
        confusion_matrix_plot_path=confusion_matrix_plot_path
    )
    session.add(metrics)
    session.commit()

    flash(f'Model fine-tuned and saved as {new_model_name}_finetuned.keras successfully!')
    return redirect(url_for('index'))

@app.route('/metrics')
def show_metrics():
    metrics = session.query(TrainingMetrics).all()
    return render_template('metrics.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
