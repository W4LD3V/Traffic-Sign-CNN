from flask import Blueprint, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from werkzeug.utils import secure_filename
from models.models import Session as DBSession
from services.training_service import process_image, allowed_file, save_plot
from models.models import Session, UserTrafficSign, TrainingMetrics
from constants import classes  # Import the classes dictionary
from datetime import datetime
import config
import os
from PIL import Image
import numpy as np
import re
import shutil
import cv2  # Required for image preprocessing
import pandas as pd  # Required for database handling
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for rendering plots
import matplotlib.pyplot as plt  # Required for plotting
from sklearn.metrics import confusion_matrix  # For confusion matrix
import seaborn as sns  # For confusion matrix heatmap
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup the database connection and session
engine = create_engine('sqlite:///traffic_signs.db')
Session = sessionmaker(bind=engine)

# Blueprint for routes
main_routes = Blueprint('main_routes', __name__)

# Load the pre-trained model
model = load_model(os.path.join('models', 'traffic_sign_model.keras'), compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Plot training and validation accuracy
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

# Plot training and validation loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

@main_routes.route('/', methods=['GET', 'POST'])
def index():
    # Fetch available models from the models directory
    model_files = [f for f in os.listdir('models') if f.endswith('.keras')]

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        selected_model = request.form.get('model_select')
        if selected_model:
            model_path = os.path.join('models', selected_model)
            model = load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            flash('No model selected')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(config.UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                img = Image.open(filepath)
                img.verify()  # Verify the image is valid
            except (IOError, SyntaxError):
                flash('Invalid image file')
                os.remove(filepath)
                return redirect(request.url)

            img = Image.open(filepath).convert('L')  # Convert to grayscale
            img = img.resize((32, 32))  # Resize the image
            img = np.array(img) / 255.0  # Normalize the image
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            img = np.expand_dims(img, axis=-1)  # Add channel dimension

            predictions = model.predict(img)
            predicted_class = np.argmax(predictions, axis=1)[0]
            prediction_text = classes.get(predicted_class, "Unknown Class")

            return render_template('index.html', filepath=filepath, prediction=prediction_text, model_files=model_files)

    return render_template('index.html', model_files=model_files, classes=classes)

# Upload new training images route
@main_routes.route('/upload_train', methods=['POST'])
def upload_train():
    session = DBSession()

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
    new_session_path = os.path.join(config.UPLOAD_FOLDER, new_session_folder)
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

    session.close()

    flash('Training images uploaded successfully!')
    return redirect(url_for('main_routes.index'))  # Updated to main_routes.index

# Clear uploaded data route (renamed function to avoid conflict)
@main_routes.route('/clear_data', methods=['POST'])
def clear_data():
    session = DBSession()

    # Find all user_data_X folders
    user_data_folders = [folder for folder in os.listdir('static/uploads/') if re.match(r'user_data_\d+', folder)]

    if not user_data_folders:
        flash('No uploaded data to clear.')
        return redirect(url_for('main_routes.index'))  # Updated to main_routes.index

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
    session.close()

    flash(f'Deleted the most recent uploaded data: {most_recent_folder}')
    return redirect(url_for('main_routes.index'))  # Updated to main_routes.index

@main_routes.route('/metrics')
def show_metrics():
    session = DBSession()

    # Perform the query
    metrics = session.query(TrainingMetrics).all()

    session.close()

    return render_template('metrics.html', metrics=metrics)

@main_routes.route('/retrain_model', methods=['POST'])
def retrain_model():
    session = DBSession()

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
        return redirect(url_for('main_routes.index'))  # Updated to main_routes.index

    # Load the pre-trained model
    try:
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        print(f"Error loading model: {e}")
        flash(f"Error loading model: {e}")
        return redirect(url_for('main_routes.index'))  # Updated to main_routes.index

    # Load original training data from the database
    df_train = pd.read_sql('SELECT * FROM train_data', engine)

    # Fetch all user-uploaded training data
    df_user_train = pd.read_sql('SELECT * FROM user_train_data', engine)

    if df_user_train.empty:
        flash('No user-uploaded data available to retrain the model.')
        return redirect(url_for('main_routes.index'))  # Updated to main_routes.index

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
    return redirect(url_for('main_routes.index'))  # Updated to main_routes.index
