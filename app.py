from flask import Flask
from routes.main_routes import main_routes
from models.models import Base, engine, Session
import config  # Import your config file
import os
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set the secret key from the environment variable
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Load the configuration from config.py
app.config.from_object(config)

# Register Blueprints (Modular Routes)
app.register_blueprint(main_routes)

# Create necessary folders if they don't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize the database
Base.metadata.create_all(engine)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

