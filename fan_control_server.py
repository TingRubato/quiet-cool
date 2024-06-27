import logging
import os
import pickle
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model

app = Flask(__name__)


IP_ADDRESS = "10.10.10.10"  # Replace with your iDrac address
USERNAME = "idaracusername"             # Replace with your username
PASSWORD = "idracpassword"           # Replace with your password

# Configuration files or environment variables should be used for these settings
DATA_FILE = 'gpu_fan_data.csv'
MODEL_FILE = 'fan_speed_model.h5'
LOG_FILE = 'training.log'
ITERATION_FILE = 'iteration_count.pkl'

model = None
training_status = "No training done yet"
iteration_count = 0

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', handlers=[
    logging.FileHandler(LOG_FILE),
    logging.StreamHandler(sys.stdout)
])

# Custom loss function
def custom_loss(y_true, y_pred):
    NOISE_THRESHOLD = 80.0
    TEMP_THRESHOLD = 85.0
    NOISE_PENALTY = 10.0
    TEMP_PENALTY = 20.0

    fan_speed_loss = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))
    noise_penalty = tf.where(y_pred[:, 1] > NOISE_THRESHOLD, NOISE_PENALTY * tf.square(y_pred[:, 1] - NOISE_THRESHOLD), 0.0)
    temp_penalty = tf.where(y_pred[:, 2] > TEMP_THRESHOLD, TEMP_PENALTY * tf.square(y_pred[:, 2] - TEMP_THRESHOLD), 0.0)

    return fan_speed_loss + tf.reduce_mean(noise_penalty + temp_penalty)

# Build model
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(3)
    ])
    model.compile(optimizer='adam', loss=custom_loss)
    return model

# Load model if available
if os.path.exists(MODEL_FILE) and os.path.getsize(MODEL_FILE) > 0:
    try:
        model = load_model(MODEL_FILE, custom_objects={'custom_loss': custom_loss})
    except Exception as e:
        logging.error(f"Error loading model: {e}")

# Load data
if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
    try:
        data = pd.read_csv(DATA_FILE)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        data = pd.DataFrame(columns=['timestamp', 'temperature', 'fan_speed', 'noise_level'])
else:
    data = pd.DataFrame(columns=['timestamp', 'temperature', 'fan_speed', 'noise_level'])

# Load iteration count
if os.path.exists(ITERATION_FILE) and os.path.getsize(ITERATION_FILE) > 0:
    try:
        with open(ITERATION_FILE, 'rb') as iteration_file:
            iteration_count = pickle.load(iteration_file)
    except Exception as e:
        logging.error(f"Error loading iteration count: {e}")

# Train model function
def train_model():
    global model, training_status, data
    training_status = "Training in progress"
    logging.info(training_status)

    X = data[['temperature']].values
    y = data[['fan_speed', 'noise_level', 'temperature']].values

    if model is None:
        model = build_model(1)

    model.fit(X, y, epochs=100, batch_size=10, verbose=1)
    model.save(MODEL_FILE)
    training_status = "Training completed successfully"
    logging.info(training_status)

def set_fan_speed(config, speed):
    # Convert the speed to a two-digit hexadecimal string
    hex_speed = f"{speed:02x}"
    
    # Create the command
    command = [
        "ipmitool",
        "-I", "lanplus",
        "-H", config["address"],
        "-U", config["username"],
        "-P", config["password"],
        "raw",
        "0x30", "0x30", "0x02", "0xff", f"0x{hex_speed}"
    ]
    
    # Execute the command
    subprocess.run(command, check=True)
    logging.info(f"Fan speed set to {speed}%")
    print(f"Fan speed set to {speed}%")


@app.route('/gpu-temperature', methods=['POST'])
def gpu_temperature():
    temp = int(request.form['temperature'])
    logging.info(f"Received GPU Temperature: {temp}°C")

    global model, iteration_count
    config = {
        "address": IP_ADDRESS,
        "username": USERNAME,
        "password": PASSWORD
    }

    if model:
        # Predict the fan speed using the model
        predicted_speed = model.predict(np.array([[temp]]))[0]
        fan_speed = int(predicted_speed[0])
        if not is_valid_fan_speed(temp, fan_speed):
            fan_speed = initial_control_logic(temp)
    else:
        fan_speed = initial_control_logic(temp)

    # Set fan speed
    set_fan_speed(config, fan_speed)

    # Collect and save data
    new_data = {'timestamp': datetime.now(), 'temperature': temp, 'fan_speed': fan_speed}
    global data
    data = pd.concat([data, pd.DataFrame([new_data])], ignore_index=True)
    data.to_csv(DATA_FILE, index=False)
    logging.info(f"Data saved: {new_data}")

    # Train model periodically
    if len(data) >= 10 and len(data) % 10 == 0:
        train_model()

    return "OK"

def initial_control_logic(temp):
    if temp < 40:
        return 10
    elif temp < 50:
        return 30
    elif temp < 60:
        return 50
    elif temp < 70:
        return 70
    elif temp < 80:
        return 80
    else:
        return 100

def is_valid_fan_speed(temp, fan_speed):
    if temp < 40:
        return 10 <= fan_speed <= 30
    elif temp < 50:
        return 30 <= fan_speed <= 50
    elif temp < 60:
        return 50 <= fan_speed <= 70
    elif temp < 70:
        return 70 <= fan_speed <= 80
    elif temp < 80:
        return 80 <= fan_speed <= 90
    else:
        return 90 <= fan_speed <= 100

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=23333)
    except Exception as e:
        logging.error(f"Error starting Flask app: {e}")
        raise