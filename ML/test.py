import numpy as np
import cv2
import tensorflow as tf

# Load the saved model
model_path = 'workout_model_2024-05-28 09_13_39.495369.h5'
model = tf.keras.models.load_model(model_path)

# Load the video
video_path = 'deadlift_12.mp4'
video = cv2.VideoCapture(video_path)

# Read the labels
labels = open('workout_label.txt', 'r').read().splitlines()

# Preprocess a frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (256, 256))
    frame = frame / 255.0
    return frame

# Process frames in smaller batches
batch_size = 10
frames = []
predicted_workouts = []

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame = preprocess_frame(frame)
    frames.append(frame)
    
    if len(frames) == batch_size:
        frames = np.array(frames)
        predictions = model.predict(frames)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_workouts.extend([labels[i] for i in predicted_classes])
        frames = []

# Process any remaining frames
if frames:
    frames = np.array(frames)
    predictions = model.predict(frames)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_workouts.extend([labels[i] for i in predicted_classes])

video.release()

# Print the predicted workouts
for workout in predicted_workouts:
    print(workout)