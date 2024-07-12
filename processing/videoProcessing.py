import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from processing.imageProcessing import keyword_predicter
import numpy as np
import os
import tempfile
import time
import cv2
import io

# Define a global variable for the temporary directory
TEMP_DIR = tempfile.mkdtemp()

# Load the model globally once
model = tf.keras.models.load_model("./processing/models/testModel0-5.keras")
keyword_model = tf.keras.models.load_model("./processing/models/keywordEngineModel.keras")

def video_processer(file):
    # Save the uploaded file to the temporary directory
    temp_file_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file.file.read())
        
    # Calculating process time and starting the video engine process
    start_time = time.time()
    return_dict = frame_by_frame(temp_file_path)
    end_time = time.time()
    
    print("Time to process: ", end_time - start_time)

    # Cleaning up by deleting the temporary file
    os.remove(temp_file_path)

    return return_dict

def frame_by_frame(file_path):
    # Dictionary that will hold all the key values for keyword occurrences
    return_dict = {}
    
    # Using cv2 to capture the video
    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    batch_size = 32
    frames = []

    # Iterating through the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 10th frame
        if frame_count % 10 == 0:
            frame = process_image(frame)
            frames.append(frame)
            # When batch size is reached, make predictions
            if len(frames) == batch_size:
                predictions = model.predict(np.vstack(frames))
                decode_and_update_dict(predictions, return_dict)
                frames = []

        frame_count += 1

    # Process remaining frames
    if frames:    
        predictions = model.predict(np.vstack(frames))
        decode_and_update_dict(predictions, return_dict)

    cap.release()
    
    return return_dict

def process_image(frame):
    # Resize frame to match the input size of the model
    img = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def decode_and_update_dict(predictions, return_dict):
    for prediction in predictions:
        decoded_predictions = decode_predictions(np.expand_dims(prediction, axis=0), top=5)[0]
        for imagenet_id, label, score in decoded_predictions:
            # If confidence is over 10% then put it in the dictionary
            if score > 0.1:
                if label not in return_dict:
                    return_dict[label] = []
                    return_dict[label].append(1)
                    # catching an exception when processing fails due to out of range keyword
                    try:
                        sub_category, super_category = keyword_predicter(label, keyword_model)
                        return_dict[label].extend([sub_category, super_category, score * 100])
                    except Exception as e:
                        print(f"Error in keyword_predicter for label {label}: {e}")
                        # Optionally, continue with some default values or skip this label
                        continue
                else:
                    return_dict[label][0] += 1

                    # Adding the score
                    return_dict[label][3] += (score * 100)
    # what im thinking of doing is first creating a sepearte analyser file that will be like a tool using tesseract and than we will intake a image 
    for label in return_dict:
        # Averaging the score
        avg = return_dict[label][3] / return_dict[label][0]
        return_dict[label][3] = round(avg, 1)
