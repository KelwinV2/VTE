import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.preprocessing import LabelEncoder
import numpy as np
import io
import json

# Load the models
model = tf.keras.models.load_model("./processing/models/testModel0-5.keras")
keyword_model = tf.keras.models.load_model("./processing/models/keywordEngineModel.keras")

# Load data from JSON file
with open('./processing/data/od_data.json', 'r') as file:
    data = json.load(file)

# Prepare data
keywords = []
subcategories = []
super_categories = []

for top_category, subcategories_dict in data.items():
    for subcategory, elements in subcategories_dict.items():
        for element in elements:
            keywords.append(element['keyword'])
            subcategories.append(subcategory)
            super_categories.append(top_category)

# Encode subcategories
subcategories_label_encoder = LabelEncoder()
subcategories_encoded = subcategories_label_encoder.fit_transform(subcategories)

# Encode super categories
super_categories_label_encoder = LabelEncoder()
super_categories_encoded = super_categories_label_encoder.fit_transform(super_categories)

# Encode keywords
keywords_label_encoder = LabelEncoder()
keywords_encoded = keywords_label_encoder.fit_transform(keywords)

# Verify the correct number of unique keywords
num_keywords = len(np.unique(keywords_encoded))

# Convert keywords to one-hot encoding
keywords_one_hot = tf.keras.utils.to_categorical(keywords_encoded, num_classes=num_keywords)

# Ensure the model's input shape matches the one-hot encoding
assert keyword_model.input_shape[1] == num_keywords, f"Model expected input shape {keyword_model.input_shape[1]}, but got {num_keywords}"

def keyword_predicter(key, keyword_model):
    keyword_index = keywords.index(key)
    keyword_vector = keywords_one_hot[keyword_index]
    keyword_vector = np.expand_dims(keyword_vector, axis=0)  # Expand dimensions to match model input shape
    predicted_subcategory_index = np.argmax(keyword_model.predict(keyword_vector)[0])
    predicted_super_category_index = np.argmax(keyword_model.predict(keyword_vector)[1])
    predicted_subcategory = subcategories_label_encoder.inverse_transform([predicted_subcategory_index])[0]
    predicted_super_category = super_categories_label_encoder.inverse_transform([predicted_super_category_index])[0]
    return predicted_subcategory, predicted_super_category                                     

def image_processer(file):
    # Read the file-like object into memory using io.BytesIO
    img_bytes = io.BytesIO(file.file.read())
    
    # Load the image directly from bytes into memory and preprocess it
    img = image.load_img(img_bytes, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Decode predictions
    return_dict = {}
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    for imagenet_id, label, score in decoded_predictions:
        if score > 0.1:
            if label not in return_dict:
                return_dict[label] = []
                return_dict[label].append(1)

                sub_category, super_category = keyword_predicter(label, keyword_model)
                return_dict[label].append(sub_category)
                return_dict[label].append(super_category)

                # Appending the score
                return_dict[label].append(round(score * 100, 1))
            else:
                return_dict[label][0] += 1

    return return_dict
