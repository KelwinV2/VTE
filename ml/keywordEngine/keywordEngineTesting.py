import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)

# Construct the absolute path to the data file
data_file_path = os.path.join(parent_dir, 'data', 'od_data.json')

# Load data from JSON file
with open(data_file_path, 'r') as file:
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

# Convert keywords to one-hot encoding
keywords_one_hot = tf.keras.utils.to_categorical(keywords_encoded)

# Construct the absolute path to the model file
model_file_path = os.path.join(parent_dir, 'models', 'keywordEngineModel.keras')

# Load the trained model
model = tf.keras.models.load_model(model_file_path)

# Predict subcategory and super category for a given keyword
def predict_category(keyword):
    try:
        keyword_index = keywords.index(keyword)
        keyword_vector = keywords_one_hot[keyword_index]
        predictions = model.predict(np.array([keyword_vector]))
        predicted_subcategory_index = np.argmax(predictions[0])
        predicted_super_category_index = np.argmax(predictions[1])
        predicted_subcategory = subcategories_label_encoder.inverse_transform([predicted_subcategory_index])[0]
        predicted_super_category = super_categories_label_encoder.inverse_transform([predicted_super_category_index])[0]
        return predicted_subcategory, predicted_super_category
    except ValueError:
        return None, None

# Example prediction
keyword = "Pepperoni"
predicted_subcategory, predicted_super_category = predict_category(keyword)
if predicted_subcategory and predicted_super_category:
    print("Predicted subcategory for keyword '{}': {}".format(keyword, predicted_subcategory))
    print("Predicted super category for keyword '{}': {}".format(keyword, predicted_super_category))
else:
    print("Keyword '{}' not found in the dataset.".format(keyword))
