import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load data from JSON file
with open('data/od_data.json', 'r') as file:
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

# Load the trained model
model = tf.keras.models.load_model('./models/keywordEngineModel.keras')

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
keyword = "African_elephant"
predicted_subcategory, predicted_super_category = predict_category(keyword)
if predicted_subcategory and predicted_super_category:
    print("Predicted subcategory for keyword '{}': {}".format(keyword, predicted_subcategory))
    print("Predicted super category for keyword '{}': {}".format(keyword, predicted_super_category))
else:
    print("Keyword '{}' not found in the dataset.".format(keyword))
