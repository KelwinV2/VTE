import tensorflow as tf
import torch
import clip
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import numpy as np
import io
import json

# Load the pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load keyword model (if necessary)
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

def keyword_predicter(key):
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
    img = preprocess(Image.open(img_bytes)).unsqueeze(0).to(device)

    # Tokenize the possible labels
    text_inputs = clip.tokenize(keywords).to(device)

    # Compute embeddings
    with torch.no_grad():
        image_features = clip_model.encode_image(img)
        text_features = clip_model.encode_text(text_inputs)

    # Normalize the embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity between image and label embeddings
    similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()

    # Prepare results with confidence scores
    return_dict = {}
    for label, score in zip(keywords, similarity):
        if score > 0.1:  # threshold to filter relevant labels
            if label not in return_dict:
                return_dict[label] = []
                return_dict[label].append(1)

                sub_category, super_category = keyword_predicter(label)
                return_dict[label].append(sub_category)
                return_dict[label].append(super_category)

                # Appending the score
                return_dict[label].append(round(score * 100, 1))
            else:
                return_dict[label][0] += 1

    return return_dict