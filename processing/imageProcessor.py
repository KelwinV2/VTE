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

def image_processer(file, top_k=5):
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

    # Select the top-k labels
    top_k_indices = similarity.argsort()[-top_k:][::-1]
    top_labels = [keywords[i] for i in top_k_indices]
    top_scores = [similarity[i] for i in top_k_indices]

    # Prepare results with confidence scores
    return_dict = {}
    for label, score in zip(top_labels, top_scores):
        if label not in return_dict:
            return_dict[label] = []
            return_dict[label].append(1)
            return_dict[label].append(round(score * 100, 1))

    return return_dict

# Example usage
# Assuming `file` is an uploaded image file
# result = image_processer(file, top_k=3)
# print(result)
