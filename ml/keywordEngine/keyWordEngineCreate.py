import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

# Convert keywords to indices
keywords_label_encoder = LabelEncoder()
keywords_encoded = keywords_label_encoder.fit_transform(keywords)

# One-hot encode keywords
keywords_one_hot = tf.keras.utils.to_categorical(keywords_encoded)

# Split data into train and test sets (for overfitting, we'll use all data for training)
X_train, y_train_sub, y_train_super = keywords_one_hot, subcategories_encoded, super_categories_encoded

# Define model architecture
input_layer = tf.keras.layers.Input(shape=(keywords_one_hot.shape[1],))
hidden_layer = tf.keras.layers.Dense(128, activation='relu')(input_layer)
hidden_layer = tf.keras.layers.Dense(128, activation='relu')(hidden_layer)
hidden_layer = tf.keras.layers.Dense(128, activation='relu')(hidden_layer)

# Output layers for subcategory and super category
output_subcategory = tf.keras.layers.Dense(len(set(subcategories_encoded)), activation='softmax', name='subcategory_output')(hidden_layer)
output_super_category = tf.keras.layers.Dense(len(set(super_categories_encoded)), activation='softmax', name='super_category_output')(hidden_layer)

# Combine input and output layers into a model
model = tf.keras.Model(inputs=input_layer, outputs=[output_subcategory, output_super_category])

# Compile model
model.compile(optimizer='adam',
              loss={'subcategory_output': 'sparse_categorical_crossentropy',
                    'super_category_output': 'sparse_categorical_crossentropy'},
              metrics={'subcategory_output': 'accuracy',
                       'super_category_output': 'accuracy'})

# Train model (use all data for training to overfit)
model.fit(X_train, 
          {'subcategory_output': y_train_sub, 'super_category_output': y_train_super}, 
          epochs=250, batch_size=32, validation_split=0.0)  # No validation split to focus on overfitting

# Predict subcategory and super category for a given keyword
def predict_category(keyword):
    keyword_index = keywords_label_encoder.transform([keyword])[0]
    keyword_vector = tf.keras.utils.to_categorical([keyword_index], num_classes=len(keywords_label_encoder.classes_))
    predicted_subcategory_index = np.argmax(model.predict(keyword_vector)[0])
    predicted_super_category_index = np.argmax(model.predict(keyword_vector)[1])
    predicted_subcategory = subcategories_label_encoder.inverse_transform([predicted_subcategory_index])[0]
    predicted_super_category = super_categories_label_encoder.inverse_transform([predicted_super_category_index])[0]
    return predicted_subcategory, predicted_super_category

# Example prediction
keyword = "pitcher"
predicted_subcategory, predicted_super_category = predict_category(keyword)
print("Predicted subcategory for keyword '{}': {}".format(keyword, predicted_subcategory))
print("Predicted super category for keyword '{}': {}".format(keyword, predicted_super_category))

model.save("./models/keywordEngineModel.keras")
