import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
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
            subcategories.append([subcategory])  # Convert to list for MultiLabelBinarizer
            super_categories.append([top_category])  # Convert to list for MultiLabelBinarizer

# Encode subcategories
subcategories_label_encoder = MultiLabelBinarizer()
subcategories_encoded = subcategories_label_encoder.fit_transform(subcategories)

# Encode super categories
super_categories_label_encoder = MultiLabelBinarizer()
super_categories_encoded = super_categories_label_encoder.fit_transform(super_categories)

# Convert keywords to one-hot encoding
keywords_one_hot = tf.keras.utils.to_categorical(np.arange(len(keywords)))

# Split data into train and test sets
X_train, X_test, y_train_sub, y_test_sub, y_train_super, y_test_super = train_test_split(keywords_one_hot, 
                                                                                            subcategories_encoded, 
                                                                                            super_categories_encoded, 
                                                                                            test_size=0.2, 
                                                                                            random_state=42)

# Define model architecture
input_layer = tf.keras.layers.Input(shape=(keywords_one_hot.shape[1],))
hidden_layer = tf.keras.layers.Dense(64, activation='relu')(input_layer)
hidden_layer = tf.keras.layers.Dense(64, activation='relu')(hidden_layer)

# Output layers for subcategory and super category with 'sigmoid' activation
output_subcategory = tf.keras.layers.Dense(subcategories_encoded.shape[1], activation='sigmoid', name='subcategory_output')(hidden_layer)
output_super_category = tf.keras.layers.Dense(super_categories_encoded.shape[1], activation='sigmoid', name='super_category_output')(hidden_layer)

# Combine input and output layers into a model
model = tf.keras.Model(inputs=input_layer, outputs=[output_subcategory, output_super_category])

# Compile model with appropriate loss functions
model.compile(optimizer='adam',
              loss={'subcategory_output': 'binary_crossentropy',
                    'super_category_output': 'binary_crossentropy'},
              metrics={'subcategory_output': 'accuracy',
                       'super_category_output': 'accuracy'})

# Train model
model.fit(X_train, {'subcategory_output': y_train_sub, 'super_category_output': y_train_super}, 
          epochs=100, batch_size=32, validation_data=(X_test, {'subcategory_output': y_test_sub, 'super_category_output': y_test_super}))

# Evaluate model
evaluation_results = model.evaluate(X_test, {'subcategory_output': y_test_sub, 'super_category_output': y_test_super})
print("Evaluation results:", evaluation_results)

# Extracting evaluation results
loss_sub = evaluation_results[0]
accuracy_sub = evaluation_results[1]
accuracy_super = evaluation_results[2]

print("Subcategory Test Accuracy:", accuracy_sub)
print("Super category Test Accuracy:", accuracy_super)

# Predict subcategory and super category for a given keyword
def predict_category(keyword):
    keyword_index = keywords.index(keyword)
    keyword_vector = keywords_one_hot[keyword_index]
    predicted_subcategory_probs = model.predict(np.array([keyword_vector]))[0][0]
    predicted_super_category_probs = model.predict(np.array([keyword_vector]))[1][0]
    predicted_subcategory_indices = np.where(predicted_subcategory_probs > 0.1)[0]
    predicted_super_category_indices = np.where(predicted_super_category_probs > 0.1)[0]
    predicted_subcategories = subcategories_label_encoder.inverse_transform(predicted_subcategory_indices)
    predicted_super_categories = super_categories_label_encoder.inverse_transform(predicted_super_category_indices)
    return predicted_subcategories, predicted_super_categories

# Example prediction
keyword = "pitcher"
predicted_subcategories, predicted_super_categories = predict_category(keyword)
print("Predicted subcategories for keyword '{}': {}".format(keyword, predicted_subcategories))
print("Predicted super categories for keyword '{}': {}".format(keyword, predicted_super_categories))
