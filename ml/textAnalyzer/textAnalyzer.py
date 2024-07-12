import easyocr
import sys
import os

# Initialize the reader
reader = easyocr.Reader(['en'])  # Supports multiple languages

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)

# Construct the absolute path to the data file
img_path = os.path.join(parent_dir, 'tests', 'assets', 'random_img.jpg')

# Load image and perform OCR
result = reader.readtext(img_path)

# Print the results
for (bbox, text, prob) in result:
    print(f"Detected text: {text} (confidence: {round(prob * 100, 1)})")
