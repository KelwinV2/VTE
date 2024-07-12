# tests/test_main.py
import unittest
from fastapi.testclient import TestClient

import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)

from main import app

class TestMain(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_read_root(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

    def test_upload_file_format_bad(self):
        with open("./tests/assets/test.pdf", "rb") as pdf_file:
            files = {'file': ('test.pdf', pdf_file, 'application/pdf')}
            response = self.client.post("/upload", files=files)

            self.assertEqual(response.status_code, 415)
    
    def test_upload_file_format_image(self):
        with open("./tests/assets/hammerhead.jpg", "rb") as img_file:
            files = {'file': ('hammerhead.jpg', img_file, 'image/jpg')}
            raw_response = self.client.post("/upload", files=files)

            self.assertEqual(raw_response.status_code, 200)

            response = raw_response.json()
            response = response['result']

            self.assertIn("hammerhead", response)
            self.assertEqual("Aquatic Animals", response["hammerhead"][1])
            self.assertEqual("Pets & Animals", response["hammerhead"][2])
            
    def test_upload_file_format_video(self):
        with open("./tests/assets/car2.mp4", "rb") as video_file:
            files = {'file': ('car2.mp4', video_file, 'video/mp4')}
            raw_response = self.client.post("/upload", files=files)  

            self.assertEqual(raw_response.status_code, 200)      




if __name__ == "__main__":
    unittest.main()
