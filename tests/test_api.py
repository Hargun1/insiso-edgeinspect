import io
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers
from src import api


def upload(data, content_type='image/png'):
    return UploadFile(filename='part.png', file=io.BytesIO(data),
                      headers=Headers({'content-type': content_type}))


class FakeDetector:
    def predict(self, image, confidence):
        assert image.shape == (10, 20, 3)
        assert image[0, 0].tolist() == [0, 0, 255]
        return {'detections': [], 'width': 20, 'height': 10, 'confidence': confidence}


class TestAPI(unittest.TestCase):
    def test_missing_model(self):
        with patch.object(api, 'detector', None):
            self.assertFalse(api.health()['model_loaded'])
            with self.assertRaises(HTTPException) as error:
                api.inspect(upload(b'x'), confidence=.25)
            self.assertEqual(error.exception.status_code, 503)

    def test_valid_image_decoded_as_bgr(self):
        buffer = io.BytesIO()
        Image.new('RGB', (20, 10), (255, 0, 0)).save(buffer, format='PNG')
        with patch.object(api, 'detector', FakeDetector()):
            response = api.inspect(upload(buffer.getvalue()), confidence=.4)
        self.assertEqual(response['confidence'], .4)

    def test_corrupt_image(self):
        with patch.object(api, 'detector', FakeDetector()):
            with self.assertRaises(HTTPException) as error:
                api.inspect(upload(b'not an image'), confidence=.25)
            self.assertEqual(error.exception.status_code, 400)

    def test_unsupported_type(self):
        with patch.object(api, 'detector', FakeDetector()):
            with self.assertRaises(HTTPException) as error:
                api.inspect(upload(b'hello', 'text/plain'), confidence=.25)
            self.assertEqual(error.exception.status_code, 415)

    def test_oversized_upload(self):
        with patch.object(api, 'detector', FakeDetector()):
            with self.assertRaises(HTTPException) as error:
                api.inspect(upload(b'x'*(10*1024*1024+1)), confidence=.25)
            self.assertEqual(error.exception.status_code, 413)

