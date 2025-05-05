import unittest
import numpy as np
import cv2
from traffic_signs import load_data, build_model, train_model, evaluate_model

class TestTrafficSigns(unittest.TestCase):
    def test_image_preprocessing(self):
        dummy_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        resized_image = cv2.resize(dummy_image, (30, 30))
        normalized_image = resized_image / 255.0
        self.assertEqual(normalized_image.shape, (30, 30, 3))
        self.assertTrue(np.max(normalized_image) <= 1.0)
    
    def test_model_training(self):
        X_train = np.random.rand(10, 30, 30, 3)
        y_train = np.eye(43)[np.random.choice(43, 10)]
        
        model = build_model()
        history = train_model(model, X_train, y_train, epochs=1)
        self.assertTrue('loss' in history.history)
        self.assertTrue(len(history.history['loss']) > 0)
    
    def test_model_evaluation(self):
        X_test = np.random.rand(5, 30, 30, 3)
        y_test = np.eye(43)[np.random.choice(43, 5)]
        
        model = build_model()
        acc = evaluate_model(model, X_test, y_test)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

if __name__ == '__main__':
    unittest.main()
