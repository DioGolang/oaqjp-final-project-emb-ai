import unittest
from unittest.mock import patch
from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_emotion_detection(self, mock_post):
        test_cases = [
            ("I am glad this happened", 'joy'),
            ("I am really mad about this", 'anger'),
            ("I feel disgusted just hearing about this", 'disgust'),
            ("I am so sad about this", 'sadness'),
            ("I am really afraid that this will happen", 'fear')
        ]

        for text, expected_emotion in test_cases:
            # Mock the response for each test case
            mock_post.return_value.json.return_value = {
                'emotionPredictions': [{'emotion': expected_emotion}]
            }

            result = emotion_detector(text)
            self.assertEqual(result, expected_emotion)

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_empty_response(self, mock_post):
        # Mock an empty response to test error handling
        mock_post.return_value.json.return_value = {}

        with self.assertRaises(KeyError):
            emotion_detector("Test text")

if __name__ == '__main__':
    unittest.main()
