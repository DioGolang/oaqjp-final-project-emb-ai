"""
This module runs a Flask application that detects emotions in text.
"""

from flask import Flask, request, jsonify
import requests

app = Flask(__name__)


@app.route('/emotionDetector', methods=['POST'])
def detect_emotion():
    """
    Detect the dominant emotion in the provided text.

    Returns:
        json: A JSON response containing the detected emotions and dominant emotion.
    """
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = emotion_detector(text)

    if result['dominant_emotion'] is None:
        response_text = "Invalid text! Please try again."
    else:
        response_text = (f"For the given statement, the system response is 'anger': {result['anger']}, "
                         f"'disgust': {result['disgust']}, 'fear': {result['fear']}, 'joy': {result['joy']} and "
                         f"'sadness': {result['sadness']}. The dominant emotion is {result['dominant_emotion']}.")

    return jsonify({"response": response_text})


def emotion_detector(text_to_analyse):
    """
    Analyze the provided text to detect emotions.

    Args:
        text_to_analyse (str): The text to analyze for emotions.

    Returns:
        dict: A dictionary containing the scores for each emotion and the dominant emotion.
    """
    if not text_to_analyse.strip():
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    input_json = {"raw_document": {"text": text_to_analyse}}

    response = requests.post(url, json=input_json, headers=headers)

    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    emotion_data = response.json()
    dominant_emotion = emotion_data['emotionPredictions'][0]['emotion']
    scores = emotion_data['emotionPredictions'][0]['emotionScores']

    return {
        'anger': scores['anger'],
        'disgust': scores['disgust'],
        'fear': scores['fear'],
        'joy': scores['joy'],
        'sadness': scores['sadness'],
        'dominant_emotion': dominant_emotion
    }


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
