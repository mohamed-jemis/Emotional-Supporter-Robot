import os
from typing import io
import werkzeug
from flask import Flask, request, jsonify, send_file

from chatbot import call_chatbot
from my_api.preprocessing import analyze

app = Flask(__name__)

# def call_chatbot(param):
#     s = '  '
#     return s

@app.route('/upload', methods=["POST"])
def upload():
    if (request.method == "POST"):
        audio_file = request.files['audio']
        emotion = request.form['emotion']
        audio_file.save('temp.wav')
        call_chatbot('temp.wav', emotion)
        return jsonify({"message": "Audio Uploaded "})


@app.route('/download', methods=['GET'])
def download():
    modified_file_path = 'tarok1.wav'
    return send_file(modified_file_path, as_attachment=True)


@app.route('/api', methods=['POST'])
def pre_processing():
    d = {}
    img = request.files['images']
    img.save('temp_img.jpg')
    input_path = 'temp_img.jpg'
    answer = analyze(img_path=input_path, actions=['emotion'])
    d['output'] = answer[0]["dominant_emotion"]
    return jsonify({'label': d['output']})


if __name__ == '__main__':
    app.run(debug=True, port=4000)
