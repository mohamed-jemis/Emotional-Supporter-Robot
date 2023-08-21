from flask import Flask, request
from keras.preprocessing import image
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

with open("load_image.pickle", "rb") as file:
    load_image = pickle.load(file)

with open("loadbase64.pickle", "rb") as file:
    loadBase64Img = pickle.load(file)

with open("buildmodel.pickle", "rb") as file:
    build_model = pickle.load(file)

with open("detectface.pickle", "rb") as file:
    detect_face = pickle.load(file)

num_classes = 7
labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

loaded_model = keras.models.load_model("my_model.h5")
loaded_model.load_weights("facial_expression_model_weights.h5")


def analyze(
        img_path,
        actions=("emotion", "age", "gender", "race"),
        enforce_detection=True,
        detector_backend="opencv",
        align=True,
        silent=False,
):
    resp_objects = []

    # build emotion model
    model = loaded_model

    img_objs = extract_facess(
        img=img_path,
        target_size=(224, 224),
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )
    for img_content, img_region, _ in img_objs:
        if img_content.shape[0] > 0 and img_content.shape[1] > 0:
            obj = {}
            img_gray = cv2.cvtColor(img_content[0], cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (48, 48))
            img_gray = np.expand_dims(img_gray, axis=0)

            emotion_predictions = model.predict(img_gray, verbose=0)[0, :]

            sum_of_predictions = emotion_predictions.sum()

            obj["emotion"] = {}

            for i, emotion_label in enumerate(labels):
                emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                obj["emotion"][emotion_label] = emotion_prediction

            obj["dominant_emotion"] = labels[np.argmax(emotion_predictions)]
            resp_objects.append(obj)

    return resp_objects


def extract_facess(
        img,
        target_size=(224, 224),
        detector_backend="opencv",
        grayscale=False,
        enforce_detection=True,
        align=True,
):
    """Extract faces from an image.

    Args:
        img: a path, url, base64 or numpy array.
        target_size (tuple, optional): the target size of the extracted faces.
        Defaults to (224, 224).
        detector_backend (str, optional): the face detector backend. Defaults to "opencv".
        grayscale (bool, optional): whether to convert the extracted faces to grayscale.
        Defaults to False.
        enforce_detection (bool, optional): whether to enforce face detection. Defaults to True.
        align (bool, optional): whether to align the extracted faces. Defaults to True.

    Raises:
        ValueError: if face could not be detected and enforce_detection is True.

    Returns:
        list: a list of extracted faces.
    """

    # this is going to store a list of img itself (numpy), it region and confidence
    extracted_faces = []

    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img = load_image(img)
    img_region = [0, 0, img.shape[1], img.shape[0]]

    if detector_backend == "skip":
        face_objs = [(img, img_region, 0)]
    else:
        face_detector = build_model(detector_backend)
        face_objs = detect_facess(face_detector, detector_backend, img, align)

    # in case of no face found
    if len(face_objs) == 0 and enforce_detection is True:
        raise ValueError(
            "Face could not be detected. Please confirm that the picture is a face photo "
            + "or consider to set enforce_detection param to False."
        )

    if len(face_objs) == 0 and enforce_detection is False:
        face_objs = [(img, img_region, 0)]

    for current_img, current_region, confidence in face_objs:
        if current_img.shape[0] > 0 and current_img.shape[1] > 0:
            if grayscale is True:
                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

            # resize and padding
            if current_img.shape[0] > 0 and current_img.shape[1] > 0:
                factor_0 = target_size[0] / current_img.shape[0]
                factor_1 = target_size[1] / current_img.shape[1]
                factor = min(factor_0, factor_1)

                dsize = (
                    int(current_img.shape[1] * factor),
                    int(current_img.shape[0] * factor),
                )
                current_img = cv2.resize(current_img, dsize)

                diff_0 = target_size[0] - current_img.shape[0]
                diff_1 = target_size[1] - current_img.shape[1]
                if grayscale is False:
                    # Put the base image in the middle of the padded image
                    current_img = np.pad(
                        current_img,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                            (0, 0),
                        ),
                        "constant",
                    )
                else:
                    current_img = np.pad(
                        current_img,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                        ),
                        "constant",
                    )

            # double check: if target image is not still the same size with target.
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)
            # normalizing the image pixels
            # what this line doing? must?
            img_pixels = image.image_utils.img_to_array(current_img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize input in [0, 1]

            # int cast is for the exception - object of type 'float32' is not JSON serializable
            region_obj = {
                "x": int(current_region[0]),
                "y": int(current_region[1]),
                "w": int(current_region[2]),
                "h": int(current_region[3]),
            }

            extracted_face = [img_pixels, region_obj, confidence]
            extracted_faces.append(extracted_face)

    if len(extracted_faces) == 0 and enforce_detection == True:
        raise ValueError(
            f"Detected face shape is {img.shape}. Consider to set enforce_detection arg to False."
        )

    return extracted_faces


def detect_facess(face_detector, detector_backend, img, align=True):
    backends = {
        "opencv": detect_face,

    }

    detect_face_fn = backends.get(detector_backend)

    if detect_face_fn:  # pylint: disable=no-else-return
        obj = detect_face_fn(face_detector, img, align)
        # obj stores list of (detected_face, region, confidence)
        return obj
    else:
        raise ValueError("invalid detector_backend passed - " + detector_backend)


test_image = "hassan.jpeg"


def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    y_pos = np.arange(len(objects))
    image_path = test_image  # Replace with the path to your image file
    image = mpimg.imread(image_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Display the image
    ax1.imshow(image)
    ax1.axis('off')  # Remove axes
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    plt.tight_layout()
    plt.show()


app = Flask(__name__)


@app.route('/api', methods=['GET'])
def pre_processing():
    d = {}
    img = request.files['images']
    img.save('temp_img.jpg')
    input_path = 'hassan.jpg'
    answer = analyze(img_path=input_path, actions=['emotion'])
    d['output'] = answer[0]["dominant_emotion"]
    return d


if __name__ == "__main__":
    app.run()


