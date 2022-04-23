#from src.detectron2.ObjectDetector import Detector
#from src.yolo.detector_test import Detector as Detector_yolo
from src.tf2.detect import Predictor
from flask import Flask, render_template, request, Response, jsonify
import os
from flask_cors import CORS, cross_origin
from src.com_ineuron_utils.utils import decodeImage

app = Flask(__name__)

#detector = Detector(filename="file.jpg")

RENDER_FACTOR = 35

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

CORS(app)


# @cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        # modelPath = 'research/ssd_mobilenet_v1_coco_2017_11_17'
        # self.objectDetection_detectron = Detector(self.filename)
        # self.objectDetection_yolo = Detector(self.filename)
        # self.objectDetection_tf2 = Predictor(self.filename, )


@app.route("/")
def home():
    # return "Landing Page"
    return render_template("index.html")


@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        framework = request.json['framework']
        model = request.json['model']
        decodeImage(image, clApp.filename)
        if framework == 'tf2':
            object_detector = Predictor(clApp.filename, model)
            result = object_detector.run_inference()

        elif framework == 'detectron2':
            object_detector = Predictor(clApp.filename, model)
            result = object_detector.run_inference()
            #object_detector = Detector(clApp.filename, model)
            #result = object_detector.inference()

        elif framework == 'yolov5':
            object_detector = Predictor(clApp.filename, model)
            result = object_detector.run_inference()
            #object_detector = Detector_yolo(clApp.filename, model)
            #result = object_detector.detect_action()

        else:
            result = 'Unknown framework'
        #result = clApp.objectDetection.inference('file.jpg', 'Right')

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"
    return jsonify(result)


# port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    port = 9000
    app.run(host='127.0.0.1', port=port)
