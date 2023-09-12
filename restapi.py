"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from PIL import Image
import torch
from flask import Flask, jsonify, request

app = Flask(__name__)
DETECTION_URL = "/v1/object-detection/Trash1"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        
        # results = model(img, size=640) # reduce size=320 for faster inference
        temp = model(img, size=320) # reduce size=320 for faster inference
        # results = temp.pandas().xyxy[0]['name']
        # results=temp.pandas().xyxy[0].to_json(orient="index")
        
        results=temp.pandas().xyxy[0].to_json(orient="columns")
        index=results.rindex('{')
        res={
            'name': results[index:-1]
        }
        return jsonify(res)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--model', default='yolov5s', help='model to run, i.e. --model yolov5s')
    args = parser.parse_args()

    # model = torch.hub.load('ultralytics/yolov5', args.model)
    model= torch.hub.load("ultralytics/yolov5", 'custom', 
                          path='/Users/jiwon/Desktop/TEST3/yolov5/runs/train/Trash1/weights/best.pt', 
                          force_reload=True, skip_validation=True)

    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
