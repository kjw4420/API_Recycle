"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from PIL import Image
import torch
from flask import Flask, jsonify, request
import datetime

app = Flask(__name__)
DETECTION_URL = "/v1/object-detection/Trash1"
   
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        
        temp = model(img, size=640) # reduce size=320 for faster inference

        # 이미지 크기 조절(화면에서 뽑기 위함(html 상에서))
        resized_image = img.resize((380, 506), Image.ANTIALIAS)


        results = model([resized_image]) #이미지 사이즈 조정
        picture=results.render()
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"/Users/jiwon/Desktop/zolup5/src/main/resources/yolov5_img/{now_time}.png"

        #객체인식 결과 사진 파일 SpringSever에 저장
        Image.fromarray(results.ims[0]).save(img_savename)
        img_result=temp.pandas().xyxy[0].to_json()
        
        res={
            'result': img_result,
            'img':  img_savename,
        }
        return jsonify(res)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--model', default='yolov5s', help='model to run, i.e. --model yolov5s')
    args = parser.parse_args()

    # model = torch.hub.load('ultralytics/yolov5', args.model)
    model= torch.hub.load("ultralytics/yolov5", 'custom', 
                        #   path='/Users/jiwon/Desktop/TEST3/yolov5/runs/train/Trash1/weights/best.pt', #원래 yolov5 
                          path='/Users/jiwon/Desktop/yolov5_api/best.pt', #새로 학습 시킨 모델(성능 UP)
                          
                          force_reload=True, skip_validation=True)


    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
