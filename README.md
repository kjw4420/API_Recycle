# 나는야 분리수거를 잘하는 어린이 재활용품 객체인식 Yolov5 api using flask🔥
본 Recycle_api는 https://github.com/robmarkcole/yolov5-flask 를 참조하여 **나는야 분리수거를 잘하는 어린이**에 필요한 형태로 수정하였다.

POST 요청으로 재활용품 사진을 넣으면 객체인식 결과를 도출한다.
## restapi.py: 데이터 출력 형식 수정
**predict()함수**에 아래 코드 부분 수정
```
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
            'result': img_result,   #결과값 Json 변환용
            'img':  img_savename,   #이미지 파일
        }
        return jsonify(res)
```

## 요청 test
<img width="682" alt="스크린샷 2023-11-13 오후 11 45 47" src="https://github.com/kjw4420/Recycle/assets/97749184/78eb1013-a4b1-463b-bf5e-7c2c8e27006f">


## reference
- [https://github.com/jzhang533/yolov5-flask](https://github.com/robmarkcole/yolov5-flask) (this repo was forked from here)
