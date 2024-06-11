from PIL import Image
from paddleocr import PaddleOCR
import datetime
import sys
import shutil
from pathlib import Path
import time
from dateutil.relativedelta import relativedelta
import paddle
import warnings
from flask import Flask, jsonify
from flask import Flask,render_template,request
import os,base64
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, check_dataset, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, strip_optimizer, xyxy2xywh, LOGGER, check_yaml
from utils.plots import Annotator, colors
from utils.paddle_utils import load_classifier, select_device, time_sync
from models.yolo import Model
import yaml
import logging
import cv2
import numpy as np
import math
import Affinedemo
import subprocess
import json
# ---------一些全局变量的设置--------------
# 要返回的车牌框类
class Coordinate:
    def __init__(self, xmin_plate, xmax_plate, ymin_plate, ymax_plate):
        self.xmin = xmin_plate
        self.xmax = xmax_plate
        self.ymin = ymin_plate
        self.ymax = ymax_plate
    def to_dict(self):
        return {
            'xmin': self.xmin,
            'xmax': self.xmax,
            'ymin': self.ymin,
            'ymax': self.ymax
            }
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Exception ignored in: <function Variable.del")
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
app = Flask(__name__)
app.config['upimgs'] = "./"
check_requirements(exclude=('tensorboard', 'thop'))
logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印
ocr = PaddleOCR()
weights="./runs/train/exp2/weights/best.pdparams" # model.pdparams path(s)
single_cls=False  # treat as single-class dataset
data = './data/truck.yaml'
cfg = './models/yolov5s.yaml'
hyp =  'data/hyps/hyp.scratch.yaml'
source='./data/images/res.jpg'  # file/dir/URL/glob, 0 for webcam
imgsz=640  # inference size (pixels)
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000# maximum detections per image
device='' # cuda device, i.e. 0 or 0,1,2,3 or cpu
view_img=False  # show results
save_txt=False  # save results to *.txt
save_conf=False  # save confidences in --save-txt labels
save_crop=False  # save cropped prediction boxes
nosave=False  # do not save images/videos
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False # class-agnostic NMS
augment=False  # augmented inference
visualize=False  # visualize features
update=False  # update all models
project= 'runs/detect'  # save results to project/name
name='exp'  # save results to project/name
exist_ok=False  # existing project/name ok, do not increment
line_thickness=3  # bounding box thickness (pixels)
hide_labels=False # hide labels
hide_conf=False  # hide confidences
dnn=False  # use OpenCV DNN for ONNX inference
if isinstance(hyp, str):
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
source = str(source)
save_img = not nosave and not source.endswith('.txt')  # save inference images
webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    ('rtsp://', 'rtmp://', 'http://', 'https://'))
# Directories
save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
# Initialize
device = select_device(device)
data = check_dataset(data)  # check
nc = 1 if single_cls else int(data['nc'])
# Load model
w = str(weights[0] if isinstance(weights, list) else weights)
classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pdparams', '.onnx', '.tflite', '.pb', '']
check_suffix(w, suffixes)  # check weights have acceptable suffix
pdparams, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
if pdparams:
    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), mode='val')  # load FP32 model
    model.set_state_dict(paddle.load(w)['state_dict'])
    model.fuse()
    model.eval()
    stride = int(model.stride.max())  # model stride
    names = paddle.load(w)['names']  # get class names
    if classify:  # second-stage classifier
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.set_state_dict(paddle.load('resnet50.pdparams')).eval()
elif onnx:
    if dnn:
        check_requirements(('opencv-python>=4.5.4',))
        net = cv2.dnn.readNetFromONNX(w)
    else:
        check_requirements(('onnx', 'onnxruntime-gpu' if paddle.device.is_compiled_with_cuda() else 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
else:  # TensorFlow models
    check_requirements(('tensorflow>=2.4.1',))
    import tensorflow as tf
    if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
        def wrap_frozen_graph(gd, inputs, outputs):
            x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
            return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                           tf.nest.map_structure(x.graph.as_graph_element, outputs))

        graph_def = tf.Graph().as_graph_def()
        graph_def.ParseFromString(open(w, 'rb').read())
        frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
    elif saved_model:
        model = tf.keras.models.load_model(w)
    elif tflite:
        interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
        interpreter.allocate_tensors()  # allocate
        input_details = interpreter.get_input_details()  # inputs
        output_details = interpreter.get_output_details()  # outputs
        int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
imgsz = check_img_size(imgsz, s=stride)  # check image size
# ---------一些全局变量的设置--------------
# 对车牌进行仿射变换
def Affine(src,degree,rro):
    degree = math.radians(degree)
    rows,cols,channel = src.shape
    M = np.float32([[1,0,0],[math.tan(degree),1,rro*100]])
    dst = cv2.warpAffine(src,M=M,dsize=(cols,rows))
    return dst
# 对车牌进行拉伸
def Scaling(src, a, b):
    rows, cols, channel = src.shape
    M = np.float32([[a, 0, 0], [0, b, 0]])
    dst = cv2.warpAffine(src, M=M, dsize=(cols, rows))
    return dst
# 判断字符是否是中文
def is_Chinese(ch):
    if '\u4e00' <= ch <= '\u9fff':
        return True
    else:
        return False
# 定义一个函数，用于读取文件并提取transcription字段
def extract_transcriptions(file_path):
    # 打开文件
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        last_line = lines[-1]
        _, json_str = last_line.split('\t', 1)
        data = json.loads(json_str)
        transcriptions = ''.join(item['transcription'] 
                            for item in data)
            # 返回拼接后的字符串
    return transcriptions
#     OCR识别,返回识别结果以及置信度
def recognition_ori(ocr_path,angle_num):
    # 指定predict_system.py脚本的路径
    script_path = "python tools/infer/predict_system.py"
    ocr_path='.'+ocr_path
    # 设置参数
    image_dir = f"--image_dir={ocr_path}"
    det_model_dir = "--det_model_dir=../inference/ch_PP-OCRv4_det_infer"
    cls_model_dir = "--cls_model_dir=../inference/ch_ppocr_mobile_v2.0_cls_infer"
    rec_model_dir = "--rec_model_dir=../inference/ch_PP-OCRv4_rec_infer"
    use_angle_cls = f"--use_angle_cls={str(angle_num).lower() == 'true'}"
    # 更改当前工作目录,替换为PaddleOCR的路径
    os.chdir('D:\\2024Spring\\Licenseplate_recognition_backend\\PaddleOCR')
    # 构建命令
    command = f"{script_path} {image_dir} {det_model_dir} {cls_model_dir} {rec_model_dir} {use_angle_cls}"

    try:
        # 使用subprocess调用脚本
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        print("命令执行成功，输出如下：")
        with open('results.txt', 'w') as f:
            f.write(result.stdout)
        print("输出结果已保存到 results.txt 文件中")
        # 从 result.stdout 中提取 recTime
        recTime = None
        for line in result.stdout.split('\n'):
            if 'The predict total time is' in line:
                recTime = float(line.split(' ')[-1].strip())
                print(line)
                print(f"recTime: {recTime}")
                break
        print(f"recTime: {recTime}")
    except subprocess.CalledProcessError as e:
        # 回到原先目录
        # 更改当前工作目录
        os.chdir('D:\\2024Spring\\Licenseplate_recognition_backend')
        print("命令执行失败：", e.stderr)
        return None

    # 假设你有一个函数来从文件中提取识别结果
    inference_results_path = 'inference_results/system_results.txt'
    transcriptions = extract_transcriptions(inference_results_path)
    print('识别结果为'+transcriptions)
    # 回到原先目录
    # 更改当前工作目录
    os.chdir('D:\\2024Spring\\Licenseplate_recognition_backend\\PaddleOCR')
    # 将 recTime 和 transcriptions 包装成字典返回
    if recTime is None:
        return None
    else:
        result_dict = {'recTime': str(recTime*1000)+'ms', 'transcriptions': transcriptions}
        return result_dict
# 直接获取车身上的识别结果
def get_result_body(result):
    s={}
    for i in result[0]:
        s[i[1][0]]=i[1][1]
    s=sorted(s.items(),key=lambda item:item[0],reverse=True)
    sum_num = 0
    rr=[]
    for i in s:
        digit = False
        str=i[0]
        str = str.replace('.', '')
        str = str.replace(',', '')
        str= str.replace('·', '')
        if  len(str)<=2 or len(str)>=5:
            continue
        for char in str:
            if char.isdigit():
                digit=True
                break
        if not digit:
            continue
        rr.append (str)
        sum_num = sum_num + 1
    return (rr)
# 获取车牌的识别结果
def get_result(result):
    s={}
    for i in result[0]:
        s[i[1][0]]=i[1][1]
    s=sorted(s.items(),key=lambda item:item[0],reverse=True)
    sum_num = 0
    rr=[]
    digit=False
    for i in s:
        rrr = ""
        str=i[0]
        str = str.replace('.', '')
        str = str.replace(',', '')
        str= str.replace('·', '')
        if is_Chinese(str[0]):
            if is_Chinese(str[1]):
                str= "沪" + str[2:]
            else:
                str= "沪" +str[1:]
        if  len(str)<=2:
            continue
        for char in str:
            if char.isdigit():
                digit=True
                break
        if not digit:
            continue
        str.replace('l','1')
        str.replace('o','0')
        str.replace('O', '0')
        for char in str:
            if is_Chinese(char) and not char == "沪":
                continue
            rrr = rrr + char
            if len(rrr)>=7:
                break
        rr.append (rrr)
        sum_num = sum_num + 1
    return (rr)
# 计算车牌角度
def calculate_plate_angle(direction, ymin_plate, xmin_plate, ymax_plate, xmax_plate, height_plate):
    if direction == "right":
        angle_num = round(180 / math.pi * math.atan((ymax_plate - ymin_plate - height_plate) / (xmax_plate - xmin_plate)))
    else:
        angle_num = round(180 - 180 / math.pi * math.atan((ymax_plate - ymin_plate - height_plate) / (xmax_plate - xmin_plate)))
    if ymax_plate - ymin_plate < 30:
        angle_num = 0
    return angle_num
# 根据位置调整车牌的高度
def process_plate(input,color, direction, xmin_plate, xmax_plate, ymin_plate, ymax_plate, ddate1):
    # 读取对应图像
    input_img=cv2.imread(input)
    # 根据位置调整高度
    if ymin_plate < 100:
        height_plate = 25
    elif xmin_plate > 1000:
        height_plate = 31
    else:
        height_plate = 33
    # 计算角度，同时加上缩放参数
    angle_num = calculate_plate_angle(direction, ymin_plate, xmin_plate, ymax_plate, xmax_plate, height_plate)
    print("角度为" + str(angle_num))
    crop = input_img[ymin_plate:ymax_plate, xmin_plate-5:xmax_plate+10]
    scaled_img = Scaling(crop, 1.02, 1)
    print("日期" + ddate1)
    cv2.imwrite("./crop_results/" + ddate1 + "-" + str(angle_num) + "-crop.jpg", scaled_img)
    # 进行保存对应的缩放后的图像文件
    cv2.imwrite("./log/output/" + ddate1 + "-" + str(angle_num) + "-scale.jpg", scaled_img)
    save_path = "./log/output/" + ddate1 + "-" + str(angle_num) + "-scale.jpg"
    # 下面是仿射后的结果图像文件
    ocr_path=''
    # filename = input.split('/')[-1]
    if direction=='left':
        ocr_path=Affinedemo.process_img(save_path,(180-angle_num),color,ddate1,height_plate)
    else:
        ocr_path=Affinedemo.process_img(save_path, ( - angle_num),color,ddate1,height_plate)
    ret = recognition_ori(ocr_path,angle_num)
    # 识别到的车牌号
    rr = ret['transcriptions']
    # 识别的车牌时间
    recTime=ret['recTime']
    # 构造成功的响应数据，rr是识别得到的车牌号
    if rr.startswith('沪'):
        pass
    else:
        rr = '沪' + rr
    coordinate = Coordinate(xmin_plate, xmax_plate, ymin_plate, ymax_plate)
    print(json.dumps(coordinate.to_dict()))
    data_success = {"data": {"truckNo": rr,"recTime": recTime,"coordinate":coordinate.to_dict()}}
    return jsonify(data_success)

@app.route('/plate', methods=['Get', 'Post'])
def detect():
    visualize = False  # visualize features
    global resu
    global zxdd
    resu={"0":"",
          "10":"",
          "20":"",
          "30":"",
          "170":"",
          "160":"",
          "150":""
          }
    zxdd={"0":0,
          "10":0,
          "20":0,
          "30":0,
          "170":0,
          "160":0,
          "150":0
          }
    data_success = {"resulType": "SUCCESS", "data": {
        "truckNo": ["", "", ""],
        "isSuccess": True},
                    "version": "1.0.0","Errmsg":"NONE"
                    }
    start = datetime.datetime.now()
    tt=str(time.time())
    tt=tt.replace('.', '')
    # Dataloader
    f = request.files['pics']
    from_data =request.form
    from_angle=from_data['from']
    color= from_data['color']
    #id是什么意思
    if "id" in request.form:
        tt=from_data['id']
        ddate1 = str(start.year) + "/" + str(start.month) + "/" + str(start.day) + "/" + str(start.hour) + "/"+str(tt)
        ddate=str(start.year) + "/" +str(start.month) + "/" + str(start.day) + "/" + str(start.hour)
    else:
        ddate1 = str(start.year) + "/" + str(start.month) + "/" + str(start.day) + "/" + str(tt) + "/" + str(start.hour) + "-" + str(start.minute) + "-" + str(start.second)
        ddate=str(start.year) + "/" +str(start.month) + "/" + str(start.day) +  "/" +str(tt)
    if not os.path.exists("./log/input/"+ddate):
        os.makedirs("./log/input/"+ddate)
    if not os.path.exists("./log/output/"+ddate):
        os.makedirs("./log/output/"+ddate)
    # 重复，创造文件夹
    if not os.path.exists("./crop_results/"+ddate):
        os.makedirs("./crop_results/"+ddate)
        # 重复，创造文件夹
    if not os.path.exists("./contrast_results/" + ddate):
        os.makedirs("./contrast_results/" + ddate)
    print("日期是"+ddate1)
    f.save(os.path.join(app.config['upimgs'], "./log/input/"+ddate1+'.jpg'))
    print("111")
    source="./log/input/"+ddate1+'.jpg'
    print(source)
    size = (1920, 1080)  # 目标尺寸
    image_resize = Image.open(source)
    # 将 RGBA 图像转换为 RGB 模式
    image_resize = image_resize.convert("RGB")
    image_resize  = image_resize.resize(size)
    image_resize.save(source)
    print(str(start.year) + "年" + str(start.month) + "月" + str(start.day) + "日"  + str(start.hour) + "时" + str(start.minute) + "分" + str(start.second)+"秒")
    print("朝向="+str(from_angle)+"  No="+str(tt))
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pdparams)
    else:
        print('现在得到的路径是'+source)
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pdparams)
    a_week = (datetime.datetime.now() + relativedelta(weeks=-1)).strftime("%Y-%m-%d %H:%M:%S")
    a_week_datetime = datetime.datetime.strptime(a_week, "%Y-%m-%d %H:%M:%S")
    year=a_week_datetime.year
    month = a_week_datetime.month
    day = a_week_datetime.day
    path = os.getcwd()
    path_output = r"{}\log\output\{}\{}\{}".format(path, year,month, day)
    path_input = r"{}\log\input\{}\{}\{}".format(path, year, month, day)
    if (os.path.exists(path_output)):
        shutil.rmtree(path_output)
    if (os.path.exists(path_input)):
        shutil.rmtree(path_input)
    # Run inference
    if pdparams and 'CUDA' in str(device):
        model(paddle.zeros([1, 3, *imgsz]).astype(model.parameters()[0].dtype))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = paddle.to_tensor(img)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        # Inference
        if pdparams:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            if dnn:
                net.setInput(img)
                pred = paddle.to_tensor(net.forward())
            else:
                pred = paddle.to_tensor(
                    session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = paddle.to_tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2
        # NMS
        pred = non_max_suppression(pred.numpy(), conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            s += '%gx%g ' % tuple(img.shape[2:])  # print string
            if len(det):
                flag_plate=False
                flag_head = False
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for res in det:
                    if int(round(res[5]))==1:
                        head_det=res
                        flag_head = True
                        xmin_head=int(res[0])
                        ymin_head=int(res[1])
                        xmax_head=int(res[2])
                        ymax_head=int(res[3])
                    else:
                        flag_plate=True
                        plate_det=res
                        xmin_plate=int(res[0])
                        ymin_plate=int(res[1])
                        xmax_plate=int(res[2])
                        ymax_plate=int(res[3])
                        xmin=xmin_plate
                        ymin=ymin_plate
                        xmax=xmax_plate
                        ymax=ymax_plate
                #  如果识别到车牌
                if flag_plate:
                    if from_angle=="1":
                        direction="right"
                    else:
                        direction="left"
                    print(path)
                #     调用角度识别函数,path是输入图像
                    return process_plate(path,color,direction,xmin_plate,xmax_plate,ymin_plate,ymax_plate,ddate1)
                # 如果没有识别到车牌
                else:
                    result = ocr.ocr(source, det=True, cls=True)  # det + text direction + recog
                    if len(result):
                        rr = get_result_body(result)
                        if (rr):
                            print("内集卡：" +str(rr))
                        else:
                            print("未识别到车号")
                        data_success["data"]["truckNo"] = rr
            # 如果没有检测到目标对象
            else:
                result = ocr.ocr(source, det=True,cls=True)  # det + text direction + recog
                if len(result):
                    rr=get_result_body(result)
                    if (rr):
                        print("内集卡：" + str(rr))
                    else:
                        print("未识别到车号")
                    data_success["data"]["truckNo"]=rr
            return jsonify(data_success)

if  __name__  == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=5080, debug=True)








