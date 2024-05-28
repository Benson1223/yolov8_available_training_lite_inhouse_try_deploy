[環境設定] 安裝Pytorch
-> conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
-->laptop RTX3060 :pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
==================================================

[1] 安裝 Ultralytics YOLO：
-> pip install -U ultralytics

[2] 建立一個新模型：
-> python -c "from ultralytics import YOLO; model = YOLO('yolov8n.yaml')"

[3] 加載一個預訓練模型（推薦用於訓練）：
-> python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"

[4] 從 YAML 建立一個新模型並傳遞預訓練的權重：
-> python -c "from ultralytics import YOLO; model = YOLO('yolov8n.yaml').load('yolov8n.pt')"

[5] 訓練模型：
-> python -c "from ultralytics import YOLO; model = YOLO('yolov8n.yaml'); results = model.train(data='coco128.yaml', epochs=100, imgsz=640)"


佈署時，要新增runs/detect 資料夾

V9   yolov9c.pt -->https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset
   # Load a model
    model = YOLO("yolov9c.yaml")  # build a new model from scratch
    model = YOLO("yolov9c.pt")  # load a pretrained model (recommended
