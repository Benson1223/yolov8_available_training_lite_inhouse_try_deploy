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

V9技術文章:
https://blog.csdn.net/csdn_xmj/article/details/136294694
https://blog.csdn.net/StopAndGoyyy/article/details/138631828
https://jeremy455576.medium.com/yolov4%E4%BD%BF%E7%94%A8%E6%8A%80%E8%A1%93%E8%84%88%E7%B5%A1%E5%BD%99%E6%95%B4-backbone-4408869015c2
https://ithelp.ithome.com.tw/m/articles/10337203
https://blog.csdn.net/YXD0514/article/details/132466512
https://blog.csdn.net/java1314777/article/details/136237323

0619進度:
https://blog.csdn.net/weixin_46875627/article/details/136711521
https://www.researchgate.net/figure/RepConv-and-RepConvNRepConv-has-one-more-Batch-Normalization-Layer-branch-than-RepConvN_fig4_372346827
https://aiacademy.tw/yolov7/
RepConv [6] 是一個著名的模型重參數化方法之一，它在訓練時將 3x3 卷積、1x1 卷積和 identity connection (id) 組合在一個卷積層中，而在推論時將他重組成一個 3x3 的卷積，這個方法在 VGG 架構上取得了優異的性能，如 圖四 所示。

RepConvN: Re-Parameterized Convolution with No ID
ID: 恒等快捷连接（identity shortcut connection）
