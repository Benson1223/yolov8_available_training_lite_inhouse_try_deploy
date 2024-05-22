from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="momo640.yaml", epochs=5)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://attach.setn.com/newsimages/2019/07/09/2010347-XXL.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format

if __name__ == '__main__':
    main()