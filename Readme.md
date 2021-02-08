#Car Detection

##Prerequisites:
Pycharm IDE
Installed libraries: tensorflow, numpy, matplotlib, PIL, glob
Installed TensorFlow Object Detection API(Follow steps given in: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tf-models-install)


##To run the Car Detection Model:
Unzip the given .zip folder
Open the folder in Pycharm IDE.

1. Place the Images to be detected in workspace/images/predict folder.
2. Run Detect_Cars.py
3. The program gives the number of cars detected which is used by our statistical model. You can view the detected bounding boxes for the images in detections folder

Note: If importing object_detection.utils is giving error, run the following command after downloading protoc https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tf-models-install

protoc object_detection/protos/*.proto --python_out=.

##How the model was obtained:
The object detection model was evaluated on coco dataset car images using model_main_tf2.py
then saved and exported using exporter_main_v2.py
These files are provided by TensorFlow Object Detection API.
The exported model is saved in the exported-models folder.
