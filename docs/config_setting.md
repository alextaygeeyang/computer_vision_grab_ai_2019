Setting config.py



server_url : Default `http://localhost:8501/v1/models/ssd_fpn_resnet50:predict` This is the url for rest api to access predict functionality of tensorflow serving

input_image_folder : absolute path for folder that contains 
input images. All images in the selected folder will be sent for prediction.

output_folder : This folder is used to store images with server response. Server response for each input images
 includes a jpg image that contains bounding boxes on detected cars and a json with the same name that contains the detection_classes, confidence score(detection scores) and detection number.

 label_map_path : The label map is contained in data folder(stanford_cars_label_map.pbtxt). 