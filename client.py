import argparse
import json

import numpy as np
import requests
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import plot_util
from object_detection.utils import label_map_util
import object_detection.utils.ops as utils_ops
from PIL import Image
import config
import os 

def preprocessing(img_path):
    img = Image.open(img_path).convert("RGB")
    img_np = plot_util.load_image_into_numpy_array(img)
    img_tensor = np.expand_dims(img_np, 0)
    json_input = json.dumps({
        "signature_name": "serving_default",
        "instances": img_tensor.tolist()
    })
    return json_input


def load_image_into_numpy_array(image):
  (img_width, img_height) = image.size
  return np.array(image.getdata()).reshape(
      (img_height, img_width, 3)).astype(np.uint8)




def process_response(server_response,image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = load_image_into_numpy_array(img)
    img_size = img_np.shape
    response = json.loads(server_response.text)
    output_dict = response['predictions'][0]
    output_dict['num_detections'] = int(output_dict['num_detections'])
    output_dict['detection_classes'] = np.array(
        [int(class_id) for class_id in output_dict['detection_classes']])
    output_dict['detection_boxes'] = np.array(output_dict['detection_boxes'])
    output_dict['detection_scores'] = np.array(output_dict['detection_scores'])

    return output_dict,img_np

if __name__ == '__main__':
    
    allImagesPath = os.listdir(config.input_image_folder)
    allAbsPath = []
    for image in allImagesPath :
        if os.path.isfile(config.input_image_folder+"/"+image):
            allAbsPath.append(config.input_image_folder+"/"+image)


    for path in allAbsPath:
        json_input = preprocessing(path)
        headers = {"content-type": "application/json"}
        server_response = requests.post(
            config.server_url, data=json_input, headers=headers)
        (output_map,img_np) = process_response(server_response,path)
        filename = path.split("/")[-1].split(".")[0]
        outputJsonPath = config.output_folder+"/"+filename+".json"
        with open(outputJsonPath, 'w+') as outfile:
            json.dump(json.loads(server_response.text), outfile)
        category_index = label_map_util.create_category_index_from_labelmap(config.label_map_path, use_display_name=True)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            img_np,
            output_map['detection_boxes'],
            output_map['detection_classes'],
            output_map['detection_scores'],
            category_index,
            instance_masks=output_map.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4,
            )
        output_image = ''.join([config.output_folder+"/", filename, '.jpg'])
        Image.fromarray(img_np).save(output_image)
        print('\n\nImage saved\n\n')
