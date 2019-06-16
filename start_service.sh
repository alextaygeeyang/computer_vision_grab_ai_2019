sudo docker run -t --rm -p 8501:8501  -v "$(pwd)/data/ssd_fpn_resnet50:/models/ssd_fpn_resnet50"  -e MODEL_NAME=ssd_fpn_resnet50  tensorflow/serving &

