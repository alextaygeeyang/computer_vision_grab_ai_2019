Setup guide

This codebase is tested on Ubuntu 16.04 and Ubuntu 18.04.


For ubuntu 18.04
1. Install docker in ubuntu. [Installation steps](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
2. Download models from [google drive](https://drive.google.com/drive/folders/1LH9St_3SSz4HM1FdhPdRXjxTz-swFhcO?usp=sharing) link(ssd fpn resnet 50 models and label map) 
3. Run `pip install -r requirements.txt`
4. Place downloaded model file (the data folder) in root directory of this project
5. Your folder structure should look like this 
   ```
    |
    |----data
    |
    |----object_detection
    |
    |--client2.py
    |
    |--config.py
   ``` 
6. From project root directory , run `sudo chmod +x start_service.sh` and `sudo ./start_service.sh` to start tensorflow serving docker service.
7. Rest api from tensorflow serving is now accessible at localhost:8501
