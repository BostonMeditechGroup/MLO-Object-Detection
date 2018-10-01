<img src=sample/test1.jpg width=100% />

## Resource 
This project was custom object detection with "https://medium.freecodecamp.org/tracking-the-millenium-falcon-with-tensorflow-c8c86419225e"


## Installation

First, with python and pip installed, install the scripts requirements:

```bash
pip install -r requirements.txt
```
Then you must compile the Protobuf libraries:

```bash
protoc object_detection/protos/*.proto --python_out=.
```
Add `models` and `models/slim` to your `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

## DATA
prepare the images, xmls(labeling files), and trainval.txt(list name of dataset) 

## Usage
'''
models
|-- annotations
|   |-- label_map.pbtxt
|   |-- trainval.txt
|   `-- xmls
|       |-- 1.xml
|       |-- 2.xml
|       |-- 3.xml
|       `-- ...
|-- images
|   |-- 1.jpg
|   |-- 2.jpg
|   |-- 3.jpg
|   `-- ...
|-- object_detection
|   `-- ...
`-- ...

'''

### 1) Create the TensorFlow Records
Run the script:

```bash
python object_detection/create_tf_record.py
```

Once the script finishes running, you will end up with a `train.record` and a `val.record` file. This is what we will use to train the model.

### 2) Download a Base Model
You can find models to download from this [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 

```bash
wget [page link that you want to use the model]
```

Extract the files and move all the `model.ckpt` to our models directory.

```bash
tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz 
```

```bash
mv * ../ 
```

### 3) Train the Model
Run the following script to train the model:

```bash
python object_detection/train.py \
        --logtostderr \
        --train_dir=train \
        --pipeline_config_path=faster_rcnn_resnet101.config
```

### 4) Export the Inference Graph
The training time is dependent on the amount of training data. My model was pretty solid at ~4.5k steps. The loss reached a minimum at ~20k steps. I let it train for 200k steps, but there wasn't much improvement.


I recommend testing your model every ~5k steps to make sure you’re on the right path.

You can find checkpoints for your model in `MLO-Object-Detection/train`.

Move the model.ckpt files with the highest number to the root of the repo:
- `model.ckpt-STEP_NUMBER.data-00000-of-00001`
- `model.ckpt-STEP_NUMBER.index`
- `model.ckpt-STEP_NUMBER.meta`

In order to use the model, you first need to convert the checkpoint files (`model.ckpt-STEP_NUMBER.*`) into a frozen inference graph by running this command:

```bash
python object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path faster_rcnn_resnet101.config \
        --trained_checkpoint_prefix model.ckpt-STEP_NUMBER \
        --output_directory output_inference_graph
```

You should see a new `output_inference_graph` directory with a `frozen_inference_graph.pb` file.

### 5) Test the Model
Just run the following command:

```bash
python MLO-Object-Detection/object_detection/object_detection_runner.py
```

It will run your object detection model found at `output_inference_graph/frozen_inference_graph.pb` on all the images in the `test_images` directory and output the results in the `output/test_images` directory.

## Results
Find a 'MLO-Object-Detection/result/' folder and check the result. 

## License

[MIT](LICENSE)
