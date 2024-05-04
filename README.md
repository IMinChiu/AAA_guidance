# Automation of Aorta Measurement in Ultrasound Images

## Env setup

Suggested hardware:

- GPU: NVIDIA RTX 3090 or higher x1 (model training using PyTorch)
- CPU: 11th Gen Intel(R) Core(TM) i9-11900KF @ 3.50GHz, or higher (model inference using OpenVINO)

Software stack:

- OS: Ubuntu 20.04 LTS
- Python: 3.8+
- Python Env: conda

```shell
conda create -n aorta python=3.8 -y
conda activate aorta
pip install -r requirements.txt
```

## Dataset

Steps to prepare the dataset:

1. Collect images and import to CVAT
2. Label the images in CVAT
3. Export the labelled data in `COCO 1.0` format using CVAT

   1. Go to CVAT > `Projects` page
   2. Click `⋮` on `aorta` project
   3. Click `Export dataset`
      - Format: `COCO 1.0`
      - Save images: `Yes`

4. Convert the new split data into YOLOv5 format

   ```shell
   python dataset.py coco2yolov5 [path/to/coco/input/dir] [path/to/yolov5/output/dir]
   ```

[CVAT](https://github.com/cvat-ai/cvat/tree/v2.3.0) info, set up with docker compose

- Server version: 2.3
- Core version: 7.3.0
- Canvas version: 2.16.1
- UI version: 1.45.0

Dataset related scripts:

- [coco2yolov5seg.ipynb](../coco2yolov5seg.ipynb): Convert COCO format to YOLOv5 format for segmentation task
- [coco_merge_split.ipnb](../coco_merge_split.ipynb): Merge and split COCO format dataset

## Training / Validation / Export

Model choice: Prefer [yolov5-seg] over [yolov7-seg] for training/validation/exporting models, performance comparison:

- yolov5s-seg, fast transfer learning (~5-10 mins for 100 epochs using RTX 3090) and CPU inference
- yolov7-seg, seems too heavy (slower inference using CPU)

Please refer to the repos of yolov5 seg & yolov7 seg for details of training/validation/exporting models.

[yolov5-seg]: https://github.com/ultralytics/yolov5/blob/master/segment/tutorial.ipynb
[yolov7-seg]: https://github.com/WongKinYiu/yolov7/tree/u7/seg

### yolov5-seg

Tested commit:

```shell
# Assume work dir is aorta/
git clone https://github.com/ultralytics/yolov5
cd yolov5
git checkout 23c492321290266810e08fa5ee9a23fc9d6a571f
git apply ../add_clearml_yolov5.patch
```

As of 2023, yolov5 seg doesn't support ClearML, but there is a [PR](https://github.com/ultralytics/yolov5/pull/10752) for it. So we can manually update these files to use ClearML to track the training process, or apply [add_clearml_yolov5.patch](./add_clearml_yolov5.patch).

```shell
# Example
## Original training script
python segment/train.py --img 640 --batch 16 --epochs 3 --data coco128-seg.yaml --weights yolov5s-seg.pt --cache

## Updated training script with ClearML support
python segment/train.py --project [clearml_project_name] --name [task_name] --img 640 --batch 16 --epochs 3 --data coco128-seg.yaml --weights yolov5s-seg.pt --cache
```

## Test video

- Test video: [Ellen_3.mp4](./Ellen_3.mp4)
- Tested video (mp4): Converted from the original avi using `ffmpeg`:

  ```shell
  ffmpeg -i "Ellen_3.avi" -vcodec h264 -acodec aac -b:v 500k -strict -2 Ellen_3.mp4`
  ```

## Demo (POC for 2022 Intel DevCup)

```shell
# run demo, using openvino model
python demo.py --video Ellen_3.mp4 --model weights/yolov5s-v2/best_openvino_model/yolov5-640-v2.xml --plot-mask --img-size 640

# or run the demo using onnx model
python demo.py --video Ellen_3.mp4 --model weights/yolov5s-v2/yolov5-640.onnx --plot-mask --img-size 640

# or run in the headless mode, generating a recording of the demo
./demo_headless.sh --video Ellen_3.mp4 --model [path/to/model]
```

## Deploy Pyinstaller EXE

Only tested on Windows 10:

```shell
pip install pyinstaller==5.9
pyinstaller demo.py
# (TODO) Replace the following manual steps with pyinstaller --add-data or spec file
#
# Manual copy files to dist\demo
# 1. Copy best_openvino_model folder to dist\demo\
# 2. Copy openvino files to dist\demo
# C:\Users\sa\miniforge3\envs\echo\Lib\site-packages\openvino\libs
#   plugins.xml
#   openvino_ir_frontend.dll
#   openvino_intel_cpu_plugin.dll
#   openvino_intel_gpu_plugin.dll
```

Troubleshooting: If the deployed EXE is not working with error `ValueError: --plotlyjs argument is not a valid URL or file path:`, please move the dist folder to another location with no special characters or Chinese in the path. Reference: <https://github.com/plotly/Kaleido/issues/57>
