# Must import torch before onnxruntime, else could not create cuda context
# ref: https://github.com/microsoft/onnxruntime/issues/11092#issuecomment-1386840174
import torch, torchvision
import onnxruntime

from time import perf_counter
from openvino.runtime import Core, Layout, get_batch, AsyncInferQueue
from pathlib import Path
import yaml
import cv2
import numpy as np
import time
from plots import Annotator, process_mask, scale_boxes, scale_image, colors
from loguru import logger


def from_numpy(x):
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x


def yaml_load(file="data.yaml"):
    # Single-line safe yaml loading
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)


def load_metadata(f=Path("path/to/meta.yaml")):
    # Load metadata from meta.yaml if it exists
    if f.exists():
        d = yaml_load(f)
        return d["stride"], d["names"]  # assign stride, names
    return None, None


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scale_fill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
    redundant=True,  # require redundant detections
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(
            x[:, :4]
        )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)

    return output


class Model:
    def __init__(
        self,
        model_path,
        imgsz=320,
        classes=None,
        device="CPU",
        plot_mask=False,
        conf_thres=0.7,
        n_jobs=1,
        is_async=False,
    ):
        # filter by class: classes=[0], or classes=[0, 2, 3]
        model_type = "onnx" if Path(model_path).suffix == ".onnx" else "openvino"
        assert Path(model_path).exists(), f"Model {model_path} not found"
        assert Path(model_path).suffix in (
            ".onnx",
            ".xml",
        ), "Model must be .onnx or .xml"
        self.model_type = model_type
        self.model_path = model_path
        self.imgsz = imgsz
        self.classes = classes
        self.plot_mask = plot_mask
        self.conf_thres = conf_thres

        # async settings
        self.n_jobs = n_jobs
        self.is_async = is_async
        self.completed_results = {}  # key: frame_id, value: inference results
        self.ori_cv_imgs = {}  # key: frame_id, value: original cv image
        self.prep_cv_imgs = {}  # key: frame_id, value: preprocessed cv image

        if self.model_type == "onnx":
            assert is_async is False, "Async mode is not supported for ONNX models"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(model_path, providers=providers)
            self.session = session
            output_names = [x.name for x in session.get_outputs()]
            self.output_names = output_names
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if "stride" in meta:
                stride, names = int(meta["stride"]), eval(meta["names"])
                self.stride = stride
                self.names = names
        elif self.model_type == "openvino":
            # load OpenVINO model
            assert Path(model_path).suffix == ".xml", "OpenVINO model must be .xml"
            ie = Core()
            weights = Path(model_path).with_suffix(".bin").as_posix()
            network = ie.read_model(model=model_path, weights=weights)
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()

            # To run inference on M1, we must export the IR model using "mo --use_legacy_frontend"
            # Otherwise, we would get the following error when compiling the model
            # https://github.com/openvinotoolkit/openvino/issues/12476#issuecomment-1222202804
            config = {}
            if n_jobs == "auto":
                config = {"PERFORMANCE_HINT": "THROUGHPUT"}
            self.executable_network = ie.compile_model(
                network, device_name=device, config=config
            )
            num_requests = self.executable_network.get_property(
                "OPTIMAL_NUMBER_OF_INFER_REQUESTS"
            )
            self.n_jobs = num_requests if n_jobs == "auto" else int(n_jobs)
            logger.info(f"Optimal number of infer requests should be: {num_requests}")
            self.stride, self.names = load_metadata(
                Path(weights).with_suffix(".yaml")
            )  # load metadata

            if is_async:
                logger.info(f"Using num of infer requests jobs: {n_jobs}")
                self.pipeline = AsyncInferQueue(self.executable_network, self.n_jobs)
                self.pipeline.set_callback(self.callback)

    def preprocess(self, cv_img, pt=False):
        im = letterbox(cv_img, self.imgsz, stride=self.stride, auto=pt)[
            0
        ]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        im = im.cpu().numpy()  # torch to numpy
        return im

    def postprocess(self, y, ori_cv_im, prep_im):
        y = [from_numpy(x) for x in y]
        pred, proto = y[0], y[-1]

        im0 = ori_cv_im

        # NMS
        iou_thres = 0.45
        agnostic_nms = False
        max_det = 1  # maximum detections per image, only 1 aorta is needed
        pred = non_max_suppression(
            pred,
            self.conf_thres,
            iou_thres,
            self.classes,
            agnostic_nms,
            max_det=max_det,
            nm=32,
        )

        # Process predictions
        line_thickness = 3
        annotator = Annotator(
            np.ascontiguousarray(im0),
            line_width=line_thickness,
            example=str(self.names),
        )
        i = 0
        det = pred[0]
        im = prep_im
        r_xyxy, r_conf, r_masks = None, None, None
        if len(pred[0]):
            masks = process_mask(
                proto[i],
                det[:, 6:],
                det[:, :4],
                (self.imgsz, self.imgsz),
                upsample=True,
            )  # HWC
            det[:, :4] = scale_boxes(
                (self.imgsz, self.imgsz), det[:, :4], im0.shape
            ).round()  # rescale boxes to im0 size

            # Mask plotting
            if self.plot_mask:
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=im[i],
                    alpha=0.1,
                )

            # Write results
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                # Add bbox to image
                c = int(cls)  # integer class
                label = f"{self.names[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))
                r_xyxy = xyxy
                r_conf = conf
            r_xyxy = [i.int().numpy().item() for i in r_xyxy]
            r_conf = r_conf.numpy().item()
            r_masks = scale_image((self.imgsz, self.imgsz), masks.numpy()[0], im0.shape)
        return annotator.result(), (r_xyxy, r_conf, r_masks)

    def predict(self, cv_img):
        # return the annotated image and the bounding box
        result_cv_img, xyxy = None, None
        im = self.preprocess(cv_img)
        if self.model_type == "onnx":
            y = self.session.run(
                self.output_names, {self.session.get_inputs()[0].name: im}
            )
        elif self.model_type == "openvino":
            # OpenVINO model inference
            # Note: Please use FP32 model on M1, otherwise you will get many runtime errors
            # Very slow on M1, but works
            # start = perf_counter()
            y = list(self.executable_network([im]).values())
            # logger.info(f"OpenVINO inference time: {perf_counter() - start:.3f}s")
        result_cv_img, others = self.postprocess(y, cv_img, im)
        return result_cv_img, others

    def callback(self, request, userdata):
        # callback function for AsyncInferQueue
        outputs = request.outputs
        frame_id = userdata
        self.completed_results[frame_id] = [i.data for i in outputs]

    def predict_async(self, cv_img, frame_id):
        assert self.is_async, "Please set is_async=True when initializing the model"
        self.ori_cv_imgs[frame_id] = cv_img
        im = self.preprocess(cv_img)
        self.prep_cv_imgs[frame_id] = im

        # Note: The start_async function call is not required to be synchronized - it waits for any available job if the queue is busy/overloaded.
        # https://docs.openvino.ai/latest/openvino_docs_OV_UG_Python_API_exclusives.html#asyncinferqueue
        #
        # idle_id = self.pipeline.get_idle_request_id()
        # self.pipeline.start_async({idle_id: im}, frame_id)
        self.pipeline.start_async({0: im}, frame_id)

    def is_free_to_infer_async(self):
        """Returns True if any free request in the pool, otherwise False"""
        assert self.is_async, "Please set is_async=True when initializing the model"
        return self.pipeline.is_ready()

    def get_result(self, frame_id):
        """Returns the inference result for the given frame_id"""
        assert self.is_async, "Please set is_async=True when initializing the model"
        if frame_id in self.completed_results:
            y = self.completed_results.pop(frame_id)
            cv_img = self.ori_cv_imgs.pop(frame_id)
            im = self.prep_cv_imgs.pop(frame_id)
            result_cv_img, others = self.postprocess(y, cv_img, im)
            return result_cv_img, others
        return None


if __name__ == "__main__":
    m_p = "weights/yolov7seg-JH-v1.onnx"
    m_p = "weights/yolov5s-seg-MK-v1.onnx"
    m_p = "weights/best_openvino_model/best.xml"
    imgsz = 320
    # imgsz = 640
    model = Model(model_path=m_p, imgsz=imgsz)

    # inference an image using the loaded model
    # source = 'Tim_3-0-00-20.05.jpg'
    path = "data/Jimmy_2-0-00-04.63.jpg"
    assert Path(path).exists(), f"Input image {path} doesn't exist"

    # output path
    save_dir = "runs/predict"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_p = f"{save_dir}/{Path(path).stem}.jpg"

    # load image and preprocess
    im0 = cv2.imread(path)  # BGR
    result_cv_img, _ = model.predict(im0)
    if result_cv_img is not None:
        cv2.imwrite(out_p, result_cv_img)
        logger.info(f"Saved result to {out_p}")
    else:
        logger.error("No result, something went wrong")
