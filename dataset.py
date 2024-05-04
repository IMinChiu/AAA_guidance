import typer
import fiftyone as fo
from fiftyone import ViewField as F
from pathlib import Path
from pycocotools.coco import COCO
from loguru import logger
import cv2
import shutil
import os
import random
from collections import defaultdict
import csv


DEFAULT_EXCLUDE_NAME = "Ellen"
DEFAULT_INS_TRAIN = "instances_Train.json"
DEFAULT_INS_TEST = "instances_Test.json"

app = typer.Typer()


@app.command()
def newsplit(
    in_dir: str,
    train_json=DEFAULT_INS_TRAIN,
    test_json=DEFAULT_INS_TEST,
    exclude_name=DEFAULT_EXCLUDE_NAME,
):
    """
    Merge the train and test datasets,
    and then split them into new train/test by leaving one person out.
    """

    # load the dataset
    logger.info("Loading datasets...")
    ds1 = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=Path(in_dir) / "images",
        labels_path=Path(in_dir) / "annotations" / train_json,
    )
    ds2 = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=Path(in_dir) / "images",
        labels_path=Path(in_dir) / "annotations" / test_json,
    )

    logger.info(f"[Before] Num samples in train: {len(ds1)}")
    logger.info(f"[Before] Num samples in test: {len(ds2)}")

    # merge the datasets
    ds1.merge_samples(ds2)

    # generate the new split
    logger.info(f"Excluding name in filepath as train set: {exclude_name}")
    new_train_view = ds1.match(~F("filepath").re_match(exclude_name))
    new_test_view = ds1.match(F("filepath").re_match(exclude_name))
    assert len(new_train_view) + len(new_test_view) == len(ds1)
    logger.info(f"[After] Num samples in train: {len(new_train_view)}")
    logger.info(f"[After] Num samples in test: {len(new_test_view)}")
    train_counts = new_train_view.count_values("detections.detections.label")
    test_counts = new_test_view.count_values("detections.detections.label")
    logger.info(f"[After] Train counts: {train_counts}")
    logger.info(f"[After] Test counts: {test_counts}")

    # export the new split
    logger.info("Exporting new train/test...")
    new_train_p = Path(in_dir) / "annotations" / f"new_train_no-{exclude_name}.json"
    new_test_p = Path(in_dir) / "annotations" / f"new_test_{exclude_name}.json"
    new_train_view.export(
        dataset_type=fo.types.COCODetectionDataset,
        labels_path=new_train_p,
        label_field="segmentations",
        classes=ds1.default_classes,
        abs_paths=True,
    )
    new_test_view.export(
        dataset_type=fo.types.COCODetectionDataset,
        labels_path=new_test_p,
        label_field="segmentations",
        classes=ds2.default_classes,
        abs_paths=True,
    )
    logger.info(f"Exported new train: {new_train_p}")
    logger.info(f"Exported new test: {new_test_p}")


def _normalize(img_size, xy_s):
    assert len(xy_s) % 2 == 0
    normalized_xy_s = []
    dw = 1.0 / (img_size[0])
    dh = 1.0 / (img_size[1])
    for i in range(len(xy_s)):
        p = xy_s[i]
        p = p * dw if i % 2 == 0 else p * dh
        assert p <= 1.0 and p >= 0.0, f"{p} should < 1 and > 0"
        normalized_xy_s.append(p)
    return normalized_xy_s


def _coco2yolo(coco_img_dir, coco_json_path, out_dir, bbox_only=False, rois=None):
    logger.info(f"Reading {Path(coco_json_path).name}...")
    coco = COCO(coco_json_path)

    cats = coco.loadCats(coco.getCatIds())
    cats = sorted(cats, key=lambda x: x["id"], reverse=False)
    assert cats[0]["id"] == 1, f"Assume cat id starts from 1, but got {cats[0]['id']}"
    logger.info(f"{len(cats)} categories: {[cat['name'] for cat in cats]}")

    img_ids = coco.getImgIds()
    prefix = Path(coco_json_path).stem.split("_")[-1].lower()  # either train or test

    # create output directories
    target_txt_r = Path(out_dir) / prefix / "labels"
    target_img_r = Path(out_dir) / prefix / "images"
    target_txt_r.mkdir(parents=True, exist_ok=False)
    target_img_r.mkdir(parents=True, exist_ok=False)

    logger.info(f"Num of imgs: {len(img_ids)}")

    n_imgs_no_annos = 0
    num_zero_area = 0
    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        img_p = Path(coco_img_dir) / img["file_name"]
        assert img_p.exists(), f"{img_p} does not exist"

        anno_ids = coco.getAnnIds(imgIds=img["id"])
        annos = coco.loadAnns(anno_ids)

        new_filename = f"{img['id']}_{img_p.stem}"

        out_img_p = target_img_r / (new_filename + img_p.suffix)

        # get roi for the image if any
        im_cv = cv2.imread(img_p.as_posix())
        im_width, im_height = im_cv.shape[1], im_cv.shape[0]
        roi = rois[(im_width, im_height)] if rois is not None else None
        has_roi = (rois is not None) and (roi is not None) and len(roi) == 4
        if not has_roi:
            # copy image to target dir
            shutil.copy(img_p, out_img_p)
        else:
            # crop the image to target dir
            assert len(roi) == 4, f"ROI should have 4 values, but got {roi}"
            cropped_img = im_cv[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
            cv2.imwrite(out_img_p.as_posix(), cropped_img)

        # bg imgs: only need to copy img, no need to create label file
        if len(annos) == 0:
            n_imgs_no_annos += 1
            continue

        # create the label txt file
        txt_p = Path(target_txt_r) / (new_filename + ".txt")
        if txt_p.exists():
            logger.warning(f"{txt_p} already exists, {img_p} skipped")
        txt_f = open(txt_p, "w")
        img = cv2.imread(img_p.as_posix())
        h, w, _ = img.shape

        # generate txt file for each image
        for ann in annos:
            cls_id = ann["category_id"] - 1  # yolov5 uses zero-based class idx

            # region bbox, for object detection
            if bbox_only:
                bbox = ann["bbox"]
                # convert coco to yolo: top-x, top-y, w, h -> center-x, center-y, w, h
                bbox_yolo = [
                    bbox[0] + bbox[2] / 2,
                    bbox[1] + bbox[3] / 2,
                    bbox[2],
                    bbox[3],
                ]
                n_bbox_p = " ".join([str(a) for a in _normalize((w, h), bbox_yolo)])
                txt_f.write(f"{cls_id} {n_bbox_p}{os.linesep}")
                continue
            # endregion

            # region seg, for instance segmentation
            seg = ann["segmentation"]
            if len(seg) > 1:
                # TODO: Investigate why sometimes there are multiple segs
                logger.warning(f"Skip {img_p} with {len(seg)} segs of {ann}")
                continue

            if len(seg) == 1:
                xy_s = seg[0]
                # handle roi if any
                if has_roi:
                    xy_s = [xy - roi[i % 2] for i, xy in enumerate(xy_s)]
                    w, h = roi[2], roi[3]
                    # remove the points outside of roi
                    new_xy_s = []
                    for i in range(0, len(xy_s), 2):
                        x, y = xy_s[i], xy_s[i + 1]
                        if x >= 0 and x <= w and y >= 0 and y <= h:
                            new_xy_s.extend([x, y])
                    xy_s = new_xy_s
                n_xy_s = _normalize((w, h), xy_s)
                seg_p = " ".join([str(a) for a in n_xy_s])
                txt_f.write(f"{cls_id} {seg_p}{os.linesep}")
            # endregion

            # region keypoint, for pose estimation
            if "keypoints" in ann:
                # skip area 0 keypoints which could cause yolov8 training error
                if int(ann["area"]) == 0:
                    num_zero_area += 1
                    continue
                kps = ann["keypoints"]
                bbox = ann["bbox"]
                # convert coco to yolo: top-x, top-y, w, h -> center-x, center-y, w, h
                bbox_yolo = [
                    bbox[0] + bbox[2] / 2,
                    bbox[1] + bbox[3] / 2,
                    bbox[2],
                    bbox[3],
                ]
                n_bbox_p = " ".join([str(a) for a in _normalize((w, h), bbox_yolo)])
                # normalize x,y of each keypoint and keep visibility as is
                n_kp = []
                for i in range(0, len(kps), 3):
                    n_kp.append(kps[i] / w)
                    n_kp.append(kps[i + 1] / h)
                    n_kp.append(kps[i + 2])
                n_kp_p = " ".join([str(a) for a in n_kp])
                txt_f.write(f"{cls_id} {n_bbox_p} {n_kp_p}{os.linesep}")
            # endregion
        txt_f.close()
        # remove empty label file which has no annos
        if txt_p.stat().st_size == 0:
            txt_p.unlink()
            n_imgs_no_annos += 1
    empty_ratio = 100 * float(n_imgs_no_annos) / len(img_ids)
    n_imgs_anns = len(img_ids) - n_imgs_no_annos
    logger.info(f"# imgs w anns: {n_imgs_anns} {(100-empty_ratio):.2f}%")
    logger.info(f"# imgs w/o anns: {n_imgs_no_annos} {empty_ratio:.2f}%")
    logger.info(f"# zero area kps: {num_zero_area}")
    txts = [f for f in target_txt_r.iterdir() if f.is_file()]
    imgs = [f for f in target_img_r.iterdir() if f.is_file()]
    assert (len(txts) + n_imgs_no_annos) == len(imgs) == len(img_ids)
    return target_img_r


@app.command(help="Convert COCO dataset to YOLOv5 format")
def coco2yolov5(
    in_dir: str,
    out_dir: str,
    split_val_ratio: float = 0.2,
    seed: int = 42,
    bbox_only: bool = False,
    crop_roi_file: str = None,
):
    """
    Convert COCO dataset to YOLOv5 format.
    Support 3 task types: object detection, instance segmentation, pose estimation.

    YOLOv5 seg labels are the same as detection labels, using txt files with one object per line.
    The difference is that instead of "class, xywh" they are "class xy1, xy2, xy3,...".
    Ref: https://github.com/ultralytics/yolov5/issues/10161#issuecomment-1315672357

    YOLOv5 keypoint labels is using txt files with one object per line.
        class cx cy w h x1 y1 v1 ... xn yn vn
    All coordinates are normalized by image width and height.
    vn (visibility): 0, 1, or 2 => not labeled, labeled but invisible, labeled and visible
    Ref: https://github.com/ultralytics/ultralytics/issues/1870#issuecomment-1498909244
    Example: https://ultralytics.com/assets/coco8-pose.zip
    """
    if Path(out_dir).exists():
        delete = typer.confirm(f"{out_dir} alread exists. Are you sure to delete it?")
        if not delete:
            logger.info("Not deleting")
            raise typer.Abort()
        shutil.rmtree(out_dir)
        logger.info(f"Deleted {Path(out_dir).name}")

    ann_dir_p = Path(in_dir) / "annotations"
    img_dir_p = Path(in_dir) / "images"
    assert ann_dir_p.exists(), f"{ann_dir_p} does not exist"
    assert img_dir_p.exists(), f"{img_dir_p} does not exist"

    # try to find the json files of train & test in annotations dir
    train_json_p = None
    test_json_p = None
    for f in ann_dir_p.iterdir():
        if f.stem.lower().endswith("train"):
            train_json_p = f
            logger.info(f"Found train json: {f.name}")
        elif f.stem.lower().endswith("test"):
            test_json_p = f
            logger.info(f"Found test json: {f.name}")
    # must have train, while test is optional
    assert train_json_p is not None, f"Cannot find train json in {ann_dir_p}"
    do_split = False
    if test_json_p is None:
        logger.warning("Cannot find test json in [in_dir]/annotations")
        do_split = typer.confirm("Do you want to split val from train?")

    # region handle ROIs
    rois = None
    if crop_roi_file is not None:
        roi_csv_p = Path(crop_roi_file)
        assert roi_csv_p.exists(), f"{roi_csv_p} does not exist"
        # read ROIs from csv, each image size should have one ROI
        rois = defaultdict(lambda: [], {})
        with open(roi_csv_p, "r") as f:
            for roi in csv.DictReader(f):
                ori_width = int(roi["ori_width"])
                ori_height = int(roi["ori_height"])
                roi_x = int(roi["roi_x"])
                roi_y = int(roi["roi_y"])
                roi_width = int(roi["roi_width"])
                roi_height = int(roi["roi_height"])

                key = (ori_width, ori_height)
                assert key not in rois, f"Duplicate ROI for {key}"
                rois[key] = [roi_x, roi_y, roi_width, roi_height]
    # endregion

    yolo_train_img_dir = None
    yolo_test_img_dir = None
    yolo_train_img_dir = _coco2yolo(img_dir_p, train_json_p, out_dir, bbox_only, rois)
    if test_json_p is not None:
        yolo_test_img_dir = _coco2yolo(img_dir_p, test_json_p, out_dir, bbox_only, rois)

    if do_split:
        yolo_test_img_dir = Path(out_dir) / "val" / "images"
        # randomly select 20% of train images
        train_imgs = [f for f in yolo_train_img_dir.iterdir() if f.is_file()]
        n_test = int(len(train_imgs) * split_val_ratio)
        logger.info(f"Split ratio {split_val_ratio}: {n_test} test images from train")
        # set random seed to make sure the same images are selected
        random.seed(seed)
        test_imgs = random.sample(train_imgs, n_test)
        # move test images to val/images
        yolo_test_img_dir.mkdir(parents=True, exist_ok=True)
        for f in test_imgs:
            shutil.move(str(f), str(yolo_test_img_dir))
        # move labels of test images to val/labels
        yolo_test_label_dir = Path(out_dir) / "val" / "labels"
        yolo_test_label_dir.mkdir(parents=True, exist_ok=True)
        for f in test_imgs:
            label_f = yolo_train_img_dir.parent / "labels" / f"{f.stem}.txt"
            if label_f.exists():
                shutil.move(str(label_f), str(yolo_test_label_dir))

    # region create yaml file

    logger.info(f"Reading {Path(train_json_p).name}...")
    train_coco = COCO(train_json_p)
    train_cats = train_coco.loadCats(train_coco.getCatIds())
    num_kps = [
        len(c["keypoints"])
        for c in train_cats
        if "keypoints" in c and len(c["keypoints"]) > 0
    ]
    # check if all categories have the same number of keypoints
    if len(num_kps) > 0:
        assert len(set(num_kps)) == 1, "Categories have different number of keypoints"
    logger.info(f"Number of keypoints: {set(num_kps)}")
    train_cats = [c["name"] for c in train_cats]
    # ensure having the same categories in the json of train & test
    # test_coco = COCO(test_json_p)
    # test_cats = test_coco.loadCats(test_coco.getCatIds())
    # test_cats = sorted(test_cats, key=lambda x: x["id"], reverse=False)
    # test_cats = [c["name"] for c in test_cats]
    # assert ",".join(train_cats) == ",".join(test_cats), "Categories mismatch"

    out_config_file = Path(out_dir) / "data.yaml"
    with open(out_config_file, "w") as f:
        if len(num_kps) > 0:
            f.write(f"kpt_shape: [{num_kps[0]},3]" + os.linesep)
            assert num_kps[0] == 1, "Only support 1 keypoint for now"
            f.write("flip_idx: [0]" + os.linesep)
        f.write("names:" + os.linesep)
        for c in train_cats:
            f.write(f"- {c}" + os.linesep)
        f.write(f"nc: {len(train_cats)}" + os.linesep)
        f.write(f"path: {Path(out_dir).absolute()}" + os.linesep)
        train_rel_path = f"{yolo_train_img_dir.parent.name}/{yolo_train_img_dir.name}"
        f.write(f"train: {train_rel_path}" + os.linesep)
        if yolo_test_img_dir is not None:
            val_rel_path = f"{yolo_test_img_dir.parent.name}/{yolo_test_img_dir.name}"
            f.write(f"val: {val_rel_path}" + os.linesep)

    logger.info(f"Config file saved: {out_config_file}")
    # endregion

    logger.info("Done ✅")


@app.command(help="List all image sizes and counts in a directory recursively")
def list_img_sizes(
    in_dir: str = typer.Argument(..., help="Input directory"),
):
    in_dir_p = Path(in_dir)
    assert in_dir_p.exists(), f"{in_dir_p} does not exist"
    assert in_dir_p.is_dir(), f"{in_dir_p} is not a directory"

    ds = fo.Dataset.from_images_dir(in_dir_p)
    ds.compute_metadata()

    logger.info(f"Found {len(ds)} images in {in_dir_p}")

    # count number of images for each size
    sizes = defaultdict(lambda: 0, {})
    for sample in ds:
        metadata = sample.metadata
        width = metadata.width
        height = metadata.height
        sizes[(width, height)] += 1
    # sort with the most frequent size first
    sizes = dict(sorted(sizes.items(), key=lambda x: x[1], reverse=True))
    for k, v in sizes.items():
        # find one example image for each size
        sample = ds.match({"metadata.width": k[0], "metadata.height": k[1]}).first()
        print(f"Size (w, h) {k}: {v} image(s), e.g., {sample.filepath}")


@app.command(help="Crop images in a directory recursively with ROIs from csv")
def crop_imgs(
    in_dir: str = typer.Argument(..., help="Input directory"),
    roi_csv: str = typer.Argument(..., help="CSV file containing ROIs"),
):
    in_dir_p = Path(in_dir)
    assert in_dir_p.exists(), f"{in_dir_p} does not exist"
    assert in_dir_p.is_dir(), f"{in_dir_p} is not a directory"

    roi_csv_p = Path(roi_csv)
    assert roi_csv_p.exists(), f"{roi_csv_p} does not exist"

    # read ROIs from csv, each image size should have one ROI
    rois = defaultdict(lambda: [], {})
    with open(roi_csv_p, "r") as f:
        for roi in csv.DictReader(f):
            ori_width = int(roi["ori_width"])
            ori_height = int(roi["ori_height"])
            roi_x = int(roi["roi_x"])
            roi_y = int(roi["roi_y"])
            roi_width = int(roi["roi_width"])
            roi_height = int(roi["roi_height"])

            key = (ori_width, ori_height)
            assert key not in rois, f"Duplicate ROI for {key}"
            rois[key] = [roi_x, roi_y, roi_width, roi_height]

    # read and crop images
    # write the cropped images to a new directory
    out_dir_p = in_dir_p.parent / f"{in_dir_p.name}_cropped"
    Path(out_dir_p).mkdir(parents=True, exist_ok=True)
    ds = fo.Dataset.from_images_dir(in_dir_p)
    logger.info(f"Found {len(ds)} images in {in_dir_p}")
    for sample in ds:
        img_path = sample.filepath

        # read and crop the image
        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]
        roi = rois[(width, height)]
        cropped_img = img[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]

        # keep the original folder structure
        out_img_p = out_dir_p / Path(img_path).relative_to(in_dir_p.absolute())
        # create the subfolder if not exist
        if not out_img_p.parent.exists():
            out_img_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img_p), cropped_img)
    logger.info(f"Cropped images saved to {out_dir_p}")


@app.command(help="Count num of images without aorta annotations")
def count_n_imgs_no_aorta(
    in_coco_json_p: str = typer.Argument(..., help="Input coco json file"),
    aorta_cat_name: str = typer.Argument("aorta", help="Name of aorta category"),
):
    logger.info(f"Reading {Path(in_coco_json_p).name}...")
    assert Path(in_coco_json_p).exists(), f"{in_coco_json_p} does not exist"
    coco = COCO(in_coco_json_p)

    cats = coco.loadCats(coco.getCatIds())
    cats = sorted(cats, key=lambda x: x["id"], reverse=False)
    # find the category id of aorta
    aorta_cat_id = None
    for cat in cats:
        if cat["name"] == aorta_cat_name:
            aorta_cat_id = cat["id"]
            break
    assert aorta_cat_id is not None, f"Cannot find {aorta_cat_name} in {in_coco_json_p}"
    logger.info(f"Found {aorta_cat_name} with id {aorta_cat_id}")

    n_img_no_aorta = 0
    for img_id in coco.getImgIds():
        anno_ids = coco.getAnnIds(imgIds=img_id)
        annos = coco.loadAnns(anno_ids)
        has_aorta = False
        for anno in annos:
            if anno["category_id"] == aorta_cat_id:
                has_aorta = True
                break
        if not has_aorta:
            n_img_no_aorta += 1
    logger.info(f"Found {n_img_no_aorta} images without {aorta_cat_name}")


@app.command(help="Remove non-aorta annotations from a YOLOv5 dataset")
def keep_only_aorta_labels_in_yolo(
    in_dir: str = typer.Argument(..., help="Input label directory"),
    aorta_class_id: int = typer.Argument(0, help="Class id of aorta"),
):
    txts = list(Path(in_dir).glob("*.txt"))
    logger.info(f"Found {len(txts)} txt files in {in_dir}")
    for txt_p in txts:
        ori_lines, new_lines = [], []
        with open(txt_p, "r") as f:
            ori_lines = f.readlines()
            for line in ori_lines:
                nums = line.split(" ")
                if int(nums[0]) == aorta_class_id:
                    new_lines.append(line)
        with open(txt_p, "w") as new_f:
            new_f.writelines(new_lines)


if __name__ == "__main__":
    app()
