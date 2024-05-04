import typer
from typing import Optional
from pathlib import Path
from loguru import logger
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import shutil
from datetime import datetime
import matplotlib
import os

matplotlib.use("Agg")  # use non-interactive backend
import matplotlib.pyplot as plt

from predict import Model

app = typer.Typer()


@app.command(help="Export videos to images (to a dir per video)")
def export_videos_to_images(
    input_dir: Path = typer.Argument(..., help="Input directory"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    ext: str = typer.Option("avi", help="Video Extension"),
    path_filter: Optional[str] = typer.Option(None, help="input path filter"),
    patient_prefix: Optional[bool] = typer.Option(
        True, help="use patient info as output dir prefix"
    ),
    copy_extent: Optional[bool] = typer.Option(
        True, help="copy extent files to output dir"
    ),
):
    # log all the arguments passed in
    logger.info(f"Function called with arguments: {locals()}")

    # find all video files in input_dir
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_files = list(input_dir.glob(f"**/*.{ext.lower()}"))
    video_files.extend(list(input_dir.glob(f"**/*.{ext.upper()}")))
    logger.info(f"# of avi videos found: {len(video_files)}")
    if path_filter is not None:
        logger.info(f"Filtering videos with {path_filter}")
        video_files = [x for x in video_files if path_filter in str(x)]
        logger.info(f"# of avi videos found after filtering: {len(video_files)}")

    video_files.sort(key=lambda x: x.name)  # sort by name ascending
    # log each video path after filtering, one per line
    logger.info(f"{os.linesep}" + f"{os.linesep}".join([str(x) for x in video_files]))

    # check that all the extent files exist
    # the extent (.csv) should be in the same directory as the video
    # the video filename would start with video_
    # the extent filename would start with extents_
    if copy_extent:
        all_exist = True
        for video_path in video_files:
            extent_filename = video_path.stem.replace("video_", "extents_")
            extent_path = video_path.parent / f"{extent_filename}.csv"
            if not extent_path.exists():
                logger.error(f"Extent file {extent_path} does not exist")
                all_exist = False
        if not all_exist:
            logger.error("Extent files do not exist for all videos")
            return

    for video_path in video_files:
        # copy extent file to output dir
        if copy_extent:
            extent_filename = video_path.stem.replace("video_", "extents_")
            extent_path = video_path.parent / f"{extent_filename}.csv"
            shutil.copy(extent_path, output_dir)

        # Dir structure: Patient_Info / [PATIENT_ID] / [DATE] / video / xxx.avi
        patient_id = (
            video_path.parent.parent.parent.name
        )  # WARNING: Hard-coded based on dir structure

        video_name = video_path.stem
        logger.info(f"Processing video {video_name} of patient {patient_id}")

        # create subdirectory for each video
        sub_dir = output_dir / (
            f"{patient_id}-{video_name}" if patient_prefix else video_name
        )
        sub_dir.mkdir(parents=True, exist_ok=True)

        # read video and export frames
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # padding frame_count with zeros
                cv2.imwrite(str(sub_dir / f"{frame_count:03}.jpg"), frame)
                frame_count += 1
            else:
                break


@app.command(help="Evaluate model on a directory of images")
def eval(
    input_dir: Path = typer.Argument(..., help="Input directory"),
    input_model: Path = typer.Argument(..., help="Input model"),
    imgsz: int = typer.Option(640, help="Image size"),
    class_id: int = typer.Option(0, help="Class id to filter"),
    conf_thresh: float = typer.Option(0.5, help="Confidence threshold"),
    video_ext: str = typer.Option("avi", help="Video Extension"),
    out_dir: Path = typer.Option("runs", help="Output directory"),
    gt_csv_path: Path = typer.Option(
        "results_20230822_aorta_identified_added_by_Ray.csv",
        help="Ground truth csv path",
    ),
    no_extent: Optional[bool] = typer.Option(True, help="no extent file"),
    write_viz: Optional[bool] = typer.Option(False, help="write viz images"),
    gt_column_name: str = typer.Option("aorta_identified", help="Ground truth column"),
):
    # check inputs are valid
    assert input_dir.exists(), f"Input directory {input_dir} does not exist"
    assert input_model.exists(), f"Input model {input_model} does not exist"
    assert gt_csv_path.exists(), f"Ground truth csv {gt_csv_path} does not exist"

    # load model
    model = Model(
        model_path=str(input_model),
        imgsz=imgsz,
        classes=[class_id],  # filter by class id, only aorta
        device="CPU",
        plot_mask=True,
        conf_thres=conf_thresh,
        is_async=False,
        n_jobs=1,
    )

    # setup output directory
    out_dir = Path(out_dir)
    # create a sub output directory of current date and time
    start_t = datetime.now()
    start_timestamp = start_t.strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = out_dir / f"max_aorta_result-{start_timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # log to file
    logger.add(str(out_dir.absolute()) + "/eval_{time}.log")

    out_csv_p = out_dir / "results.csv"
    out_trace_csv_p = out_dir / "trace.csv"
    logger.info(f"Output directory: {out_dir}")

    # find all directories in input_dir
    input_dir = Path(input_dir)
    sub_dirs = [x for x in input_dir.iterdir() if x.is_dir()]
    sub_dirs.sort(key=lambda x: x.name)  # sort sub_dirs by name ascending
    logger.info(f"# of subdirectories found: {len(sub_dirs)}")
    num_sub_dirs = len(sub_dirs)
    has_patient_prefix = False if sub_dirs[0].name.startswith("video") else True

    # setup csv headers
    trace_headers = ["video", "image_idx", "aorta_pixels", "aorta_mm", "conf"]
    headers = ["video", "max_aorta_pixels", "max_aorta_mm", "max_image_idx", "conf"]
    if has_patient_prefix:
        headers.insert(0, "patient_info")

    # loop through each subdirectory of images
    for idx, sub_dir in enumerate(sub_dirs):
        max_aorta_w = -1  # max aorta width in pixels
        max_aorta_w_mm = -1  # max aorta width in mm
        max_aorta_viz = None
        max_aorta_im_path = None
        max_center_x, max_center_y = -1, -1
        max_conf = None
        max_im_n = ""

        # read the extent file of the images
        # the extent file should be in the same directory as the video
        video_filename = (
            sub_dir.name
            if not has_patient_prefix
            else "-".join(sub_dir.name.split("-")[1:])
        )
        extent_filename = video_filename.replace("video_", "extents_")
        extent_file = sub_dir.parent / f"{extent_filename}.csv"
        extents = None
        if not no_extent:
            assert extent_file.exists(), f"Extent file {extent_file} does not exist"
            extents = pd.read_csv(extent_file).to_dict("records")

        logger.info(f"Processing subdir {sub_dir.name} ({idx+1}/{num_sub_dirs})")
        # find all images in sub_dir
        images = list(sub_dir.glob("*.jpg"))
        # Sort the list of images in ascending order by name
        images.sort(key=lambda img: img.name)
        logger.info(f"\t# of images found: {len(images)}")

        # create a viz output directory for each sub_dir
        out_sub_viz_dir = out_dir / sub_dir.name
        Path(out_sub_viz_dir).mkdir(parents=True, exist_ok=True)

        for im_idx, image_path in enumerate(tqdm(images)):
            # read image
            cv_frame = cv2.imread(str(image_path))
            cv_width = cv_frame.shape[1]

            # inference
            viz_frame, results = model.predict(cv_frame)
            bbox_xyxy = results[0]
            conf = results[1]
            masks = results[2]

            # output viz image if the flag is set
            if write_viz:
                cv2.imwrite(
                    str(out_sub_viz_dir / f"{image_path.stem}_viz.jpg"),
                    viz_frame,
                )

            trace_row = [
                f"{sub_dir.name}.{video_ext}",
                image_path.stem,
                -1,
                -1,
                conf,
            ]

            if masks is not None or bbox_xyxy is not None:
                # method 1: find the largest contour
                # find min enclosing circle of mask
                # mask = (masks * 255).astype(np.uint8)
                # contours, _ = cv2.findContours(
                #     mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                # )
                # largest_contour = max(contours, key=cv2.contourArea)
                # (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
                # aorta_width = radius * 2

                # method 2: use the height of the bbox as a measure of aorta width
                # because we observed that the width of the bbox is too large
                aorta_width = bbox_xyxy[3] - bbox_xyxy[1]

                # get physical unit
                w_mm_left, w_mm_right, w_mm_per_pixel = None, None, None
                if not no_extent:
                    w_mm_left = extents[im_idx]["Width-Left(mm)"]
                    w_mm_right = extents[im_idx]["Width-Right(mm)"]
                    assert w_mm_right > 0 and w_mm_left < 0
                    w_mm_per_pixel = (w_mm_right - w_mm_left) / cv_width

                # update trace when aorta is found
                trace_row[2] = aorta_width
                trace_row[3] = aorta_width * w_mm_per_pixel if not no_extent else None

                # output viz image when aorta is found
                cv2.imwrite(
                    str(out_sub_viz_dir / f"{image_path.stem}_viz.jpg"),
                    viz_frame,
                )
                # copy the raw image to the output directory
                shutil.copy(image_path, out_sub_viz_dir)

                if aorta_width > max_aorta_w:
                    max_aorta_w = aorta_width
                    max_aorta_viz = viz_frame.copy()
                    max_aorta_im_path = image_path

                    # Note: only need to calculate the center if using method 1
                    # max_center_x = center_x
                    # max_center_y = center_y

                    max_im_n = image_path.stem
                    max_conf = conf
                    logger.info(
                        f"\tNew max aorta (pixels): {max_aorta_w:.2f}, conf: {max_conf:.2f}"
                    )

                    # convert pixels to mm
                    max_aorta_w_mm = (
                        max_aorta_w * w_mm_per_pixel if not no_extent else None
                    )

            # save trace to csv
            df = pd.DataFrame([trace_row], columns=trace_headers)
            df.to_csv(
                out_trace_csv_p,
                mode="a",
                header=not out_trace_csv_p.exists(),
                index=False,
                float_format="%.3f",
            )

        if max_aorta_w > 0:
            logger.info(f"\tMax aorta (pixels): {max_aorta_w:.2f}")
            # copy the raw image to the output directory
            out_raw_p = out_dir / f"raw_{sub_dir.name}_{max_im_n}.jpg"
            shutil.copy(max_aorta_im_path, out_raw_p)

            # method 1 viz: draw enclosing circle on max_aorta_viz
            # plot circle on max_aorta_viz
            # cv2.circle(
            #     max_aorta_viz,
            #     (int(max_center_x), int(max_center_y)),
            #     int(max_aorta_w / 2),
            #     (0, 255, 0),
            #     2,
            # )

            # region Save the image with extent
            # convert the BGR image to RGB image
            out_viz_p = out_dir / f"viz_{sub_dir.name}_{max_im_n}.jpg"
            max_aorta_viz_rgb = cv2.cvtColor(max_aorta_viz, cv2.COLOR_BGR2RGB)
            # Use matplotlib to save the image
            # Get the size of the image in inches
            dpi = plt.rcParams["figure.dpi"]  # Get the default dpi value
            figsize = (
                max_aorta_viz_rgb.shape[1] / dpi,
                max_aorta_viz_rgb.shape[0] / dpi,
            )  # width, height
            # Create a new figure with the same aspect ratio as the image
            fig = plt.figure(figsize=figsize)
            if not no_extent:
                # specify the extent of the image in the form [xmin, xmax, ymin, ymax]
                extent = [
                    extents[im_idx]["Width-Left(mm)"],
                    extents[im_idx]["Width-Right(mm)"],
                    extents[im_idx]["Depth-Bottom(mm)"],
                    extents[im_idx]["Depth-Top(mm)"],
                ]
                plt.imshow(max_aorta_viz_rgb, extent=extent)
                plt.xlabel("Width [mm]")
                plt.ylabel("Depth [mm]")
            else:
                plt.imshow(max_aorta_viz_rgb)
            plt.savefig(str(out_viz_p))
            plt.close(fig)
            # cv2.imwrite(str(out_viz_p), max_aorta_viz)
            # endregion
        else:
            logger.warning(f"\tNo aorta found in {sub_dir.name}")
        patient_info = sub_dir.name.split("-")[0] if has_patient_prefix else ""
        row = [
            f"{sub_dir.name}.{video_ext}",
            max_aorta_w,
            max_aorta_w_mm,
            max_im_n,
            max_conf,
        ]
        if has_patient_prefix:
            row.insert(0, patient_info)
            # remove patient info from sub_dir name
            video_name = "-".join(sub_dir.name.split("-")[1:]) + f".{video_ext}"
            row[1] = video_name

        # export results to csv
        # If file does not exist, this will create it, otherwise it will append to the file
        df = pd.DataFrame([row], columns=headers)
        df.to_csv(
            out_csv_p,
            mode="a",
            header=not out_csv_p.exists(),
            index=False,
            float_format="%.3f",
        )

    # join the results with ground truth to add the ground truth column
    # df_results = pd.read_csv(out_csv_p)
    # df_gt = pd.read_csv(gt_csv_path)[["video", gt_column_name]]  # id & gt columns
    # df_gt_first = df_gt.drop_duplicates(subset="video", keep="first")  # avoid new rows
    # df_merged = pd.merge(df_results, df_gt_first, on="video", how="left")
    # df_merged.to_csv(out_csv_p, header=True, index=False, float_format="%.3f")

    # # show stats
    # value_counts_with_nan = df_merged[gt_column_name].value_counts(dropna=False)
    # total = len(df_merged)
    # percentage = (value_counts_with_nan / total) * 100
    # # Combine value counts and percentages into a DataFrame for better visualization
    # stats = pd.DataFrame({"Count": value_counts_with_nan, "Percentage": percentage})
    # logger.info(stats)

    logger.info(f"Done! Results written to {out_csv_p}")


@app.command(help="Copy source images to viz result folder")
def copy_srcimg_to_vizdir(
    src_img_dir: Path = typer.Argument(..., help="Source Images root directory"),
    out_viz_dir: Path = typer.Argument(..., help="Target viz dirtectory"),
):
    vizs = list(Path(out_viz_dir).glob("**/*.jpg"))
    for viz in vizs:
        splits = viz.stem.split("_")
        ori_img = Path(src_img_dir) / splits[1] / f"{splits[2]}.jpg"
        shutil.copy(ori_img, Path(out_viz_dir) / f"{viz.stem}_src.jpg")


if __name__ == "__main__":
    app()
