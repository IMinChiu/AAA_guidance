from pathlib import Path
import sys
import time
from time import perf_counter
import argparse
from loguru import logger
import os

from predict import Model

from datetime import datetime
from scipy import signal
import plotly.graph_objects as go
import numpy as np
import io
from PIL import Image

import cv2
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# for telemed
import matplotlib.pyplot as plt
import ctypes
from ctypes import *

# 720p
video_w = 1280
video_h = 720


# Copy from detection.py from telemed sample code
class Telemed:
    def __init__(self):
        # starting copy from the origianl main

        # Setting ultrasound size
        # w = 512
        # h = 512
        w = 640
        h = 640

        # Load dll
        # usgfw2 = cdll.LoadLibrary('./usgfw2wrapper_C++_sources/usgfw2wrapper/x64/Release/usgfw2wrapper.dll')
        usgfw2 = cdll.LoadLibrary("./usgfw2wrapper.dll")

        # Ultrasound initialize
        usgfw2.on_init()
        ERR = usgfw2.init_ultrasound_usgfw2()

        # Check probe
        if ERR == 2:
            logger.error("Main Usgfw2 library object not created")
            usgfw2.Close_and_release()
            sys.exit()

        ERR = usgfw2.find_connected_probe()

        if ERR != 101:
            logger.error("Probe not detected")
            usgfw2.Close_and_release()
            sys.exit()

        ERR = usgfw2.data_view_function()

        if ERR < 0:
            logger.error(
                "Main ultrasound scanning object for selected probe not created"
            )
            sys.exit()

        ERR = usgfw2.mixer_control_function(0, 0, w, h, 0, 0, 0)
        if ERR < 0:
            logger.error("B mixer control not returned")
            sys.exit()

        # Probe setting
        res_X = ctypes.c_float(0.0)
        res_Y = ctypes.c_float(0.0)
        usgfw2.get_resolution(ctypes.pointer(res_X), ctypes.pointer(res_Y))

        X_axis = np.zeros(shape=(w))
        Y_axis = np.zeros(shape=(h))
        if w % 2 == 0:
            k = 0
            for i in range(-w // 2, w // 2 + 1):
                if i < 0:
                    j = i + 0.5
                    X_axis[k] = j * res_X.value
                    k = k + 1
                else:
                    if i > 0:
                        j = i - 0.5
                        X_axis[k] = j * res_X.value
                        k = k + 1

        else:
            for i in range(-w // 2, w // 2):
                X_axis[i + w / 2 + 1] = i * res_X.value

        for i in range(0, h - 1):
            Y_axis[i] = i * res_Y.value

        old_resolution_x = res_X.value
        old_resolution_y = res_X.value

        # Image setting
        p_array = (ctypes.c_uint * w * h * 4)()

        fig, ax = plt.subplots()
        usgfw2.return_pixel_values(ctypes.pointer(p_array))
        buffer_as_numpy_array = np.frombuffer(p_array, np.uint)
        reshaped_array = np.reshape(buffer_as_numpy_array, (w, h, 4))

        img = ax.imshow(
            reshaped_array[:, :, 0:3],
            cmap="gray",
            vmin=0,
            vmax=255,
            origin="lower",
            extent=[np.amin(X_axis), np.amax(X_axis), np.amax(Y_axis), np.amin(Y_axis)],
        )

        # starting copy from the original __int__
        self.w = w
        self.h = h

        (
            self.usgfw2,
            self.p_array,
            self.res_X,
            self.res_Y,
            self.old_resolution_x,
            self.old_resolution_y,
            self.X_axis,
            self.Y_axis,
            self.img,
        ) = (
            usgfw2,
            p_array,
            res_X,
            res_Y,
            old_resolution_x,
            old_resolution_y,
            X_axis,
            Y_axis,
            img,
        )

    # return the image from telemed
    def imaging(self):
        self.usgfw2.return_pixel_values(ctypes.pointer(self.p_array))
        buffer_as_numpy_array = np.frombuffer(self.p_array, np.uint)
        reshaped_array = np.reshape(buffer_as_numpy_array, (self.w, self.h, 4))

        self.usgfw2.get_resolution(
            ctypes.pointer(self.res_X), ctypes.pointer(self.res_Y)
        )
        if (
            self.res_X.value != self.old_resolution_x
            or self.res_Y.value != self.old_resolution_y
        ):
            if self.w % 2 == 0:
                k = 0
                for i in range(-self.w // 2, self.w // 2 + 1):
                    if i < 0:
                        j = i + 0.5
                        self.X_axis[k] = j * self.res_X.value
                        k = k + 1
                    else:
                        if i > 0:
                            j = i - 0.5
                            self.X_axis[k] = j * self.res_X.value
                            k = k + 1
            else:
                for i in range(-self.w // 2, self.w // 2):
                    self.X_axis[i + self.w / 2 + 1] = i * self.res_X.value

            for i in range(0, self.h - 1):
                self.Y_axis[i] = i * self.res_Y.value

            self.old_resolution_x = self.res_X.value
            self.old_resolution_y = self.res_X.value

        self.img.set_data(reshaped_array[:, :, 0:3])
        self.img.set_extent(
            [
                np.amin(self.X_axis),
                np.amax(self.X_axis),
                np.amax(self.Y_axis),
                np.amin(self.Y_axis),
            ]
        )

        # Transfer image format to cv2
        img_array = np.asarray(self.img.get_array())
        img_array = img_array[::-1, :, ::-1]  # format same as plt image, RBG to BGR
        return img_array


class Thread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None, args=None):
        QThread.__init__(self, parent)
        self.status = True
        self.cap = True
        self.args = args

        # init telemed
        if args.video is None:
            self.telemed = Telemed()

        # init model
        is_async = (
            True if self.args.jobs == "auto" or int(self.args.jobs) > 1 else False
        )
        self.model = Model(
            model_path=self.args.model,
            imgsz=self.args.img_size,
            classes=self.args.classes,
            device=self.args.device,
            plot_mask=self.args.plot_mask,
            conf_thres=self.args.conf_thres,
            is_async=is_async,
            n_jobs=self.args.jobs,
        )

    def get_stats_fig(self, aorta_widths, aorta_confs, fig_w, fig_h, ts):
        title_font_size = 28
        body_font_size = 24
        img_quality = 100 * np.mean(aorta_confs)
        avg_width = np.mean(aorta_widths)
        max_width = np.max(aorta_widths)
        suggestions = [
            "N/A, within normal limit",
            "Follow up in 5 years",
            "Make an appointment as soon as possible",
        ]
        s = None
        if avg_width < 3:
            s = suggestions[0]
        elif avg_width < 5:
            s = suggestions[1]
        else:
            s = suggestions[2]

        # region smoothing: method 2, keep the peaks
        # peaks = signal.find_peaks(aorta_widths, height=0.5, distance=40)
        # new_y = []
        # # smooth the values between the peaks
        # start = 0
        # end = peaks[0][0]
        # new_y.extend(signal.savgol_filter(aorta_widths[start:end], end - start, 2))
        # for i in range(len(peaks[0]) - 1):
        #     start = peaks[0][i] + 1
        #     end = peaks[0][i + 1]
        #     new_y.append(aorta_widths[peaks[0][i]])  # add peak value
        #     new_y.extend(
        #         signal.savgol_filter(
        #             aorta_widths[start:end],
        #             end - start,  # window size used for filtering
        #             2,
        #         )
        #     )  # order of fitted polynomial
        # # add the last peak
        # new_y.append(aorta_widths[peaks[0][-1]])
        # start = peaks[0][-1] + 1
        # end = len(aorta_widths)
        # new_y.extend(signal.savgol_filter(aorta_widths[start:end], end - start, 2))
        # endregion

        # region smoothing: method 1, do not keep the peaks
        window_size = 53
        if len(aorta_widths) < window_size:
            window_size = len(aorta_widths) - 1
        new_y = signal.savgol_filter(aorta_widths, window_size, 3)
        # endregion

        x = np.arange(1, len(aorta_widths) + 1, dtype=int)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x, y=aorta_widths, mode="lines", line=dict(color="royalblue", width=1)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=new_y,
                mode="lines",
                marker=dict(
                    size=3,
                    color="mediumpurple",
                ),
            )
        )
        fig.update_layout(
            autosize=False,
            width=fig_w,
            height=fig_h,
            margin=dict(l=50, r=50, b=50, t=400, pad=4),
            paper_bgcolor="LightSteelBlue",
            showlegend=False,
        )
        fig.add_annotation(
            text=f"max={max_width:.2f} cm",
            x=np.argmax(aorta_widths),
            y=np.max(aorta_widths),
            xref="x",
            yref="y",
            showarrow=True,
            font=dict(color="#ffffff"),
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8,
        )
        fig.add_annotation(
            text=f"smoothed max={np.max(new_y):.2f} cm",
            x=np.argmax(new_y),
            y=np.max(new_y),
            xref="x",
            yref="y",
            showarrow=True,
            font=dict(color="#ffffff"),
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            ax=-100,
            ay=-50,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8,
        )
        fig.add_annotation(
            text="<b>Report of Abdominal Aorta Examination</b>",
            xref="paper",
            yref="paper",
            x=0.5,
            y=2.3,
            showarrow=False,
            font=dict(size=title_font_size),
        )
        fig.add_annotation(
            text=f"Image acquisition quality: {img_quality:.0f}%",
            xref="paper",
            yref="paper",
            x=0,
            y=2.0,
            showarrow=False,
            font=dict(size=body_font_size),
        )
        fig.add_annotation(
            text=f"Aorta Maximal Width: {max_width:.2f} cm",
            xref="paper",
            yref="paper",
            x=0,
            y=1.8,
            showarrow=False,
            font=dict(size=body_font_size),
        )
        fig.add_annotation(
            text=f"Aorta Maximal Width (Smoothed): {np.max(new_y):.2f} cm",
            xref="paper",
            yref="paper",
            x=0,
            y=1.6,
            showarrow=False,
            font=dict(size=body_font_size),
        )
        fig.add_annotation(
            text=f"Average: {avg_width:.2f} cm",
            xref="paper",
            yref="paper",
            x=0,
            y=1.4,
            showarrow=False,
            font=dict(size=body_font_size),
        )
        fig.add_annotation(
            text=f"Suggestion: {s}",
            xref="paper",
            yref="paper",
            x=0,
            y=1.2,
            showarrow=False,
            font=dict(size=body_font_size),
        )
        fig.add_annotation(
            text=f"Generated at {ts}",
            xref="paper",
            yref="paper",
            x=1,
            y=1,
            showarrow=False,
        )
        return fig

    def run(self):
        one_cm_in_pixels = 48  # hard-coded
        aorta_cm_thre1 = 3
        aorta_cm_thre2 = 5
        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (0, 0, 255)
        green = (0, 255, 0)

        aorta_widths_stats = [0, 0, 0]  # three ranges: <3, 3-5, >5
        aorta_widths = []
        aorta_confs = []

        expected_fps = None
        frame_count = None
        frame_w = None
        frame_h = None
        if self.args.video:
            self.cap = cv2.VideoCapture(self.args.video)
            expected_fps = self.cap.get(cv2.CAP_PROP_FPS)
            secs_per_frame = 1 / expected_fps
            frame_w, frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video source FPS: {expected_fps}")
            logger.info(f"Milliseconds per frame: {secs_per_frame}")
            logger.info(f"Video source resolution (WxH): {frame_w}x{frame_h}")
            logger.info(f"Video source frame count: {frame_count}")
            assert frame_count > 0, "No frame found"

        n_read_frames = 0
        next_frame_to_infer = 0
        next_frame_to_show = 0
        n_repeat_failure = 0
        is_last_failed = False
        start_time = perf_counter()
        while self.status:
            frame = None

            # avoid infinite loop
            if n_repeat_failure > 30:
                break

            # inference
            color_frame, others, results, xyxy, conf = None, None, None, None, None
            if self.model.is_async:
                results = self.model.get_result(next_frame_to_show)
                if results:
                    color_frame, others = results
                    xyxy, conf, _ = others
                    next_frame_to_show += 1

            if self.model.is_async and self.model.is_free_to_infer_async():
                if self.args.video:
                    ret, frame = self.cap.read()

                    if not ret:
                        n_repeat_failure += 1 if is_last_failed else 0
                        is_last_failed = True
                        continue
                else:
                    # read the frame from telemed
                    # TODO(martin): Check read failure
                    frame = self.telemed.imaging()

                n_read_frames += 1
                self.model.predict_async(frame, next_frame_to_infer)
                next_frame_to_infer += 1
            elif not self.model.is_async:
                if self.args.video:
                    ret, frame = self.cap.read()
                    if not ret:
                        n_repeat_failure += 1 if is_last_failed else 0
                        is_last_failed = True
                        continue
                else:
                    # read the frame from telemed
                    # TODO(martin): Check read failure
                    frame = self.telemed.imaging()

                n_read_frames += 1
                results = self.model.predict(frame)
                color_frame, others = results
                xyxy, conf, _ = others  # bbox and confidence
            if results is None:
                continue

            is_last_failed = False

            # check if aorta is within the ROI box, and draw the box
            aorta_width_in_cm = 0
            is_found = xyxy is not None
            is_in_box = False
            is_too_left, is_too_right = False, False
            w, h = color_frame.shape[1], color_frame.shape[0]
            box_w = int(w * 0.1)
            box_h = int(h * 0.5)
            box_top_left = (w // 2 - box_w // 2, h // 4)
            box_bottom_right = (w // 2 + box_w // 2, h // 4 + box_h)
            if xyxy is not None:
                x1, y1, x2, y2 = xyxy

                # check aorta width
                aorta_width_in_cm = (x2 - x1) / one_cm_in_pixels
                aorta_widths.append(aorta_width_in_cm)
                aorta_confs.append(conf)
                if aorta_width_in_cm < aorta_cm_thre1:
                    aorta_widths_stats[0] += 1
                elif aorta_width_in_cm < aorta_cm_thre2:
                    aorta_widths_stats[1] += 1
                else:
                    aorta_widths_stats[2] += 1

                # check whether aorta is in the box
                if (
                    x1 > box_top_left[0]
                    and x2 < box_bottom_right[0]
                    and y1 > box_top_left[1]
                    and y2 < box_bottom_right[1]
                ):
                    is_in_box = True
                is_too_right = x2 > box_bottom_right[0]
                is_too_left = x1 < box_top_left[0]

            # plot ROI box with color status
            box_color = green if is_in_box else red
            color_frame = cv2.rectangle(
                color_frame, box_top_left, box_bottom_right, box_color, 2
            )
            assert not (
                is_too_left and is_too_right
            ), "Cannot be both too left and too right"
            if is_too_left:
                start_p = (box_top_left[0], int(h * 0.9))
                end_p = (box_bottom_right[0], int(h * 0.9))
                cv2.arrowedLine(color_frame, start_p, end_p, red, 3)
            if is_too_right:
                start_p = (box_bottom_right[0], int(h * 0.9))
                end_p = (box_top_left[0], int(h * 0.9))
                cv2.arrowedLine(color_frame, start_p, end_p, red, 3)
            if is_in_box:
                cv2.putText(
                    color_frame,
                    "GOOD",
                    (box_top_left[0], int(h * 0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    green,
                    3,
                )

            # plot aorta width
            text = (
                f"Aorta width: {aorta_width_in_cm:.2f} cm"
                if is_found
                else "Aorta width: N/A"
            )
            cv2.putText(
                color_frame, text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 3
            )

            # region FPS
            fps = None
            if n_read_frames > 0:
                fps = n_read_frames / (perf_counter() - start_time)

                # Slow down the loop if FPS is too high
                if self.args.sync:
                    while fps > expected_fps:
                        time.sleep(0.001)
                        fps = n_read_frames / (perf_counter() - start_time)

                cv2.putText(
                    color_frame,
                    f"FPS: {fps:.2f}",
                    (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    white,
                    3,
                )
            # endregion

            # Creating and scaling QImage
            h, w, ch = color_frame.shape
            img = QImage(color_frame.data, w, h, ch * w, QImage.Format_BGR888)
            scaled_img = img.scaled(video_w, video_h, Qt.KeepAspectRatio)

            # Emit signal
            self.updateFrame.emit(scaled_img)

            if self.args.video:
                progress = 100 * n_read_frames / frame_count
                fps_msg = f", FPS: {fps:.2f}" if fps is not None else ""
                print(
                    f"Processed {n_read_frames}/{frame_count} ({progress:.2f}%) frames"
                    + fps_msg,
                    end="\r" if n_read_frames < frame_count else os.linesep,
                )
                if n_read_frames >= frame_count:
                    logger.info("Finished processing video")
                    break
        if self.args.video:
            self.cap.release()

        if not self.status:
            logger.info("Stopped by user")
            return

        # draw a black image with frame width & height
        # with some text in center indicating generating report
        # it's just a dummy step to make demo more real
        im = np.zeros((frame_h, frame_w, 3), np.uint8)
        cv2.putText(
            im,
            "Generating report for you...",
            (frame_w // 3, frame_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            white,
            3,
        )
        img = QImage(im.data, frame_w, frame_h, ch * w, QImage.Format_BGR888)
        scaled_img = img.scaled(video_w, video_h, Qt.KeepAspectRatio)
        self.updateFrame.emit(scaled_img)
        time.sleep(3)

        # plot aorta width tracing line chart
        now_t = datetime.now()
        ts1 = now_t.strftime("%Y%m%d_%H%M%S")
        ts2 = now_t.strftime("%Y/%m/%d %I:%M:%S")
        Path("runs").mkdir(parents=True, exist_ok=True)
        # np.save("runs/aorta_widths.npy", aorta_widths)
        fig_out_p = f"runs/aorta_report_{ts1}.jpeg"
        fig = self.get_stats_fig(aorta_widths, aorta_confs, video_w, video_h, ts2)

        # This may hang under Windows: https://github.com/plotly/Kaleido/issues/110
        # The workaround is to install older kaleido version (see requirements.txt)
        fig.write_image(fig_out_p)

        logger.info(f"Saved aorta report: {fig_out_p}")
        img_bytes = fig.to_image(format="jpg", width=video_w, height=video_h)
        line_chart = np.array(Image.open(io.BytesIO(img_bytes)))
        line_chart = cv2.cvtColor(line_chart, cv2.COLOR_RGB2BGR)
        h, w, ch = line_chart.shape
        img = QImage(line_chart.data, video_w, video_h, ch * w, QImage.Format_BGR888)
        scaled_img = img.scaled(w, h, Qt.KeepAspectRatio)
        # Emit signal
        self.updateFrame.emit(scaled_img)
        time.sleep(5)

        # keep report open until user closes the window
        while self.status and not self.args.exit_on_end:
            time.sleep(0.1)


class Window(QMainWindow):
    def __init__(self, args=None):
        super().__init__()
        # Title and dimensions
        self.setWindowTitle("Demo")
        self.setGeometry(0, 0, 800, 500)

        # Create a label for the display camera
        self.label = QLabel(self)
        # self.label.setFixedSize(self.width(), self.height())
        self.label.setFixedSize(video_w, video_h)

        # Thread in charge of updating the image
        self.th = Thread(self, args)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.setImage)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        self.button1 = QPushButton("Start")
        self.button2 = QPushButton("Stop/Close")
        self.button1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.button2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        buttons_layout.addWidget(self.button2)
        buttons_layout.addWidget(self.button1)

        right_layout = QHBoxLayout()
        # right_layout.addWidget(self.group_model, 1)
        right_layout.addLayout(buttons_layout, 1)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(right_layout)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Connections
        self.button1.clicked.connect(self.start)
        self.button2.clicked.connect(self.kill_thread)
        self.button2.setEnabled(False)

        if args.start_on_open:
            # start thread
            self.start()

    @Slot()
    def kill_thread(self):
        logger.info("Finishing...")
        self.th.status = False
        time.sleep(1)
        # Give time for the thread to finish
        self.button2.setEnabled(False)
        self.button1.setEnabled(True)
        cv2.destroyAllWindows()
        self.th.exit()
        # Give time for the thread to finish
        time.sleep(1)

    @Slot()
    def start(self):
        logger.info("Starting...")
        self.button2.setEnabled(True)
        self.button1.setEnabled(False)
        self.th.start()
        logger.info("Thread started")

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    # get user inputs using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="path to video file, if None (default) would read from telemed",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_openvino_model/best.xml",
        help="path to model file",
    )
    parser.add_argument("--img-size", type=int, default=640, help="image size")
    parser.add_argument(
        "--classes", nargs="+", type=int, default=[0], help="filter by class"
    )
    parser.add_argument("--device", type=str, default="CPU", help="device to use")
    parser.add_argument("--sync", action="store_true", help="sync video FPS")
    parser.add_argument("--plot-mask", action="store_true", help="plot mask")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="conf thresh")
    parser.add_argument("--jobs", type=str, default=1, help="num of jobs, async if > 1")
    parser.add_argument("--start-on-open", action="store_true", help="start on open")
    parser.add_argument("--exit-on-end", action="store_true", help="exit if video ends")
    args = parser.parse_args()
    assert (
        args.jobs == "auto" or int(args.jobs) > 0
    ), f"--jobs must be > 0 or auto, got {args.jobs}"
    if args.video:
        assert Path(args.video).exists(), f"Video file {args.video} not found"
    assert Path(args.model).exists(), f"Model file {args.model} not found"
    app = QApplication()
    w = Window(args)
    w.show()
    sys.exit(app.exec())
