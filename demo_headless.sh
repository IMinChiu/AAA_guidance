#!/bin/bash

VIRTUAL_DISPLAY_NUM=99

OUTPUT_VIDEO="runs/demo_recording_$(date +"%Y-%m-%d_%H-%M-%S").mp4"

# start xvfb server
Xvfb :$VIRTUAL_DISPLAY_NUM -screen 0 1280x720x24 > /dev/null & XVFB_PID=$!

# start recording
ffmpeg -f x11grab -draw_mouse 0 -video_size 1280x720 \
-i :$VIRTUAL_DISPLAY_NUM \
-codec:v libx264 -r 25 $OUTPUT_VIDEO \
> /dev/null 2>&1 < /dev/null & FFMPEG_PID=$!

# start the demo program
DISPLAY=:$VIRTUAL_DISPLAY_NUM QT_QPA_PLATFORM=xcb \
python demo.py "$@" --start-on-open --exit-on-end 

# kill the recording
kill $FFMPEG_PID

# kill xvfb server
kill $XVFB_PID

# success msg
echo -e "Recording saved: $OUTPUT_VIDEO"