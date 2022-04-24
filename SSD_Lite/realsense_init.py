import pyrealsense2 as rs
import cv2
import numpy as np

class camera():
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)
    def read_rgb_image(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image
    def read_aligned_image(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        self.depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image