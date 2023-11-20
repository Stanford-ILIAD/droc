import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco
import imageio

class MarkSearch:

    def __init__(self):
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters =  aruco.DetectorParameters()

        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)

    def find_marker(self, image):
        """
        Obtain marker id list from still image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)
        
        if ids is None:
            return (None,None),None
        
        ids = ids.flatten()

        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers

            #u = np.mean((topLeft[0], bottomRight[0])).astype(int)
            #v = np.mean((topLeft[1], bottomRight[1])).astype(int)
            u = np.mean((topLeft[0], bottomRight[0]))
            v = np.mean((topLeft[1], bottomRight[1]))

            cv2.circle(image, (int(u),int(v)), 5, (0,0,255), -1)
            cv2.imshow("rgb", image)
            cv2.waitKey(1)

            return (u,v), image


class RealsenseStreamer():
    def __init__(self, serial_no=None):

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if serial_no is not None:
            self.config.enable_device(serial_no)


        self.width = 640
        self.height = 480

        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.align_to_color = rs.align(rs.stream.color)

        # Start streaming
        self.pipe_profile = self.pipeline.start(self.config)
        self.is_open = True

        self.serial_no = serial_no
        # print serial no
        #for d in rs.context().devices:
        #    self.serial_no = d.get_info(rs.camera_info.serial_number)
        #    break

        # Intrinsics & Extrinsics
        frames = self.pipeline.wait_for_frames()
        frames = self.align_to_color.process(frames)
        depth_frame = frames.get_depth_frame()

        color_frame = frames.get_color_frame()

        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        self.depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        self.colorizer = rs.colorizer()

        self.K = np.array([[self.depth_intrin.fx, 0, self.depth_intrin.ppx],
                           [0, self.depth_intrin.fy, self.depth_intrin.ppy],
                           [0, 0, 1]])


    def deproject(self, px, depth_frame):
        u,v = px
        depth = depth_frame.get_distance(u,v)
        xyz = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [u,v], depth)
        return xyz

    def capture_rgb(self):
        color_frame = None
        color_image = None
        # if not self.is_open:
        #     self.pipeline.start(self.config)
        #     self.is_open = True
        #try:            
        while self.is_open:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame is not None:
                color_image = np.asanyarray(color_frame.get_data())
                break
        # finally:
        #     self.pipeline.stop()
        #     self.is_open = False
        return color_image
        

    def capture_rgbd(self):
        frame_error = True
        while frame_error:
            try:
                frames = self.align_to_color.process(frames)  
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                frame_error = False
            except:
                frames = self.pipeline.wait_for_frames()
                continue
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        return color_frame, color_image, depth_frame, depth_image

    def stop_stream(self):
        self.pipeline.stop()

    def show_image(self, image):
        cv2.imshow('img', image)
        cv2.waitKey(0)


