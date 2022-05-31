#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
import numpy as np
import message_filters
from image_geometry import PinholeCameraModel
from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import threading

pub_pc = None
HEADER = Header(frame_id='camera_depth_optical_frame')
FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=16, datatype=PointField.UINT32, count=1),
    PointField(name='zero', offset=31, datatype=PointField.UINT8, count=1),
]

class Depth2PointCloud():
    def __init__(self):
        camera_info = rospy.wait_for_message( "/camera/aligned_depth_to_color/camera_info", CameraInfo )
        camera_model = PinholeCameraModel()
        camera_model.fromCameraInfo( camera_info )
        width = camera_info.width
        height = camera_info.height

        self.vecs = np.zeros( (height, width,3) )
        for y in range(height):
            for x in range(width):
                v = np.array(camera_model.projectPixelTo3dRay( (x, y) ))
                v /= v[2]
                self.vecs[y,x] = v

        self.br = CvBridge()
        self.pub_pc = rospy.Publisher('/points', PointCloud2, queue_size=1)
        self.sub_depth = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image )
        self.sub_image = message_filters.Subscriber("/camera/color/image_raw", Image )
        self.syncronizer = message_filters.ApproximateTimeSynchronizer( [self.sub_depth, self.sub_image], 10, 0.5 )

        self.current_rgb = None
        self.current_points = None
        self.current_depth = None
        self.event_cap_img = threading.Event()
        self.syncronizer.registerCallback( self.callback )


    def callback( self, depth, rgb ):
        #rgb_img = self.br.compressed_imgmsg_to_cv2( rgb, "bgr8" )
        rgb_img = self.br.imgmsg_to_cv2( rgb, "bgr8" )

        depth_data = np.frombuffer(depth.data, dtype=np.uint16).reshape( rgb_img.shape[:2] )
        points = self.vecs * depth_data.reshape(rgb_img.shape[0], rgb_img.shape[1], 1) / 1000.0

        self.current_rgb = rgb_img
        self.current_points = points
        self.current_depth = depth_data
        self.event_cap_img.set()


    def get_latest_data(self, timeout=None):
        self.event_cap_img.wait(timeout)
        self.event_cap_img.clear()
        return self.current_rgb, self.current_points

    def get_depth_img(self):
        depth_img = np.zeros( self.current_depth.shape[:2], dtype=np.uint8 )
        depth_img[:,:] = self.current_depth/10
        return depth_img

    def pubish_pointcloud2(self):
        rgb_img = self.current_rgb
        points = self.current_points
        data = np.zeros( (rgb_img.shape[0],rgb_img.shape[1],5), dtype=object )
        data[:,:,1] = points[:,:,1]
        data[:,:,2] = points[:,:,2]
        data[:,:,0] = points[:,:,0]

        rgb_img = np.array(rgb_img, dtype=int)
        data[:,:,3] = rgb_img[:,:,0] + rgb_img[:,:,1]*256 + rgb_img[:,:,2]*256*256
        data[:,:,4] = 0

        pc = pc2.create_cloud(HEADER, FIELDS, data.reshape(-1,5))

        pc.width = rgb_img.shape[1]
        pc.height = rgb_img.shape[0]
        pc.header.stamp = rospy.Time.now()
        self.pub_pc.publish( pc )


if __name__=="__main__":
    rospy.init_node('depth2cloud')
    d2p = Depth2PointCloud()

    while not rospy.is_shutdown():
        img, points = d2p.get_latest_data()
        d2p.pubish_pointcloud2()
        depth_img =  d2p.get_depth_img()

        cv2.imshow('img', img)
        cv2.imshow('depth', depth_img)
        cv2.waitKey(10)


