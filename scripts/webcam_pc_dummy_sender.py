#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import cv2
from std_msgs.msg import String, Header
import numpy as np
import tf

pub_pc = None
HEADER = Header(frame_id='camera_depth_optical_frame')
FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=16, datatype=PointField.UINT32, count=1),
    PointField(name='zero', offset=31, datatype=PointField.UINT8, count=1),
]
def pubish_pointcloud2( img, dummy_depth ):
    data = np.zeros( (img.shape[0],img.shape[1],5), dtype=object )
    data[:,:,1] = np.tile( np.linspace(-0.5, 0.5, img.shape[1]), (img.shape[0], 1) )
    data[:,:,2] = np.tile( np.linspace(1.0, 0.0, img.shape[0]).reshape(-1, 1), (1, img.shape[1]) )
    data[:,:, 0] = dummy_depth

    img_int = np.array(img, dtype=int)
    data[:,:,3] = img_int[:,:,0] + img_int[:,:,1]*256 + img_int[:,:,2]*256*256
    data[:,:,4] = 0

    pc = pc2.create_cloud(HEADER, FIELDS, data.reshape(-1,5))

    pc.width = img.shape[1]
    pc.height = img.shape[0]
    pub_pc.publish( pc )


def broadcast_ft():
    br = tf.TransformBroadcaster()
    br.sendTransform( (0, 0, 0), (0, 0, 0, 1), rospy.Time.now(), "camera_depth_optical_frame", "base_link")


def capture():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        ret, frame = capture.read()

        depth = cv2.getTrackbarPos("depth", "webcam")
        
        if frame is not None:
            cv2.imshow('webcam',frame)
            broadcast_ft()
            pubish_pointcloud2( frame, depth/100 )
        else:
            print("エラー発生．カメラ初期化．")
            capture.release()
            capture = cv2.VideoCapture(0)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty("webcam",cv2.WND_PROP_VISIBLE)==0.0:
            break

    capture.release()
    cv2.destroyAllWindows()


def on_change(val):
    pass


def main():
    global pub_pc
    
    rospy.init_node("pc_dummy_sender")

    cv2.namedWindow("webcam")
    cv2.createTrackbar("depth", "webcam", 30, 100, on_change)

    pub_pc = rospy.Publisher('/camera/depth_registered/points', PointCloud2, queue_size=1)
    capture()



if __name__ == '__main__':
    main()
