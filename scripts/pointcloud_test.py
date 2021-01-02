#!/usr/bin/env python
from __future__ import print_function, unicode_literals
import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
import cv2


def pointcloud_cb( pc2 ):
    #xyz = list(point_cloud2.read_points(pc2, field_names=["x", "y", "z"], skip_nans=False)) 
    #xyz = np.array(xyz)

    xyz = np.frombuffer(pc2.data, dtype=np.float32).reshape(-1, 8)[:,0:3]
    bgr = np.frombuffer(pc2.data, dtype=np.uint8).reshape(pc2.height, pc2.width, 32)[:, :,16:19]

    cv2.namedWindow("img")
    cv2.imshow("img", bgr)

    z = np.copy(xyz[:,2])
    z[ np.isnan(z) ] = 0
    z[ z>5 ] = 5
    depth = z.reshape(pc2.height, pc2.width)/5*255
    depth = np.asarray( depth, dtype=np.uint8 )
    cv2.namedWindow("depth")
    cv2.imshow("depth", depth )
    cv2.waitKey(10)

    return

def main():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/camera/depth_registered/points", PointCloud2, pointcloud_cb )
    rospy.spin()

if __name__ == '__main__':
    main()
