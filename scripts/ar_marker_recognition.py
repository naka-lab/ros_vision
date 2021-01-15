#!/usr/bin/env python
from __future__ import print_function, unicode_literals
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import cv2
import numpy as np
import tf
import yaml
import os
 
 
# 物体情報のpublisher
pub_objinfo = None
dictionary = None
 

def ilist(lst): return [ int(i) for i in lst ]
def flist(lst): return [ float(i) for i in lst ]

# 物体情報を送信
def send_objects_info(rects, positions, labels):
    global pub_objinfo
    br = tf.TransformBroadcaster()
    object_info = []
    for i, p in enumerate(positions):
        br.sendTransform(p,
                tf.transformations.quaternion_from_euler(0, 0, 0),
                rospy.Time.now(), "ar:%d"%(labels[i]), "camera_depth_optical_frame")
        object_info.append(
            { 
                "lefttop" : ilist( rects[i][0] ),
                "rightbottom" : ilist( rects[i][1] ),
                "position" : flist( positions[i] ),
                "label" : int(labels[i])
             }
        )
    pub_objinfo.publish( yaml.dump( object_info )  )

def pointcloud_cb( pc2 ):
    xyz = np.frombuffer(pc2.data, dtype=np.float32).reshape(pc2.height, pc2.width, 8)[:,:,0:3]
    img = np.frombuffer(pc2.data, dtype=np.uint8).reshape(pc2.height, pc2.width, 32)[:, :,16:19]

    h, w = img.shape[:2]

    corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary)

    print( corners )

    N = len(corners)
    rects = []
    positions = []
    labels = []
    for i in range(N):
        lt = np.min( corners[i][0], 0 )
        rb = np.max( corners[i][0], 0 )
        cx, cy = np.average( corners[i][0], 0 )

 
        rects.append( (lt, rb) )
        positions.append( xyz[ int(cy), int(cx) ] )
        labels.append( ids[i] )
    
    img_display = np.copy( img )
    cv2.aruco.drawDetectedMarkers(img_display, corners, ids, (0,255,0))

    cv2.imshow('img',img_display)
    cv2.waitKey(10)
    
    send_objects_info( rects, positions, labels )


def main():
    global pub_objinfo, dictionary
    rospy.init_node('ar_marker_rec', anonymous=True)

    rospy.Subscriber("/camera/depth_registered/points", PointCloud2, pointcloud_cb, queue_size=1)
    pub_objinfo = rospy.Publisher('/ar_marker_rec/object_info', String, queue_size=1)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    rospy.spin()


if __name__ == '__main__':
    main()
