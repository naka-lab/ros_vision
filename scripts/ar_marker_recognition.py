#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import cv2
import numpy as np
import tf
import yaml
import os
import math
 
 
# 物体情報のpublisher
pub_objinfo = None
dictionary = None
 

def ilist(lst): return [ int(i) for i in lst ]
def flist(lst): return [ float(i) for i in lst ]

# 物体情報を送信
def send_objects_info(rects, positions, labels, quaternions, timestamp):
    global pub_objinfo
    br = tf.TransformBroadcaster()
    object_info = []
    for i, p in enumerate(positions):
        br.sendTransform(p,
                #tf.transformations.quaternion_from_euler(0, 0, 0),
                quaternions[i],
                rospy.Time.now(), "ar:%d"%(labels[i]), "camera_depth_optical_frame")
        object_info.append(
            { 
                str("lefttop") : ilist( rects[i][0] ),
                str("rightbottom") : ilist( rects[i][1] ),
                str("position") : flist( positions[i] ),
                str("label") : int(labels[i]),
                str("stamp") : float(timestamp)
             }
        )
    pub_objinfo.publish( yaml.dump( object_info )  )

def pointcloud_cb( pc2 ):
    xyz = np.frombuffer(pc2.data, dtype=np.float32).reshape(pc2.height, pc2.width, 8)[:,:,0:3]
    img = np.frombuffer(pc2.data, dtype=np.uint8).reshape(pc2.height, pc2.width, 32)[:, :,16:19]

    h, w = img.shape[:2]

    # 画像が回転してた場合の処理
    if rospy.get_param("point_cloud/rotate_image"):
        img =  cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        xyz = cv2.rotate( xyz, cv2.ROTATE_90_CLOCKWISE)
        h, w = w, h

    corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary)

    print( corners )

    N = len(corners)
    rects = []
    positions = []
    labels = []
    quaternions = []
    img_display = np.copy( img )

    for i in range(N):
        lt = np.min( corners[i][0], 0 )
        rb = np.max( corners[i][0], 0 )
        cx, cy = np.average( corners[i][0], 0 )

 
        rects.append( (lt, rb) )
        positions.append( xyz[ int(cy), int(cx) ] )
        labels.append( ids[i] )

        # 姿勢（クオータニオン）を計算
        p1 = xyz[ int(corners[i][0][0][1]), int(corners[i][0][0][0]) ]
        p2 = xyz[ int(corners[i][0][1][1]), int(corners[i][0][1][0]) ]
        p3 = xyz[ int(corners[i][0][3][1]), int(corners[i][0][3][0]) ]

        v = p3 - p1
        v = v/np.linalg.norm(v)

        rz = math.atan2( v[1], v[0] )
        ry =  math.atan2( -v[2]*math.cos(rz), v[0] )

        # 180度ずれた解が出てきてないか確認
        if np.sign(math.cos(rz)*math.cos(ry))!=np.sign(v[0]):
            rz = math.atan2( v[1], v[0] )+math.pi
            ry =  math.atan2( -v[2]*math.cos(rz), v[0] )

        v = p2 - p1
        v = v/np.linalg.norm(v)
        rx = math.atan2( math.sin(rz) * v[2], math.cos(rz)*math.sin(ry)*v[2]-math.cos(ry)*v[0])

        # 180度ずれた解がでてきていないか確認
        if np.sign(v[2])!=np.sign(math.cos(ry)*math.sin(rx)):
            rx += math.pi

        q = tf.transformations.quaternion_from_euler(rz, ry, rx, "rzyx")
        quaternions.append( q )

        # 計算使った点を描画
        cv2.circle( img_display, tuple(ilist(corners[i][0][0])), 10, (255,0,0) )
        cv2.circle( img_display, tuple(ilist(corners[i][0][1])), 10, (0,255,0) )
        cv2.circle( img_display, tuple(ilist(corners[i][0][3])), 10, (0,0,255) )

    cv2.aruco.drawDetectedMarkers(img_display, corners, ids, (0,255,0))
    img_display = cv2.resize(img_display, dsize=None, fx=0.6, fy=0.6)
    cv2.imshow('img',img_display)
    cv2.waitKey(10)
    
    send_objects_info( rects, positions, labels, quaternions, pc2.header.stamp.to_sec() )

def set_param( name, value ):
    # 存在しなければドフォルト値，存在すればその値を利用
    value = rospy.get_param( name, value )
    rospy.set_param( name, value )


def main():
    global pub_objinfo, dictionary
    rospy.init_node('ar_marker_rec', anonymous=True)

    # デフォルトパラメータ
    set_param("point_cloud/rotate_image", False )

    rospy.Subscriber("/camera/depth_registered/points", PointCloud2, pointcloud_cb, queue_size=1)
    pub_objinfo = rospy.Publisher('/ar_marker_rec/object_info', String, queue_size=1)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    rospy.spin()


if __name__ == '__main__':
    main()
