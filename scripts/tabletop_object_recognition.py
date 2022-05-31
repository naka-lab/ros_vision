#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
import numpy as np
import open3d as o3d
import cv2
import glob
import os
from sklearn.svm import SVC
import tf
import yaml
import pickle
import sys

PATH = os.path.abspath(os.path.dirname(__file__))
os.chdir(PATH)

# 学習用物体画像の保存先ディレクトリ
OBJECT_DIR = "objects"

# 物体情報のpublisher
pub_objinfo = None

# 特徴量抽出用ネットワーク
feat_extract_net = cv2.dnn.readNetFromCaffe( "bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel")

# 物体認識用SVM
svm = None

# 物体検出
def detect_objects( cloud_points, height, width, depth_thresh ):
    # ポイントクラウドの画素位置を計算
    pix_pos = np.zeros((height, width, 3))
    y_index = np.arange( height )
    x_index = np.arange( width  )
    pix_pos[:,:,0], pix_pos[:,:,1] = np.meshgrid(x_index, y_index)
    pix_pos = pix_pos.reshape( -1, 3 )

    # 範囲を限定する
    skip_num = rospy.get_param("object_rec/plane_detection/depth_skip_num")
    filter_cond = (cloud_points[:,2]<depth_thresh) * (cloud_points[:,2]>0)
    cloud_points = cloud_points[ filter_cond ].reshape(-1, 3)[::skip_num,:]
    pix_pos = pix_pos[ filter_cond ].reshape(-1, 3)[::skip_num,:]
    #color = __color.flatten().reshape(-1, 3)[ filter_cond ][::10,:]

    # ポイントクラウドが0個だと落ちるのでreturn
    if len(cloud_points)==0:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_points)
    #pcd.colors = o3d.utility.Vector3dVector(color.reshape(-1,3)/255.0)
    pcd.normals = o3d.utility.Vector3dVector( pix_pos )
    #o3d.visualization.draw_geometries([pcd])

    # 平面検出
    thresh = rospy.get_param("object_rec/plane_detection/distance_threshold" )
    ransac_n = rospy.get_param("object_rec/plane_detection/ransac_n" )
    num_itr = rospy.get_param("object_rec/plane_detection/num_iterations" )
    plane_model, inliers = pcd.segment_plane(distance_threshold=thresh, ransac_n=ransac_n, num_iterations=num_itr)

    [a, b, c, d] = plane_model
    print("equation: %lfx + %lfy + %lfz + %lf = 0" % (a, b, c, d))

    # バージョンによって関数の名前が違う
    try:
        plane_cloud = pcd.select_down_sample(inliers)
        object_cloud = pcd.select_down_sample(inliers, invert=True)
    except AttributeError:
        plane_cloud = pcd.select_by_index(inliers)
        object_cloud = pcd.select_by_index(inliers, invert=True)

    # 平面から遠い点は除外する
    dist_min = rospy.get_param("object_rec/plane_detection/min_dist_from_plane")
    dist_max = rospy.get_param("object_rec/plane_detection/max_dist_from_plane")
    np_cloud = np.asarray( object_cloud.points )
    dist = np_cloud[:,0]*a + np_cloud[:,1]*b + np_cloud[:,2]*c + d
    object_index = np.where( (dist>dist_min) & (dist<dist_max) )[0]

    try:
        object_cloud = object_cloud.select_down_sample( object_index )
    except AttributeError:
        object_cloud = object_cloud.select_by_index( object_index )

    # 点の数が0個だと落ちる（？）のでリターン
    if len(object_cloud.points)==0:
        return None

    #plane_cloud.paint_uniform_color([1.0, 0, 0])    
    #o3d.visualization.draw_geometries([plane_cloud])
    #o3d.visualization.draw_geometries([object_cloud])


    # クラスタリング
    eps = rospy.get_param("object_rec/pointcloud_clustering/eps" )
    min_pts = rospy.get_param("object_rec/pointcloud_clustering/min_points" )

    #with o3d.utility.VerbosityContextManager(
    #        o3d.utility.VerbosityLevel.Debug) as cm:
    #    labels = np.array(
    #        object_cloud.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))

    labels = np.array(object_cloud.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))

    if len(labels):
        max_label = labels.max()
    else:
        max_label = -1
    print("point cloud has %d clusters" % (max_label + 1) )

    # 画像上の矩形を計算
    pix_pos = np.asarray(object_cloud.normals)[:,0:2].astype(np.int)
    points = np.asarray(object_cloud.points)
    rects = []
    positions = []
    positions_mindepth = []
    pixpos_mindepth = []
    rect_min = rospy.get_param("object_rec/pointcloud_clustering/rect_min", )
    rect_max = rospy.get_param("object_rec/pointcloud_clustering/rect_max", )
    for l in range(max_label+1):
        l_th_obj = (labels==l)
        left_top = np.min(pix_pos[l_th_obj], 0)
        right_bottom = np.max(pix_pos[l_th_obj], 0)
        w = right_bottom[0]-left_top[0]
        h = right_bottom[1]-left_top[1]

        # 下側を少しだけ伸ばす
        right_bottom[1] = min( right_bottom[1]+h*0.3, height )

        left_top = (int(left_top[0]), int(left_top[1]))
        right_bottom = (int(right_bottom[0]), int(right_bottom[1]))

        if w>rect_min and h>rect_min and w<rect_max and h<rect_max: 
            rects.append( (left_top, right_bottom) )
            
            pos = np.average( points[l_th_obj], axis=0 )
            positions.append(pos)

            idx = np.argmin( points[l_th_obj][:,2] )
            positions_mindepth.append( points[l_th_obj][idx] )
            pixpos_mindepth.append( pix_pos[l_th_obj][idx] )

    return rects, positions, object_cloud, plane_cloud, positions_mindepth, pixpos_mindepth


# 物体情報を送信
def send_objects_info(rects, positions, labels, positions_mindepth, positions_center, positions_bottom):
    global pub_objinfo
    br = tf.TransformBroadcaster()
    object_info = []
    for i, p in enumerate(positions):
        br.sendTransform(p,
                tf.transformations.quaternion_from_euler(0, 0, 0),
                rospy.Time.now(), "obj:%d,id:%d"%(i, labels[i]), "camera_depth_optical_frame")
        object_info.append(
            { 
                str("lefttop") : list(rects[i][0]),
                str("rightbottom") : list(rects[i][1]),
                str("position") : [ float(p) for p in positions[i]],
                str("label") : int(labels[i]),
                str("position_mindepth") : [ float(p) for p in positions_mindepth[i]],
                str("position_center") : [ float(p) for p in positions_center[i]],
                str("position_bottom") : [ float(p) for p in positions_bottom[i]],
             }
        )
    pub_objinfo.publish( yaml.dump( object_info )  )


def pointcloud_cb( pc2 ):
    #lag = rospy.get_time()-pc2.header.stamp.secs
    #if lag>0.5:
    #    print("discard queue")
    #    return 

    # この方法だと遅い
    #cloud_points = list(point_cloud2.read_points(pc2, field_names=["x", "y", "z"], skip_nans=False)) 
    #cloud_points = np.array(cloud_points)
    cloud_points = np.frombuffer(pc2.data, dtype=np.float32).reshape(-1, 8)[:,0:3]
    img = np.frombuffer(pc2.data, dtype=np.uint8).reshape(pc2.height, pc2.width, 32)[:, :,16:19]

    height = pc2.height
    width = pc2.width

    # 画像が回転仕立て場合の処理
    if rospy.get_param("point_cloud/rotate_image"):
        img =  cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cloud_points = cv2.rotate( cloud_points.reshape(height, width, 3), cv2.ROTATE_90_CLOCKWISE).reshape(-1, 3)
        height, width = width, height

    # 物体検出
    depth_thresh = rospy.get_param("object_rec/plane_detection/depth_threshold")
    detection_res = detect_objects(cloud_points, height, width, depth_thresh)

    if detection_res!=None:
        rects, positions, object_cloud, plane_cloud, positions_mindepth, pixpos_mindepth = detection_res
    else:
        print("検出失敗")
        return

    object_images = []
    positions_center = []
    pixpos_center = []
    positions_bottom = []
    pixpos_bottom = []
    for r in rects:
        object_images.append( img[ r[0][1]:r[1][1], r[0][0]:r[1][0], : ] )

        y = int((r[0][1]+r[1][1])/2)
        x = int((r[0][0]+r[1][0])/2)
        pixpos_center.append( (x, y)  )
        positions_center.append( cloud_points[y*width+x] )

        y = int((y+r[1][1])/2)
        pixpos_bottom.append( (x, y)  )
        positions_bottom.append( cloud_points[y*width+x] )


    # 物体認識
    if len(object_images):
        labels = svm_recog( object_images )
    else:
        labels = []

    # 検出・認識された物体情報をpublish
    send_objects_info( rects, positions, labels, positions_mindepth, positions_center, positions_center )

    # 結果を表示
    if rospy.get_param("object_rec/show_result" ):
        img_display = np.copy(img)
        pix_pos = np.asarray(object_cloud.normals)[:,0:2].astype(np.int)
        for r, l  in zip(rects, labels):
            cv2.rectangle( img_display, r[0], r[1], (255, 0, 0), 3 )
            cv2.putText(img_display, 'ID: %d'%l, r[0], cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)

        # 平面として推定された点を黒く
        for p in plane_cloud.normals:
            img_display[ int(p[1]), int(p[0]) ] = np.zeros(3)

        # 平面以外のポイントクラウドは緑に
        for p in object_cloud.normals:
            img_display[ int(p[1]), int(p[0]) ] = np.array([0, 255, 0])

        for p in pixpos_mindepth:
            cv2.circle( img_display, tuple(p), 10, (0, 0, 255), 3 )

        for p in pixpos_center:
            cv2.circle( img_display, tuple(p), 10, (0, 255, 0), 3 )

        for p in pixpos_bottom:
            cv2.circle( img_display, tuple(p), 10, (255, 0, 0), 3 )

        cv2.namedWindow("img")
        img_display = cv2.resize(img_display, dsize=None, fx=0.6, fy=0.6)
        cv2.imshow("img", img_display)
        cv2.waitKey(10)
    else:
        cv2.destroyAllWindows()


# 特徴量抽出
def extract_feature( image ):
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
    feat_extract_net.setInput(blob)
    preds = feat_extract_net.forward("pool5/7x7_s1")
    features = preds[0, :, 0, 0]
    return features

def svm_train():
    global svm
    svm = SVC()
    features = []
    labels = []

    print("loading mages...")
    for i in range(100):
        for f in glob.glob( os.path.join( OBJECT_DIR, "%03d"%i, "*" ) ):
            print(f)
            img = cv2.imread( f )
            feat = extract_feature( img )
            features.append( feat )
            labels.append( i )
    print("train svm...")
    svm.fit( features, labels )

    print("done!")

def svm_recog( images ):
    global svm

    if svm==None:
        print( "SVM is not trained. " )
        return [-1]*len(images)
    features = []
    for img in images:
        feat = extract_feature( img )
        features.append( feat )
    
    labels = svm.predict( features )

    return labels

def set_param( name, value ):
    # 存在しなければドフォルト値，存在すればその値を利用
    value = rospy.get_param( name, value )
    rospy.set_param( name, value )

def main():
    global pub_objinfo
    global svm

    svm_model_path = None
    if len(sys.argv)>1:
        print(sys.argv[1])
        svm_model_path = sys.argv[1]

    if svm_model_path==None or svm_model_path=="":
        # モデルが指定されてない場合
        print("モデル指定なし")
        svm_train()
    elif not os.path.exists( svm_model_path ):
        print("モデルファイルなし", svm_model_path)
        print( "save model:", svm_model_path )
        svm_train()
        pickle.dump(svm, open(svm_model_path, 'wb'))
    else:
        # モデルが存在する場合
        print( "load model:", svm_model_path )
        svm = pickle.load(open(svm_model_path, 'rb'))

    rospy.init_node('object_rec', anonymous=True)
    #rospy.Subscriber("/camera/depth_registered/points", PointCloud2, pointcloud_cb, queue_size=1)
    pub_objinfo = rospy.Publisher('/object_rec/object_info', String, queue_size=1)

    # デフォルトパラメータ
    set_param("point_cloud/rotate_image", False )
    set_param("object_rec/plane_detection/depth_skip_num", 5 )
    set_param("object_rec/plane_detection/depth_threshold", 1.3 )
    set_param("object_rec/plane_detection/distance_threshold", 0.03 )
    set_param("object_rec/plane_detection/ransac_n", 3 )
    set_param("object_rec/plane_detection/num_iterations", 1000 )
    set_param("object_rec/plane_detection/min_dist_from_plane", -0.3 )
    set_param("object_rec/plane_detection/max_dist_from_plane", 0.3 )

    set_param("object_rec/pointcloud_clustering/eps", 0.01 )
    set_param("object_rec/pointcloud_clustering/min_points", 10 )
    set_param("object_rec/pointcloud_clustering/rect_min", 20 )
    set_param("object_rec/pointcloud_clustering/rect_max", 100 )
    set_param("object_rec/show_result", True )

    #rospy.spin()
    while not rospy.is_shutdown():
        pc = rospy.wait_for_message( "/camera/depth_registered/points", PointCloud2 )
        pointcloud_cb( pc ) 
        
if __name__ == '__main__':
    main()
