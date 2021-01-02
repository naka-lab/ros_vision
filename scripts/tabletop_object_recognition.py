#!/usr/bin/env python
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
def detect_objects( cloud_points, h, w ):
    # ポイントクラウドの画素位置を計算
    pix_pos = np.zeros((h, w, 3))
    y_index = np.arange( h )
    x_index = np.arange( w  )
    pix_pos[:,:,0], pix_pos[:,:,1] = np.meshgrid(x_index, y_index)
    pix_pos = pix_pos.reshape( -1, 3 )

    # 範囲を限定する
    filter_cond = (cloud_points[:,2]<2.0) * (cloud_points[:,2]>0)
    cloud_points = cloud_points[ filter_cond ].reshape(-1, 3)[::10,:]
    pix_pos = pix_pos[ filter_cond ].reshape(-1, 3)[::10,:]
    #color = __color.flatten().reshape(-1, 3)[ filter_cond ][::10,:]

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
    plane_cloud = pcd.select_by_index(inliers)
    #inlier_cloud.paint_uniform_color([1.0, 0, 0])
    object_cloud = pcd.select_by_index(inliers, invert=True)
    #o3d.visualization.draw_geometries([inlier_cloud])
    #o3d.visualization.draw_geometries([object_cloud])


    # クラスタリング
    eps = rospy.get_param("object_rec/pointcloud_clustering/eps" )
    min_pts = rospy.get_param("object_rec/pointcloud_clustering/min_points" )

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            object_cloud.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # 画像上の矩形を計算
    pix_pos = np.asarray(object_cloud.normals)[:,0:2].astype(np.int)
    points = np.asarray(object_cloud.points)
    rects = []
    positions = []
    rect_min = rospy.get_param("object_rec/pointcloud_clustering/rect_min", )
    for l in range(max_label+1):
        l_th_obj = (labels==l)
        top_left = np.min(pix_pos[l_th_obj], 0)
        bottom_right = np.max(pix_pos[l_th_obj], 0)
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        w = bottom_right[0]-top_left[0]
        h = bottom_right[1]-top_left[1]

        if w>rect_min and h>rect_min: 
            rects.append( (top_left, bottom_right) )
            
            pos = np.average( points[l_th_obj], axis=0 )
            positions.append(pos)


    return rects, positions, object_cloud, plane_cloud


# 物体情報を送信
def send_objects_info(rects, positions, labels):
    global pub_objinfo
    br = tf.TransformBroadcaster()
    object_info = []
    for i, p in enumerate(positions):
        br.sendTransform(p,
                tf.transformations.quaternion_from_euler(0, 0, 0),
                rospy.Time.now(), "obj:%d,id:%d"%(i, labels[i]), "camera_depth_optical_frame")
        object_info.append(
            { 
                "topleft" : rects[i][0],
                "bottomright" : rects[i][1],
                "position" : [ float(p) for p in positions[i]],
                "label" : int(labels[i])
             }
        )
    pub_objinfo.publish( yaml.dump( object_info )  )


def pointcloud_cb( pc2 ):
    # この方法だと遅い
    #cloud_points = list(point_cloud2.read_points(pc2, field_names=["x", "y", "z"], skip_nans=False)) 
    #cloud_points = np.array(cloud_points)
    cloud_points = np.frombuffer(pc2.data, dtype=np.float32).reshape(-1, 8)[:,0:3]
    img = np.frombuffer(pc2.data, dtype=np.uint8).reshape(pc2.height, pc2.width, 32)[:, :,16:19]

    # 物体検出
    rects, positions, object_cloud, plane_cloud = detect_objects(cloud_points, pc2.height, pc2.width)

    object_images = []
    for r in rects:
        object_images.append( img[ r[0][1]:r[1][1], r[0][0]:r[1][0], : ] )

    # 物体認識
    labels = svm_recog( object_images )

    # 検出・認識された物体情報をpublish
    send_objects_info( rects, positions, labels )

    # 結果を表示
    if rospy.get_param("object_rec/show_result" ):
        img_display = np.copy(img)
        pix_pos = np.asarray(object_cloud.normals)[:,0:2].astype(np.int)
        for r, l  in zip(rects, labels):
            cv2.rectangle( img_display, r[0], r[1], (255, 0, 0), 3 )
            cv2.putText(img_display, 'ID: %d'%l, r[0], cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)

        for p in plane_cloud.normals:
            img_display[ int(p[1]), int(p[0]) ] = np.zeros(3)

        cv2.namedWindow("img")
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


def main():
    global pub_objinfo
    svm_train()
    rospy.init_node('object_rec', anonymous=True)
    rospy.Subscriber("/camera/depth_registered/points", PointCloud2, pointcloud_cb, queue_size=1)
    pub_objinfo = rospy.Publisher('/object_rec/object_info', String, queue_size=1)

    # デフォルトパラメータ
    rospy.set_param("object_rec/plane_detection/distance_threshold", 0.01 )
    rospy.set_param("object_rec/plane_detection/ransac_n", 3 )
    rospy.set_param("object_rec/plane_detection/num_iterations", 1000 )
    rospy.set_param("object_rec/pointcloud_clustering/eps", 0.03 )
    rospy.set_param("object_rec/pointcloud_clustering/min_points", 20 )
    rospy.set_param("object_rec/pointcloud_clustering/rect_min", 30 )
    rospy.set_param("object_rec/show_result", True )

    rospy.spin()
        
if __name__ == '__main__':
    main()