#!/usr/bin/env python
from __future__ import print_function, unicode_literals
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
import cv2
import numpy as np
import tf
import yaml
import os
import gdown
import torch
from open_pose_utils.decode_pose import decode_pose
from open_pose_utils.openpose_net import OpenPoseNet

os.chdir( os.path.abspath(os.path.dirname(__file__)) )

# pip install gdown
 
def ilist(lst): return [ int(i) for i in lst ]
def flist(lst): return [ float(i) for i in lst ]

def send_pose_info(pos2d_list, pos3d_list, facedir_list, conf_list):
    pose_info = []

    for n in range(len(pos2d_list)):
        pose_info.append(
            { 
                "joint_pos_2d" : [ flist(p) for p in pos2d_list[n] ],
                "joint_pos_3d" : [ flist(p) for p in pos3d_list[n] ],
                "conf" : flist( conf_list[n] ),
                "facedir" : flist( facedir_list[n] )
             }
        )
    pub_pose.publish( yaml.dump( pose_info )  )

def send_maker( x, y, z, id, scale=0.05, rgb=(1,0,0) ):
    marker_data = Marker()
    marker_data.header.frame_id = "camera_depth_optical_frame"
    marker_data.header.stamp = rospy.Time.now()

    marker_data.ns = "pose"
    marker_data.id = id

    marker_data.action = Marker.ADD

    marker_data.pose.position.x = x
    marker_data.pose.position.y = y
    marker_data.pose.position.z = z

    marker_data.pose.orientation.x=0.0
    marker_data.pose.orientation.y=0.0
    marker_data.pose.orientation.z=1.0
    marker_data.pose.orientation.w=0.0

    marker_data.color.r = rgb[0]
    marker_data.color.g = rgb[1]
    marker_data.color.b = rgb[2]
    marker_data.color.a = 1.0

    marker_data.scale.x = scale
    marker_data.scale.y = scale
    marker_data.scale.z = scale

    marker_data.lifetime = rospy.Duration(1.0)
    marker_data.type = Marker.SPHERE

    pub_marker.publish(marker_data)


def pointcloud_cb( pc2 ):
    xyz = np.frombuffer(pc2.data, dtype=np.float32).reshape(pc2.height, pc2.width, 8)[:,:,0:3]
    img = np.frombuffer(pc2.data, dtype=np.uint8).reshape(pc2.height, pc2.width, 32)[:, :,16:19]

    h, w = img.shape[:2]

    result_img, poses = estimate_pose( img )

    id = 0
    pos2d_list = []
    pos3d_list = []
    conf_list = []
    facedir_list = []
    for n in range(len(poses)):
        pos3d = []
        pos2d = np.array( poses[n][0] )
        conf = poses[n][1] 

        for i in range(len(pos2d)):
            px = int( pos2d[i][0] )
            py = int( pos2d[i][1] )
            c = conf[i]

            p = xyz[py ,px] 
            pos3d.append( p )

            if c!=0:
                send_maker( p[0], p[1], p[2], id )
            id += 1

        # 顔向き計算
        # 0:鼻，14:右目，15:左目，16:右耳，17:左耳
        facedir = (0, 0, 0)
        if conf[0]!=0 and conf[1]!=0 and conf[14]!=0 and conf[15]!=0:
            eyes_2d = (pos2d[14] + pos2d[15])/2
            nose_2d = pos2d[0]
            mouse_2d = ilist( eyes_2d + (nose_2d - eyes_2d)*2.5 )
            mouse_3d = xyz[ mouse_2d[1], mouse_2d[0] ]

            v1 = pos3d[15] - pos3d[14]
            v2 = pos3d[15] - mouse_3d

            facedir = np.cross(v1, v2)
            facedir = facedir/np.linalg.norm(facedir)
            print("fecedir:", facedir)
            print("---------")

            # 可視化
            for i in range(10):
                p = pos3d[0] + facedir*i/10
                send_maker( p[0], p[1], p[2], id, 0.02, (0,1,0) )
                id += 1

        pos2d_list.append( pos2d )
        pos3d_list.append( pos3d )
        conf_list.append( conf )
        facedir_list.append( facedir )

    send_pose_info( pos2d_list, pos3d_list, facedir_list, conf_list )
    cv2.imshow('img',result_img)
    cv2.waitKey(10)


def load_model():
    # 学習済みモデルのロード
    model = "open_pose_utils/pose_model_scratch.pth"
    if not os.path.exists( model ):
        print("download openpose model...")
        gdown.download( "https://drive.google.com/uc?id=1MSExHSkIzGd4pP8qP11PelVXsMtrxJRr", "open_pose_utils/pose_model_scratch.pth" )

    print("loading model...")

    net_weights = torch.load('open_pose_utils/pose_model_scratch.pth', map_location={'cuda:0': device})
    keys = list(net_weights.keys())
    weights_load = {}

    for i in range(len(keys)):
        weights_load[list(net.state_dict().keys())[i]] = net_weights[list(keys)[i]]

    state = net.state_dict()
    state.update(weights_load)
    net.load_state_dict(state)

    net.to(device)


def estimate_pose( ori_img ):
    size = (368, 368)
    img = cv2.resize(ori_img, size, interpolation=cv2.INTER_CUBIC)
    preprocessed_img = img.astype(np.float32) / 255.

    color_mean = [0.485, 0.456, 0.406]
    color_std = [0.229, 0.224, 0.225]

    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

    img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)

    img = torch.from_numpy(img)
    x = img.unsqueeze(0).to(device)

    predicted_outputs, _ = net(x)

    if device=="cuda":
        pafs = predicted_outputs[0][0].cpu().detach().numpy().transpose(1, 2, 0)
        heatmaps = predicted_outputs[1][0].cpu().detach().numpy().transpose(1, 2, 0)
    else:
        pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
        heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)

    pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
    heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)

    pafs = cv2.resize(pafs, (ori_img.shape[1], ori_img.shape[0]), interpolation=cv2.INTER_CUBIC )
    heatmaps = cv2.resize(heatmaps, (ori_img.shape[1], ori_img.shape[0]), interpolation=cv2.INTER_CUBIC )

    _, result_img, joint_list, person_to_joint_assoc = decode_pose(ori_img, heatmaps, pafs)

    N = len(person_to_joint_assoc)

    esimation_results = []
    for n in range(N):
        joints_2d = []
        conf = []

        for i in person_to_joint_assoc[n][:18]:
            i = int(i)

            if i==-1:
                joints_2d.append( (0, 0) )
                conf.append( 0.0 )
                continue

            x, y, c, _, _ = joint_list[i]

            joints_2d.append( (x, y) )
            conf.append( c )

        esimation_results.append( (joints_2d, conf) )

    return result_img, esimation_results
            


def main():
    load_model()
    rospy.Subscriber("/camera/depth_registered/points", PointCloud2, pointcloud_cb, queue_size=1)
    rospy.spin()


if __name__ == '__main__':

    rospy.init_node('open_pose', anonymous=True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    net = OpenPoseNet()
    net.eval()

    pub_pose = rospy.Publisher('/open_pos/pose_info', String, queue_size=1)
    pub_marker = rospy.Publisher("pose_marker", Marker, queue_size = 10)

    main()