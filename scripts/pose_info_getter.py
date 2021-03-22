#!/usr/bin/env python
from __future__ import print_function, unicode_literals
import rospy
from std_msgs.msg import String
import yaml

"""
送られてくる情報の並び順は以下の通り
0 : 鼻
1 : 首
2 : 右肩
3 : 右肘
4 : 右手首
5 : 左肩
6 : 左肘
7 : 左手首
8 : 右足付け根
9 : 右膝
10 : 右足首
11 : 左足付け根
12 : 左膝
13 : 左足首
14 : 右目
15 : 左目
16 : 右耳
17 : 左耳
"""

def callback(data):
    pose_info = yaml.load(data.data)

    for p in pose_info:
        print( "画像座標:", p["joint_pos_2d"] )
        print( "三次元座標:", p["joint_pos_3d"] )
        print( "確信度:", p["conf"] )
        print( "-----------" )

        # 送られてくる座標はカメラ座標系（camera_depth_optical_frame）で送られてくるので注意
        # x: 左側が正，右側が負
        # y: 下が正，上が負
        if p["conf"][1]>0 and p["conf"][4]>0 and p["joint_pos_3d"][1][1] > p["joint_pos_3d"][4][1]:
            print("右手上げてる")

        if p["conf"][1]>0 and p["conf"][7]>0 and p["joint_pos_3d"][1][1] > p["joint_pos_3d"][7][1]:
            print("左手上げてる")


    
def main():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/open_pos/pose_info", String, callback)
    rospy.spin()
        
if __name__ == '__main__':
    main()
