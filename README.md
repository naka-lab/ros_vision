# ros_vision

## 準備
- Realsenseパッケージをインストール
  ```
  sudo apt-key adv --keyserver keys.gnupg.net --keyserver-option http-proxy=http://proxy.uec.ac.jp:8080 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE  || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
  sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo focal main" -u
  sudo apt-get install librealsense2-dkms
  sudo apt-get install librealsense2-utils
  sudo apt-get install ros-noetic-rgbd-launch
  sudo apt-get install ros-noetic-realsense2-camera
  ```
  ※1つ目のコマンドはproxyを設定しているので，学外でインストールするときは注意

- 物体認識rosパッケージと依存ライブラリをダウンロード
  ```
  pip install open3d opencv-python
  pip install opencv-contrib-python
  pip install gdown
  cd ~/catkin_ws/src
  git clone https://github.com/naka-lab/ros_vision.git
  git clone https://github.com/naka-lab/ros_utils.git
  ```


<!--
- [ここ](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)に従いrealsenseのライブラリをインストール
- realsense rosパッケージをダウンロード
  ```
  cd ~/catkin_ws/src
  git clone https://github.com/pal-robotics/ddynamic_reconfigure.git
  git clone https://github.com/IntelRealSense/realsense-ros.git
  ```
- 物体認識rosパッケージをダウンロード
  ```
  pip install open3d opencv-python
  cd ~/catkin_ws/src
  git clone https://github.com/naka-lab/ros_vision.git
  git clone https://github.com/naka-lab/ros_utils.git
  ```

- 追加パッケージをダウンロード
  ```
  git clone https://github.com/ros-drivers/rgbd_launch.git
  ```
-->

## 平面検出に基づく物体検出と，CNN+SVMによる物体認識
### ノードの実行
- realsenseを起動
  ```
  roslaunch realsense2_camera rs_rgbd.launch align_depth:=True
  ```

- 物体検出・認識ノード起動
  ```
  rosrun ros_vision tabletop_object_recognition.py 
  ```

- 物体認識結果の受信
  - String型の`/object_rec/object_info`というトピック名でyaml形式の文字列送信されます
  - 受信方法は，[これ](https://github.com/naka-lab/ros_vision/blob/master/scripts/object_info_getter.py)を参照

### パラメータ
- 物体検出・認識のパラメータはrosparamで設定
- `rosrun ros_utils param_setting_gui.py`でGUIからも設定可能
- 設定可能なパラメータ
  - `point_cloud/rotate_image`:画像を回転するかどうか 
  - `object_rec/plane_detection/distance_threshold`：平面に含まれる点の距離
  - `object_rec/plane_detection/ransac_n`：平面のパラメータを計算するのに使われる点の数
  - `object_rec/plane_detection/num_iterations`：ransacの繰り返し回数
  - `object_rec/pointcloud_clustering/eps`：1つの物体に含まれる点の密度
  - `object_rec/pointcloud_clustering/min_points`：物体として検出される最小の点の数
  - `object_rec/pointcloud_clustering/rect_min`：物体として検出される最小の縦と横の長さ
  - `object_rec/pointcloud_clustering/rect_max`：物体として検出される最大の縦と横の長さ
  - `object_rec/show_result`：結果を表示するかどうか

### 認識物体の追加・変更
- `~/catkin_ws/src/ros_vision/scripts/objects/`内に`000`，`001`，・・・というディレクトリに認識対象物体の画像を入れる
- ディレクトリ名`000`，`001`，・・・が認識された際のlabelになる
- 物体画像の保存
  ```
  rosrun ros_vision image_saver.py
  ```
  - ウィンドウに表示されている画像をドラッグして保存領域を設定
  - `s`をを押すと画像が連番で，`~/catkin_ws/src/ros_vision/scripts/tmp_img/`に保存される
  - 保存された画像を`~/catkin_ws/src/ros_vision/scripts/objects/`内に`000`，`001`，・・・へ移動する


## SSDに基づく物体検出と認識
### ノードの実行
- realsenseを起動
  ```
  roslaunch realsense2_camera rs_rgbd.launch align_depth:=True
  ```

- 物体検出・認識ノード起動
  ```
  rosrun ros_vision ssd_object_recognition.py
  ```

- 物体認識結果の受信
  - String型の`/ssd_object_rec/object_info`というトピック名でyaml形式の文字列送信されます
  - 受信方法は，[これ](https://github.com/naka-lab/ros_vision/blob/master/scripts/object_info_getter.py)を参照
  - 物体のlabel番号とラベルの対応は[このソース](https://github.com/naka-lab/ros_vision/blob/master/scripts/ssd_object_recognition.py)内の`classNames`を参照
  
### パラメータ
- 物体検出・認識のパラメータはrosparamで設定
- `rosrun ros_utils param_setting_gui.py`でGUIからも設定可能
- 設定可能なパラメータ
  - `ssd_object_rec/conf_thresh`：物体検出するconfidenceのしきい値
  - `ssd_object_rec/show_result`：結果を表示するかどうか

## ARマーカー認識
### ノードの実行
- realsenseを起動
  ```
  roslaunch realsense2_camera rs_rgbd.launch align_depth:=True
  ```

- ARマーカー認識ノード起動
  ```
  rosrun ros_vision ar_marker_recognition.py
  ```

- 物体認識結果の受信
  - String型の`/ar_marker_rec/object_info`というトピック名でyaml形式の文字列送信されます
  - 受信方法は，[これ](scripts/object_info_getter.py)を参照
  - 物体のlabel番号にARマーカーのIDが入る

### ARマーカー
- 0〜9までのARマーカーは[ここ](https://github.com/naka-lab/ros_vision/tree/master/scripts/ARMarker)
- それ以外のマーカーが必要な場合には，[生成プログラム](scripts/ar_gen.py)を実行

### パラメータ
- `point_cloud/rotate_image`:画像を回転するかどうか 



## 人の姿勢推定
### ノードの実行
- realsenseを起動
  ```
  roslaunch realsense2_camera rs_rgbd.launch align_depth:=True
  ```

- OpenPoseノード起動
  ```
  rosrun ros_vision open_pose.py
  ```
  初回起動時はモデルをダウンロードしてくるため起動に時間がかかる


- 姿勢推定結果の受信
  - String型の`/open_pos/pose_info`というトピック名でyaml形式の文字列送信されます
  - 受信方法は，[これ](scripts/pose_info_getter.py)を参照
