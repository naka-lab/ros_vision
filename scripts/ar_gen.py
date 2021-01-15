#!/usr/bin/env python
from __future__ import print_function, unicode_literals
import cv2

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

for i in range(10):
    img = aruco.drawMarker(dictionary, i, 200)#第２引数がID　第３引数がピクセルサイズ
    cv2.imwrite( "ar%03d.png"%i, img )
