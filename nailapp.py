
# ステップ1. インポート
import PySimpleGUI as sg  # PySimpleGUIをsgという名前でインポート
import os  # OS依存の操作（Pathやフォルダ操作など）用ライブラリのインポート
import numpy as np  # numpyのインポート
import cv2  # OpenCV（python版）のインポート
import mediapipe as mp  #mediapipeのインクルード
from datetime import datetime  # 追加
from math import atan2, pi, sqrt, sin, cos  # 追加
# ---------  関数群  ----------

# アイコンを読み込む関数
def load_image(path):
    icon = cv2.imread(path, -1)
    if icon.data == 0:
        print('画像が読み込めませんでした')
    return icon

# 画像をリサイズする関数
def img_resize(img, scale):
    h, w  = img.shape[:2]
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

# 画像を保存する関数
def save_image(img):
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = "./" + date + ".png"
    cv2.imwrite(path, img) # ファイル保存

# 画像を合成する関数
def merge_images(bg, fg_alpha, s_x, s_y):
    alpha = fg_alpha[:,:,3]  # アルファチャンネルだけ抜き出す(要は2値のマスク画像)
    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # grayをBGRに
    alpha = alpha / 255.0    # 0.0〜1.0の値に変換
    
    fg = fg_alpha[:,:,:3]
    
    f_h, f_w, _ = fg.shape # アルファ画像の高さと幅を取得
    b_h, b_w, _ = bg.shape # 背景画像の高さを幅を取得
    
    # 画像の大きさと開始座標を表示
#    print("f_w:{} f_h:{} b_w:{} b_h:{} s({}, {})".format(f_w, f_h, b_w, b_h, s_x, s_y))
    
    bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] * (1.0 - alpha)).astype('uint8') # アルファ以外の部分を黒で合成
    bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] + (fg * alpha)).astype('uint8')  # 合成
    
    return bg

# 画像リサイズ関数（高さが指定した値になるようにリサイズ (アスペクト比を固定)）
def scale_to_height(img, height):
    h, w = img.shape[:2]
    width = round(w * (height / h))
    dst = cv2.resize(img, dsize=(width, height))
    
    return dst
    


# ---- 顔認識エンジンセット ----
#mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_detection = mp.solutions.face_detection

# ---- 大域変数 ----
display_size = (400, 300)  # ディスプレイサイズ
isOpened = 0   # カメラがオープンになっているかどうかのフラグ
isWriting = 0   # 録画ON/OFF
IMAGE_PATH = "./nc73730.png"  # 画像パス
IMAGE_PATH1 = "./1-1.png"######
IMAGE_PATH2 = "./1-2.png"######
IMAGE_PATH3 = "./2-1.png"######
IMAGE_PATH4 = "./2-2.png"######
IMAGE_PATH5 = "./3-1.png"######
IMAGE_PATH6 = "./3-2.png"######


# ネイル画像読み込み
testicon = load_image(IMAGE_PATH)
finger1_1 = load_image(IMAGE_PATH1)####
finger1_2 = load_image(IMAGE_PATH2)####
finger2_1 = load_image(IMAGE_PATH3)####
finger2_2 = load_image(IMAGE_PATH4)####
finger3_1 = load_image(IMAGE_PATH5)####
finger3_2 = load_image(IMAGE_PATH6)####


def image_rotation(nail, rad, z):
    #遠近
    scale=-1.0/0.11*z
    
    #画像回転変換行列
    height, width = nail.shape[:2]###
    width2 = int(height*sin(rad) + width*cos(rad)+3)
    height2 = int(height*cos(rad) + width*sin(rad)+3)
    rotate = rad * 180 / pi  #度

    #アフィン変換行列を作成する
    center = (width/2, height/2)
    diff = (width2/2 - width/2, height2/2 - height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center, angle=rotate, scale=scale)
    rotate_matrix[0, 2] += diff[0]
    rotate_matrix[1, 2] += diff[1]

    #アフィン変換行列を画像に適用する
    rot_image = cv2.warpAffine(nail, rotate_matrix, (width2,height2)) ###
    return rot_image
    
def right_fitting(img1, img2):
    p_1 = results.right_hand_landmarks.landmark[4] #親第一
    x_1 = int(p_1.x * width) - int(img1.shape[1] / 2)
    y_1 = int(p_1.y * height) - int(img1.shape[0] / 2)
    p_12 = results.right_hand_landmarks.landmark[3] #親第二
    x_12 = int(p_12.x * width) - int(img1.shape[1] / 2)
    y_12 = int(p_12.y * height) - int(img1.shape[0] / 2)
    diff = (p_1.x * width - p_12.x * width, p_1.y * height - p_12.y * height)
    rad1 = atan2(diff[1], diff[0]) + pi/2.0 #ラジアン
#                    print(str(p_1.z))
    rotated_image1 = image_rotation(img1, -rad1, p_1.z)

    p_2 = results.right_hand_landmarks.landmark[8]
    x_2 = int(p_2.x * width) - int(img1.shape[1] / 2)
    y_2 = int(p_2.y * height) - int(img1.shape[0] / 2)
    p_22 = results.right_hand_landmarks.landmark[7] #人第二
    x_22 = int(p_22.x * width) - int(img1.shape[1] / 2)
    y_22 = int(p_22.y * height) - int(img1.shape[0] / 2)
    diff = (p_2.x * width - p_22.x * width, p_2.y * height - p_22.y * height)
    rad2 = atan2(diff[1], diff[0]) + pi/2.0 #ラジアン
    rotated_image2 = image_rotation(img1, -rad2, p_2.z)

    p_3 = results.right_hand_landmarks.landmark[12]  # 中第一
    x_3 = int(p_3.x * width) - int(img2.shape[1] / 2)
    y_3 = int(p_3.y * height) - int(img2.shape[0] / 2)
    p_32 = results.right_hand_landmarks.landmark[11]  # 中第二
    x_32 = int(p_32.x * width) - int(img2.shape[1] / 2)
    y_32 = int(p_32.y * height) - int(img2.shape[0] / 2)
    diff = (p_3.x * width - p_32.x * width, p_3.y * height - p_32.y * height)
    rad3 = atan2(diff[1], diff[0]) + pi/2.0 #ラジアン
    rotated_image3 = image_rotation(img2, -rad3, p_3.z)

    p_4 = results.right_hand_landmarks.landmark[16] #薬第一
    x_4 = int(p_4.x * width) - int(img1.shape[1] / 2)
    y_4 = int(p_4.y * height) - int(img1.shape[0] / 2)
    p_42 = results.right_hand_landmarks.landmark[15] #薬第二
    x_42 = int(p_42.x * width) - int(img1.shape[1] / 2)
    y_42 = int(p_42.y * height) - int(img1.shape[0] / 2)
    diff = (p_4.x * width - p_42.x * width, p_4.y * height - p_42.y * height)
    rad4 = atan2(diff[1], diff[0]) + pi/2.0 #ラジアン
    rotated_image4 = image_rotation(img1, -rad4, p_4.z)

    p_5 = results.right_hand_landmarks.landmark[20] #小指第一
    x_5 = int(p_5.x * width) - int(img1.shape[1] / 2)
    y_5 = int(p_5.y * height) - int(img1.shape[0] / 2)
    p_52 = results.right_hand_landmarks.landmark[19] #小指第二
    x_52 = int(p_52.x * width) - int(img1.shape[1] / 2)
    y_52 = int(p_52.y * height) - int(img1.shape[0] / 2)
    diff = (p_5.x * width - p_52.x * width, p_5.y * height - p_52.y * height)
    rad5 = atan2(diff[1], diff[0]) + pi/2.0 #ラジアン
    rotated_image5 = image_rotation(img1, -rad5,p_5.z)
    if (0 <= y_1) and (y_1 <= (height - int(rotated_image1.shape[0]))) and (0 <= x_1) and (x_1 <= (width - int(rotated_image1.shape[1]))):
        # 画面の範囲内だったら画像を合成
        nailfitting = merge_images(image, rotated_image1, x_1, y_1)
    if (0 <= y_2) and (y_2 <= (height - int(rotated_image2.shape[0]))) and (0 <= x_2) and (x_2 <= (width - int(rotated_image2.shape[1]))):
        nailfitting = merge_images(image, rotated_image2, x_2, y_2)

    if (0 <= y_3) and (y_3 <= (height - int(rotated_image3.shape[0]))) and (0 <= x_3) and (x_3 <= (width - int(rotated_image3.shape[1]))):
        nailfitting = merge_images(image, rotated_image3, x_3, y_3)

    if (0 <= y_4) and (y_4 <= (height - int(rotated_image4.shape[0]))) and (0 <= x_4) and (x_4 <= (width - int(rotated_image4.shape[1]))):
        nailfitting = merge_images(image, rotated_image4, x_4, y_4)

    if (0 <= y_5) and (y_5 <= (height - int(rotated_image5.shape[0]))) and (0 <= x_5) and (x_5 <= (width - int(rotated_image5.shape[1]))):
        nailfitting = merge_images(image, rotated_image5, x_5, y_5)
    return nailfitting
    
def left_fitting(img1, img2):
    p_1 = results.left_hand_landmarks.landmark[4] #親第一
    x_1 = int(p_1.x * width) - int(img1.shape[1] / 2)
    y_1 = int(p_1.y * height) - int(img1.shape[0] / 2)
    p_12 = results.left_hand_landmarks.landmark[3] #親第二
    x_12 = int(p_12.x * width) - int(img1.shape[1] / 2)
    y_12 = int(p_12.y * height) - int(img1.shape[0] / 2)
    diff = (p_1.x * width - p_12.x * width, p_1.y * height - p_12.y * height)
    rad1 = atan2(diff[1], diff[0]) + pi/2.0 #ラジアン
#                    print(str(p_1.z))
    rotated_image1 = image_rotation(img1, -rad1, p_1.z)

    p_2 = results.left_hand_landmarks.landmark[8]
    x_2 = int(p_2.x * width) - int(img1.shape[1] / 2)
    y_2 = int(p_2.y * height) - int(img1.shape[0] / 2)
    p_22 = results.left_hand_landmarks.landmark[7] #人第二
    x_22 = int(p_22.x * width) - int(img1.shape[1] / 2)
    y_22 = int(p_22.y * height) - int(img1.shape[0] / 2)
    diff = (p_2.x * width - p_22.x * width, p_2.y * height - p_22.y * height)
    rad2 = atan2(diff[1], diff[0]) + pi/2.0 #ラジアン
    rotated_image2 = image_rotation(img1, -rad2, p_2.z)

    p_3 = results.left_hand_landmarks.landmark[12]  # 中第一
    x_3 = int(p_3.x * width) - int(img2.shape[1] / 2)
    y_3 = int(p_3.y * height) - int(img2.shape[0] / 2)
    p_32 = results.left_hand_landmarks.landmark[11]  # 中第二
    x_32 = int(p_32.x * width) - int(img2.shape[1] / 2)
    y_32 = int(p_32.y * height) - int(img2.shape[0] / 2)
    diff = (p_3.x * width - p_32.x * width, p_3.y * height - p_32.y * height)
    rad3 = atan2(diff[1], diff[0]) + pi/2.0 #ラジアン
    rotated_image3 = image_rotation(img2, -rad3, p_3.z)

    p_4 = results.left_hand_landmarks.landmark[16] #薬第一
    x_4 = int(p_4.x * width) - int(img1.shape[1] / 2)
    y_4 = int(p_4.y * height) - int(img1.shape[0] / 2)
    p_42 = results.left_hand_landmarks.landmark[15] #薬第二
    x_42 = int(p_42.x * width) - int(img1.shape[1] / 2)
    y_42 = int(p_42.y * height) - int(img1.shape[0] / 2)
    diff = (p_4.x * width - p_42.x * width, p_4.y * height - p_42.y * height)
    rad4 = atan2(diff[1], diff[0]) + pi/2.0 #ラジアン
    rotated_image4 = image_rotation(img1, -rad4, p_4.z)

    p_5 = results.left_hand_landmarks.landmark[20] #小指第一
    x_5 = int(p_5.x * width) - int(img1.shape[1] / 2)
    y_5 = int(p_5.y * height) - int(img1.shape[0] / 2)
    p_52 = results.left_hand_landmarks.landmark[19] #小指第二
    x_52 = int(p_52.x * width) - int(img1.shape[1] / 2)
    y_52 = int(p_52.y * height) - int(img1.shape[0] / 2)
    diff = (p_5.x * width - p_52.x * width, p_5.y * height - p_52.y * height)
    rad5 = atan2(diff[1], diff[0]) + pi/2.0 #ラジアン
    rotated_image5 = image_rotation(img1, -rad5,p_5.z)
    if (0 <= y_1) and (y_1 <= (height - int(rotated_image1.shape[0]))) and (0 <= x_1) and (x_1 <= (width - int(rotated_image1.shape[1]))):
        # 画面の範囲内だったら画像を合成
        nailfitting = merge_images(image, rotated_image1, x_1, y_1)
    if (0 <= y_2) and (y_2 <= (height - int(rotated_image2.shape[0]))) and (0 <= x_2) and (x_2 <= (width - int(rotated_image2.shape[1]))):
        nailfitting = merge_images(image, rotated_image2, x_2, y_2)

    if (0 <= y_3) and (y_3 <= (height - int(rotated_image3.shape[0]))) and (0 <= x_3) and (x_3 <= (width - int(rotated_image3.shape[1]))):
        nailfitting = merge_images(image, rotated_image3, x_3, y_3)

    if (0 <= y_4) and (y_4 <= (height - int(rotated_image4.shape[0]))) and (0 <= x_4) and (x_4 <= (width - int(rotated_image4.shape[1]))):
        nailfitting = merge_images(image, rotated_image4, x_4, y_4)

    if (0 <= y_5) and (y_5 <= (height - int(rotated_image5.shape[0]))) and (0 <= x_5) and (x_5 <= (width - int(rotated_image5.shape[1]))):
        nailfitting = merge_images(image, rotated_image5, x_5, y_5)
    return nailfitting
    



# ステップ2. デザインテーマの設定
sg.theme('LightBrown')

# ステップ3. ウィンドウの部品とレイアウト
layout = [
          [sg.Button('カメラ', key='camera')],
          [sg.Image(filename='', size=display_size, key='-input_image-')],
          [sg.Text('ネイル選択', size=(10, 1)), sg.Combo(('なし', 'nail1', 'nail2', 'nail3'), default_value='なし', size=(7, 1), key='nail')],########
#          [sg.Text('ランドマークの表示', size=(15, 1)), sg.Combo(('あり', 'なし'), default_value='なし', size=(5, 1), key='landmark')],,     font("Helvetica", 10, "bold")
          [sg.Button('動画保存', key='save'), sg.Button('終了', key='exit')],
          [sg.Output(size=(86,10))]
          ]

# ステップ4. ウィンドウの生成
window = sg.Window('人体を認識するツール', layout, location=(400, 20))

# ステップ5. カメラ，mediapipeの初期設定
#cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=.5, min_tracking_confidence=.5) as holistic:
    # ステップ6. イベントループ
    while True:
        event, values = window.read(timeout=10)
        
        if event in (None, 'exit'): #ウィンドウのXボタンまたは”終了”ボタンを押したときの処理
            break
        
        if event == 'camera':  #「カメラ」ボタンが押された時の処理
            window['camera'].update('カメラ稼働中!!!', button_color=('red', 'white'))
            print("カメラが起動しました")
            cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する
            isOpened, orig_img = cap.read()
            if isOpened:  # 正常にフレームを読み込めたら
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                print("Frame size = " + str(orig_img.shape))
                frame_size = (width, height)
                # Vtuber画像の読み込み
                icon = load_image(IMAGE_PATH)
                # 表示用に画像を固定サイズに変更（大きい画像を入力した時に認識ボタンなどが埋もれないように）
                disp_img = scale_to_height(orig_img, display_size[1])
                # 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
                imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
                # ウィンドウへ表示
                window['-input_image-'].update(data=imgbytes)
                
                
                
            else:
                print("Cannot capture a frame image")

        if isOpened == 1:
            # ---- フレーム読み込み ----
            ret, frame = cap.read()
            if ret:  # 正常にフレームを読み込めたら
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #BGRからRGBに変換
                results = holistic.process(image) # 検出
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGBからBGRに変換
                
                if event == 'camera':  #「カメラ」ボタンが押された時の処理
                    print("試着したいネイルチップを選択してください")
                    
                
                if(values['nail'] == 'なし'):
                    movie_name = 'empty.mp4'
#                    print("選択されていません")
                    
                if(values['nail'] == 'nail1'):
                    movie_name = 'nail1.mp4'
#                    print("【nail1】を着用中…")
                     #ネイル設置右手
                    if results.right_hand_landmarks:
                        right_fitting(finger1_1,finger1_2)
                    #ネイル設置左手
                    if results.left_hand_landmarks:
                        left_fitting(finger1_1,finger1_2)
                        
                if(values['nail'] == 'nail2'):###
                    movie_name = 'nail2.mp4'###
#                    print("【nail2】を着用中…")####
                     #ネイル設置右手
                    if results.right_hand_landmarks:
                        right_fitting(finger2_1,finger2_2)###
                    #ネイル設置左手
                    if results.left_hand_landmarks:
                        left_fitting(finger2_1,finger2_2)###
                        
                if(values['nail'] == 'nail3'):###
                    movie_name = 'nail3.mp4'###
#                    print("【nail2】を着用中…")####
                     #ネイル設置右手
                    if results.right_hand_landmarks:
                        right_fitting(finger3_1,finger3_2)###
                    #ネイル設置左手
                    if results.left_hand_landmarks:
                        left_fitting(finger3_1,finger3_2)###

                    

                    


            
                                
#                if(values['landmark'] == 'あり'):
#                    # 顔
#                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
#                                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
#                                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
#                                             )
#
#                    # 右手
#                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
#                                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
#                                             )
#
#                    # 左手
#                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
#                                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
#                                             )
#
#                    # 姿勢
#                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
#                                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
#                                             )
                # 表示用に画像を固定サイズに変更
                disp_img = scale_to_height(image, display_size[1])
                # 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
                imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
                # ウィンドウへ表示
                window['-input_image-'].update(data=imgbytes)


        if event == 'save': #「保存」ボタンが押されたときの処理
            # VideoWriterでムービー書き出し処理を書く
            # 「保存」ボタンをもう一度押したら停止するようにする
            if isWriting == 0:
                window['save'].update('動画撮影中!!!', button_color=('red', 'white'))
                print("撮影開始 -> " + movie_name)
                isWriting = 1
                codec = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(movie_name, codec, 10.0, frame_size)
            elif isWriting == 1:
                window['save'].update('動画保存', button_color=('white', 'darkslategrey'))
                print("撮影終了")
                writer.release()
                isWriting = 0
        
        if isWriting == 1:  # if isOpened == 1:の中に入れても良い
            writer.write(image)

cap.release()
window.close()
