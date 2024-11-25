import cv2
import numpy as np




def put_text_in_image(img, text, place = 'bottom-right', size = 1, color = 'black', thickness = 2, margin = 5, bordering = None):
    """
    画像に文字を入れる

    Parameters
    ----------
    img : np.array
        文字を入れたい画像イメージ（cv2形式）
    
    text : str
        入れたい文字列（英数字のみ）

    place : str
        'top' : 上部中央
        'top-left' : 上部左寄せ
        'top-right' : 上部右寄せ
        'center' : 中央
        'bottom' : 下部中央
        'bottom-left' : 下部左寄せ
        'bottom-right' : 下部右寄せ
    
    size : float
        フォントサイズ

    color : str
        文字色。['black', 'red', 'blue', 'green', 'orange', 'yellow', 'white']のいずれか

    thickness : int > 0
        フォントの太さ

    margin : int
        余白の大きさ

    bordering : dic
        縁取りしたい時に指定する
        'color' : 縁取りの色
        'thickness' : 縁取りの太さ

    return : np.array
        ウィンドウの画像イメージ（cv2形式）
        ウィンドウが見つからなかった場合はNoneを返す
    """
    # textをstrに変換
    text = str(text)

    # 文字の大きさを取得する
    # 黒バックの仮画像を生成
    height = int(50*size)
    width = int(len(text)*20*size + 10 + thickness - 1)
    blank = np.zeros((height, width, 3))
    
    # 黒バックに白で文字を入れる
    top, bottom, left, right = [0,0,0,0]
    cv2.putText(blank, text, (10,int(height/2)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = size, color = (255,255,255), thickness=thickness)
    # 単純な2次元配列に落とす
    temp = np.array([[int(j[0]) for j in i] for i in blank])
    # 文字がある上限を検索
    for i in range(height):
        if 255 in temp[i]:
            top = i
            break
    # 文字がある下限を検索
    for i in reversed(range(height)):
        if 255 in temp[i]:
            bottom = i
            break
    # 文字がある左限を検索
    for i in range(width):
        if 255 in temp[:,i]:
            left = i
            break
    # 文字がある右限を検索
    for i in reversed(range(width)):
        if 255 in temp[:,i]:
            right = i
            break
    
    # 描画開始位置の左限と実際のピクセル左限との差
    left_diff = left - 10

    # 描画開始位置の下限と実際のピクセル下限との差
    bottom_diff = bottom - int(height/2)

    # 文字を置く座標を算出
    height, width = img.shape[:2]
    text_height, text_width = [bottom-top, right-left]
    loc = (0,0)
    if place == 'top':
        loc = (width/2 - text_width/2, text_height + margin - size*7)
    elif place == 'top-left':
        loc = (margin, text_height + margin - size*7)
    elif place == 'top-right':
        loc = (width - text_width - margin + left_diff, text_height + margin - size*7)
    elif place == 'center':
        loc = (width/2 - text_width/2, height/2 - text_height/2)
    elif place == 'bottom':
        loc = (width/2 - text_width/2, height - margin - bottom_diff)
    elif place == 'bottom-left':
        loc = (margin, height - margin - bottom_diff)
    elif place == 'bottom-right':
        loc = (width - text_width - margin + left_diff, height - margin - bottom_diff)

    # 座標を整数値化
    loc = [int(i) for i in loc]

    # 色を指定
    color_BGR = {
        'black' : (0,0,0),
        'red' : (0,0,255),
        'blue' : (255,0,0),
        'green' : (0,255,0),
        'orange' : (0,127,255),
        'yellow' : (0,255,255),
        'white' : (255,255,255),
    }

    text_img = img.copy()

    # 縁取り
    if bordering:
        cv2.putText(text_img, text, loc, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = size, color = color_BGR[bordering['color']], thickness=thickness + bordering['thickness'])

    cv2.putText(text_img, text, loc, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = size, color = color_BGR[color], thickness=thickness)
    
    return text_img

