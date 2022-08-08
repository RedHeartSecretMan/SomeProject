import os
import sys
import cv2
import json
import numpy as np


def rect_pts_order(pts_2ds):
    """
    sort rectangle points by counterclockwise
    """

    cen_x, cen_y = np.mean(pts_2ds, axis=0)

    d2s = []
    for i in range(len(pts_2ds)):

        o_x = pts_2ds[i][0] - cen_x
        o_y = pts_2ds[i][1] - cen_y
        atan2 = np.arctan2(o_y, o_x)
        if atan2 < 0:
            atan2 += np.pi * 2
        d2s.append([pts_2ds[i], atan2])

    d2s = sorted(d2s, key=lambda x: x[1])
    order_2ds = np.array([x[0] for x in d2s])

    return order_2ds


def json_plot_label(file_path, save_path):
    with open(file_path, 'r') as f:
        label_dict = json.load(f)
    shapes = label_dict['shapes']

    points_1st = np.ndarray(shape=(0, 2), dtype=np.uint8)
    points_2st = np.ndarray(shape=(0, 2), dtype=np.uint8)
    vein_points = np.ndarray(shape=(0, 2), dtype=np.uint8)

    for k in range(len(shapes)):
        kind = shapes[k]['label']
        x = int(shapes[k]['points'][0][0])
        y = int(shapes[k]['points'][0][1])
        point = np.asarray([x, y])

        if kind == '1':
            points_1st = np.vstack((points_1st, point))
        elif kind == '2':
            points_2st = np.vstack((points_2st, point))
        elif kind == '3':
            vein_points = np.vstack((vein_points, point))
        else:
            raise ValueError(f'point label error: {kind}')

    image_path = file_path.replace(".json", ".png")
    cv_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)

    if len(cv_img.shape) == 3:
        label_img = np.zeros_like(cv_img)
    else:
        cv_img = np.expand_dims(cv_img, axis=-1).repeat(axis=-1, repeats=3)
        label_img = np.zeros_like(cv_img)

    # 填充
    if np.min(points_1st) < np.min(points_2st) and np.max(points_1st) > np.max(points_2st):
        cv2.fillPoly(label_img,
                     pts=[rect_pts_order(points_1st)],
                     color=(0, 0, 255))
        cv2.fillPoly(label_img,
                     pts=[rect_pts_order(points_2st)],
                     color=(0, 255, 0))
    elif np.min(points_1st) > np.min(points_2st) and np.max(points_1st) < np.max(points_2st):
        cv2.fillPoly(label_img,
                     pts=[rect_pts_order(points_2st)],
                     color=(0, 0, 255))
        cv2.fillPoly(label_img,
                     pts=[rect_pts_order(points_1st)],
                     color=(0, 255, 0))
    else:
        print("出现特殊的label")
        sys.exit()
    if vein_points.shape[0] != 0:
        cv2.fillPoly(label_img,
                     pts=[rect_pts_order(vein_points)],
                     color=(255, 0, 0))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imencode('.png', label_img)[1].tofile(os.path.join(save_path, label_dict['imagePath']))


if __name__ == "__main__":
    path = r"/Users/WangHao/工作/实习相关/微创卜算子医疗科技有限公司/陈嘉懿组/数据/王昊数据_0801/"

    # 通过DFS实现统一逻辑处理同一层次的文件对象
    for root, dirs, files in os.walk(path):
        if not dirs:
            keywords = ["label", "visual"]
            dir_name = root.split("/")[-1]
            keys = [key for key in keywords if key in dir_name]
            if not keys:
                ext = ".json"
                for file_name in files:
                    if ext in file_name:
                        file_path = os.path.join(root, file_name)
                        save_path = f"{root}_label"
                        json_plot_label(file_path, save_path)

    print("运行结束")
