import sys

import numpy as np
import cv2
import os
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage import measure, morphology
from skimage.feature import peak_local_max


def remove_isolate(inputs, threshold_area=0.5):
    """
    inputs: x*y*3
    """
    mask = np.zeros((inputs.shape[0], inputs.shape[1]), dtype=np.uint8)
    mask[inputs > 0] = 1
    mask = morphology.remove_small_objects(mask.astype(np.bool8), np.sum(mask) * threshold_area, connectivity=8).astype(np.uint8)
    outputs = mask * inputs

    return outputs


def whether_to_post_process(mask):
    """
    mask: x*y*3
    """
    lumen_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    lumen_mask[mask > 1] = 1
    image_label, nums = measure.label(lumen_mask, neighbors=8, return_num=True)
    if nums <= 1:
        return mask
    else:
        distance = ndi.distance_transform_edt(lumen_mask)
        coords = peak_local_max(distance, footprint=np.ones((20, 20)), labels=lumen_mask)
        distance[distance < 0] = 0
        res_mask = np.zeros_like(distance, dtype=bool)
        res_mask[tuple(coords.T)] = True
        s = ndi.generate_binary_structure(2, 2)
        markers, _ = ndi.label(res_mask, s)
        labels = watershed(-distance, markers, mask=mask.copy())
        circleDegree = []
        for i in range(1, labels.max() + 1):
            circleDegree.append(circle_degree_evaluate((labels == i).astype(np.uint8)))
        return (labels == (np.argmax(circleDegree) + 1)).astype(np.uint8) * mask


def circle_degree_evaluate(label_image):
    """
    label_image:binary image x*y
    """
    contours, _ = cv2.findContours(np.array(label_image, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a = cv2.contourArea(contours[0]) * 4 * np.pi
    b = cv2.arcLength(contours[0], True) ** 2
    if b == 0:
        return 0
    return a / b


def file_remove(file_list):
    try:
        file_list.remove('.DS_Store')
    except ValueError:
        pass
    try:
        file_list.remove('._.DS_Store')
    except ValueError:
        pass


def update_model_weight(model, last_weight_dict):
    cur_weight_dict = model.state_dict()
    updated_weight_dict = {k: v for k, v in last_weight_dict.items() if k in cur_weight_dict}
    cur_weight_dict.update(updated_weight_dict)
    model.load_state_dict(cur_weight_dict)

    last_params = len(last_weight_dict)
    cur_params = len(cur_weight_dict)
    matched_params = len(updated_weight_dict)

    infos = [last_params, cur_params, matched_params]
    return model, infos


def visual_results(imgs, preds, trus):
    """
    params: imgs, preds, trus size is [B, C, H, W]
    """
    assert all([imgs.shape[1] == 3, preds.shape[1] == 3, trus.shape[1] == 3])

    cv_pred = np.zeros_like(imgs)
    cv_pred[preds >= 0.5] = 255

    cv_tru = np.zeros_like(imgs)
    cv_tru[trus >= 0.5] = 255

    img_pred = cv2.addWeighted(imgs, 0.7, cv_pred, 0.3, 0)
    img_tru = cv2.addWeighted(imgs, 0.7, cv_tru, 0.3, 0)

    outputs = np.concatenate((img_pred, img_tru), axis=0)

    return outputs


def save_result(root_name, data_file, imgs, preds, trus, index):
    cv_pred = np.zeros_like(imgs)
    cv_pred[preds == 1] = 255

    cv_tru = np.zeros_like(imgs)
    cv_tru[trus == 1] = 255

    img_pred = cv2.addWeighted(imgs, 0.7, cv_pred, 0.3, 0)
    img_tru = cv2.addWeighted(imgs, 0.7, cv_tru, 0.3, 0)

    img_pred = img_pred.transpose((0, 2, 3, 1))
    img_tru = img_tru.transpose((0, 2, 3, 1))

    for idx, data_name in enumerate(data_file):
        pred_save_path = os.path.join(root_name, data_name, f'img_pred{index}.png')
        cv2.imwrite(pred_save_path, img_pred[idx])
        tru_save_path = os.path.join(root_name, data_name, f'img_tru{index}.png')
        cv2.imwrite(tru_save_path, img_tru[idx])


def minimum_external_circle(img, mask_path=None):
    if mask_path is not None:
        img = cv2.imread(mask_path)

    assert len(img.shape) == 3, "Input data size error"

    # import matplotlib.pyplot as plot
    # plot.figure(1)
    # plot.imshow(img)
    # plot.figure(2)
    # plot.imshow(img[..., 1])
    # plot.figure(3)
    # plot.imshow(img[..., 2])
    # plot.show()

    center_list = []
    radius_list = []
    if 0 < img.shape[-1] <= 4:
        img_inner = img.transpose((2, 0, 1))
    else:
        img_inner = img
    for idx, data in enumerate(img_inner):
        contours, _ = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if not contours:
            print(f"Exist data frame {idx} is none information")
            sys.exit()
        else:
            cnt = contours[0]
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center_list.append((int(x), int(y)))
            radius_list.append(radius)

            # plot.figure(0)
            # plot.imshow(data)
            # cv2.circle(data, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            # cv2.circle(data, (int(x), int(y)), 1, (255, 0, 0), 2)
            # plot.figure(1)
            # plot.imshow(data)
            # plot.show()

    return center_list, radius_list


def rate_stenosis(preds, labs, green=1):
    """
    params: preds, labs size is [B, C, H, W]
    return: every batch stenosis_rate
    """
    assert all([preds.shape[1] == 3, labs.shape[1] == 3])
    cv_pred = np.zeros_like(preds, dtype=np.uint8)
    cv_pred[preds >= 0.5] = 1
    cv_pred[preds < 0.5] = 0

    _, radius = minimum_external_circle(cv_pred[:, green, ...])
    pred_green_area = np.sum(cv_pred[:, green, ...], axis=(-2, -1), keepdims=True)
    pred_circle_area = np.pi*(np.expand_dims(np.stack(radius), axis=(-2, -1)) / 2)**2
    pred_stenosis_rate = 1 - pred_green_area / pred_circle_area

    cv_lab = np.zeros_like(labs, dtype=np.uint8)
    cv_lab[labs >= 0.5] = 1
    cv_lab[labs < 0.5] = 0

    lab_green_area = np.sum(cv_lab[:, green, ...], axis=(-2, -1), keepdims=True)
    _, radius = minimum_external_circle(cv_lab[:, green, ...])
    lab_circle_area = np.pi*(np.expand_dims(np.stack(radius), axis=(-2, -1)) / 2)**2
    lab_stenosis_rate = 1 - lab_green_area / lab_circle_area

    return pred_stenosis_rate, lab_stenosis_rate


if __name__ == "__main__":
    path = "/Users/WangHao/工作/实习相关/微创卜算子医疗科技有限公司/陈嘉懿组/数据/dataset_0714/HSF杭肿正常二维横切_63/label.png"
    minimum_external_circle(img=np.random.rand(2, 3, 128, 128), mask_path=path)
