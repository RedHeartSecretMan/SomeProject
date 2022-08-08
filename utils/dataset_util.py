import os
import platform
import random
from glob import glob
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk


def batch_means_stdevs(
        path='./*/image*',
        image_size=(384, 320)
):
    imagePath = glob(path)
    imgs = []
    for i in range(len(imagePath)):
        if os.path.splitext(imagePath[i])[-1] in [".png", ".bmp", ".tif", ".jpg"]:
            imgTemp = cv2.imdecode(np.fromfile(imagePath[i], dtype=np.uint8), -1)
        elif os.path.splitext(imagePath[i])[-1] == ".gz":
            imgTemp = sitk.GetArrayFromImage(sitk.ReadImage(imagePath[i]))
        else:
            imgTemp = None
        assert len(imgTemp.shape) == 3
        imgs.append(cv2.resize(imgTemp, image_size, interpolation=cv2.INTER_CUBIC))

    imgs = np.concatenate(imgs, axis=-1)
    imgs = np.expand_dims(imgs, -1).repeat(3, -1).transpose((0, 1, 3, 2))
    means = []
    stdevs = []
    for j in range(imgs.shape[-2]):
        pixels = imgs[..., j, :].ravel() / 255
        means.append([np.mean(pixels)])
        stdevs.append([np.std(pixels)])

    return means, stdevs


def data_to_normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    assert len(mean.shape) == 2
    if len(img.shape) == 3:
        mean = mean[..., np.newaxis]
    elif len(img.shape) == 4:
        mean = mean[np.newaxis, ..., np.newaxis]
    mean *= max_pixel_value
    mean[mean == 0] = 1

    std = np.array(std, dtype=np.float32)
    assert len(std.shape) == 2
    if len(img.shape) == 3:
        std = std[..., np.newaxis]
    elif len(img.shape) == 4:
        std = std[np.newaxis, ..., np.newaxis]
    std *= max_pixel_value
    std[std == 0] = 1

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def data_to_one(inputs):
    data_min = np.nanmin(inputs)
    data_max = np.nanmax(inputs)
    outputs = (inputs - data_min) / (data_max - data_min)
    return outputs


def data_onehot(inputs, bg_channel=0):
    multilabel = np.zeros_like(inputs, np.int64)  # inputs size is [..., h, w, c]
    # Channel C is [BGR] - [静脉, 外膜, 内膜-外膜]
    # C0-B-蓝色 [静脉]/[背景]
    # C1-G-绿色 [内膜]
    # C2-R-红色 [内膜-外膜]
    if len(multilabel.shape) == 3:
        for i in range(inputs.shape[-1]):
            if i == bg_channel:
                multilabel[np.sum(inputs, -1) == 0, i] = 1
                multilabel[np.sum(inputs, -1) == 255, i] = 0
            else:
                multilabel[inputs[..., i] != 0, i] = 1
        assert np.all(np.sum(multilabel, -1) == 1)

    if len(multilabel.shape) == 4:
        for idx, _ in enumerate(multilabel):
            for i in range(inputs.shape[-1]):
                if i == bg_channel:
                    multilabel[idx, np.sum(inputs[idx], -1) == 0, i] = 1
                    multilabel[idx, np.sum(inputs[idx], -1) == 255, i] = 0
                else:
                    multilabel[idx, inputs[idx, ..., i] != 0, i] = 1
            assert np.all(np.sum(multilabel[idx], -1) == 1)

    return multilabel


class CsDataset(Dataset):
    def __init__(self, root_dir, data_file_list, img_size=None, num_class=None, transform=None):
        super().__init__()
        if img_size is None:
            img_size = [320, 384]
        self.root_dir = root_dir
        self.data_file_list = data_file_list
        self.img_size = img_size
        self.num_class = num_class
        self.transform = transform
        if transform == "norm":
            self.means, self.stds = batch_means_stdevs(path=f"{root_dir}*/image*", image_size=img_size)

    def __getitem__(self, index):
        file_dir = os.path.join(self.root_dir, self.data_file_list[index])
        image_dir = os.path.join(file_dir, 'image.png')
        label_dir = os.path.join(file_dir, 'label.png')

        # Channel is HWC
        image = cv2.imdecode(np.fromfile(image_dir, dtype=np.uint8), -1)
        label = cv2.imdecode(np.fromfile(label_dir, dtype=np.uint8), -1)
        image = cv2.resize(image, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)

        # Channel-HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        if self.transform == "norm":
            # 批标准化
            trans_image = data_to_normalize(image, self.means, self.stds)
        elif self.transform == "one":
            # 单归一化
            trans_image = data_to_one(image)
        else:
            trans_image = image

        # import matplotlib.pyplot as plot
        # plot.imshow(label[:, :, 0])
        # plot.show()
        # plot.imshow(label[:, :, 1])
        # plot.show()
        # plot.imshow(label[:, :, 2])
        # plot.show()

        # 生成onehot标签
        multilabel = data_onehot(label, bg_channel=0)

        # plot.imshow(multilabel[:, :, 0])
        # plot.show()
        # plot.imshow(multilabel[:, :, 1])
        # plot.show()
        # plot.imshow(multilabel[:, :, 2])
        # plot.show()

        multilabel = np.transpose(multilabel, (2, 0, 1))

        return trans_image, multilabel, {"image": image, "label": label}, self.data_file_list[index]

    def __len__(self):
        return len(self.data_file_list)


class CsDatasetVideo01(Dataset):
    def __init__(self, root_dir, data_file_list, img_size=None, num_class=None, transform=None):
        super().__init__()
        if img_size is None:
            img_size = [320, 384]
        self.root_dir = root_dir
        self.data_file_list = data_file_list
        self.img_size = img_size
        self.num_class = num_class
        self.transform = transform
        self.means, self.stds = batch_means_stdevs(path=f"{root_dir}*/*.png", image_size=img_size)

    def __getitem__(self, index):
        file_dir_image = os.path.join(self.root_dir, self.data_file_list[index])
        file_dir_label = file_dir_image.replace("image", "label")
        image_path_list = sorted(glob(f"{file_dir_image}/*.png"))
        label_path_list = sorted(glob(f"{file_dir_label}/*.png"))

        image = []
        for image_path in image_path_list:
            image_data = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            image_data = cv2.resize(image_data, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
            image_data = np.transpose(image_data, (2, 0, 1))
            image.append(image_data)
        image = np.stack(image, axis=0)  # [t, c, h, w]

        if self.transform == "norm":
            # 批标准化
            trans_image = data_to_normalize(image, self.means, self.stds)
        elif self.transform == "one":
            # 单归一化
            trans_image = data_to_one(image)
        else:
            trans_image = image

        label = []
        for label_path in label_path_list:
            label_data = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), -1)
            label_data = cv2.resize(label_data, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
            label.append(label_data)
        label = np.stack(label, axis=0)  # [t, h, w, c]

        multilabel = data_onehot(label, bg_channel=0)
        multilabel = np.transpose(multilabel, (0, 3, 1, 2))

        return trans_image, multilabel, {"image": image, "label": label}, self.data_file_list[index]

    def __len__(self):
        return len(self.data_file_list)


class CsDatasetVideo02(Dataset):
    def __init__(self, root_dir, data_file_list, img_size=None, num_class=None, transform=None):
        super().__init__()
        if img_size is None:
            img_size = [320, 384]
        self.root_dir = root_dir
        self.data_file_list = data_file_list
        self.img_size = img_size
        self.num_class = num_class
        self.transform = transform
        self.means, self.stds = batch_means_stdevs(path=f"{root_dir}/*.nii.gz", image_size=img_size)

    def __getitem__(self, index):
        file_dir_image = os.path.join(self.root_dir, self.data_file_list[index])
        file_dir_label = file_dir_image.replace("image", "label")

        image = sitk.GetArrayFromImage(sitk.ReadImage(file_dir_image))
        image = cv2.resize(image, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image, -1).repeat(3, -1)
        image = np.transpose(image, (2, 3, 0, 1))  # [t, c, h, w]

        if self.transform == "norm":
            # 批标准化
            trans_image = data_to_normalize(image, self.means, self.stds)
        elif self.transform == "one":
            # 单归一化
            trans_image = data_to_one(image)
        else:
            trans_image = image

        label = sitk.GetArrayFromImage(sitk.ReadImage(file_dir_label))
        label = cv2.resize(label, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        label = np.expand_dims(label, -1)
        label = np.transpose(label, (2, 0, 1, 3))  # [t, h, w, 1]

        multilabel = np.zeros_like(trans_image, np.int64)  # [t, c, h, w]
        for idx, _ in enumerate(multilabel):
            for k in range(trans_image.shape[1]):
                if k == 0:
                    multilabel[idx, k, label[idx, ..., 0] == 0] = 1
                elif k == 1:
                    multilabel[idx, k, label[idx, ..., 0] == 76] = 1
                elif k == 2:
                    multilabel[idx, k, label[idx, ..., 0] == 149] = 1
            assert np.all(np.sum(multilabel[idx], 0) == 1)

        return trans_image, multilabel, {"image": image, "label": label}, self.data_file_list[index]

    def __len__(self):
        return len(self.data_file_list)


if __name__ == '__main__':
    root_path = '/Users/WangHao/工作/实习相关/微创卜算子医疗科技有限公司/陈嘉懿组/数据/佳文数据_0722/短轴视频_0722/image_crop/'
    file_list = os.listdir(root_path)
    if platform.system() == 'Darwin':
        from imager_util import file_remove
        file_remove(file_list)
    file_list = sorted(file_list)

    seed = 51
    random.seed(seed)
    random.shuffle(file_list)

    train_list = file_list[:int(0.8 * len(file_list))]
    val_list = file_list[int(0.8 * len(file_list)):int(0.9 * len(file_list))]
    test_list = file_list[int(0.9 * len(file_list)):]

    batch_size = 1
    num_workers = 2
    train_set = CsDatasetVideo02(root_path, train_list, img_size=(256, 256), num_class=3, transform="norm")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    for _, data in enumerate(train_loader):
        print(len(data))
        pass
