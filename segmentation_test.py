import os
import platform
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils.dataset_util import CsDataset
from utils.imager_util import file_remove, update_model_weight, visual_results, save_result, rate_stenosis
from utils.visualizer_util import Visualizers
from utils.losses_util import loss_functions


def test_process(md, root_path, tsld, vis, dev):
    md.eval()
    dice_list = []
    ac_list = []
    pre_list = []
    r_list = []
    f1_list = []
    iou_list = []
    ap_05_list = []
    ap_07_list = []
    stenosis_rate_dict = {}
    with torch.no_grad():
        for idx, data in enumerate(tsld):
            trans_image, multilabel, image_label, file_name = data

            trans_image = trans_image.to(dev).float()
            outputs = md(trans_image)

            # 计算指标
            dice_all = loss_functions("dice")(outputs, multilabel)
            dice_list.append(dice_all)
            ac_all = loss_functions("ac")(outputs, multilabel)
            ac_list.append(ac_all)
            pre_all = loss_functions("pre")(outputs, multilabel)
            pre_list.append(pre_all)
            r_all = loss_functions("r")(outputs, multilabel)
            r_list.append(r_all)
            f1_all = loss_functions("f1")(outputs, multilabel)
            f1_list.append(f1_all)
            iou_all = loss_functions("iou")(outputs, multilabel)
            iou_list.append(iou_all)
            ap_05_all = loss_functions("ap@0.5")(outputs, multilabel, eps=1e-7)
            ap_05_list.append(ap_05_all)
            ap_07_all = loss_functions("ap@0.7")(outputs, multilabel, eps=1e-7)
            ap_07_list.append(ap_07_all)

            # dice_bg = dice(outputs[:, 0, :, :], multilabel[:, 0, :, :])
            # dice_nm = dice(outputs[:, 1, :, :], multilabel[:, 1, :, :])
            # dice_nm_wm = dice(outputs[:, 2, :, :], multilabel[:, 2, :, :])
            # visual.vis_write('test_dice', {'dice_all': dice_all.item(),
            #                                'dice_bg': dice_bg.item(),
            #                                'dice_nm': dice_nm.item(),
            #                                'dice_nm_wm': dice_nm_wm.item(),
            #                                }, idx)
            #
            # print(dice_all.item(), dice_bg.item(), dice_nm.item(), dice_nm_wm.item())
            # dice_list.append([dice_all.item(), dice_bg.item(), dice_nm.item(), dice_nm_wm.item()])
            #
            #
            # iou_bg = iou(outputs[:, 0, :, :], multilabel[:, 0, :, :])
            # iou_nm = iou(outputs[:, 1, :, :], multilabel[:, 1, :, :])
            # iou_nm_wm = iou(outputs[:, 2, :, :], multilabel[:, 2, :, :])
            # vis.vis_write('test_iou', {'iou_all': iou_all.item(),
            #                            'iou_bg': iou_bg.item(),
            #                            'iou_nm': iou_nm.item(),
            #                            'iou_nm_wm': iou_nm_wm.item(),
            #                            }, idx)
            #
            # print(iou_all.item(), iou_bg.item(), iou_nm.item(), iou_nm_wm.item())
            # iou_list.append([iou_all.item(), iou_bg.item(), iou_nm.item(), iou_nm_wm.item()])

            # pred = outputs.cpu().numpy()
            # tru = multilabel.cpu().numpy()
            # pred_stenosis_rate, lab_stenosis_rate = rate_stenosis(pred, tru, 1)
            # stenosis_rate_dict[file_name] = [pred_stenosis_rate, lab_stenosis_rate]
            #
            # # 保存结果
            # img = image_label["image"]
            # img = img.detach().cpu().numpy()  # B3HW
            # pred = outputs.detach().cpu().numpy()
            # tru = multilabel.detach().cpu().numpy()
            # results = visual_results(img, pred, tru)
            # vis.vis_images(f'test_{dice_nm_wm.item():.6f}', results, idx)
            #
            # save_result(root_path, file_name, img, pred, tru, idx)

        test_loss_dict = dict({"dice": np.array(dice_list).mean(axis=0),
                               "ae": np.array(ac_list).mean(axis=0),
                               "pre": np.array(pre_list).mean(axis=0),
                               "r": np.array(r_list).mean(axis=0),
                               "f1": np.array(f1_list).mean(axis=0),
                               "iou": np.array(iou_list).mean(axis=0),
                               "ap@0.5": np.array(ap_05_list).mean(axis=0),
                               "ap@0.7": np.array(ap_07_list).mean(axis=0),
                               })
    print(test_loss_dict)

    return test_loss_dict, stenosis_rate_dict


if __name__ == '__main__':
    """路径设置"""
    root_dir = '/Users/WangHao/工作/实习相关/微创卜算子医疗科技有限公司/陈嘉懿组/数据/嘉懿数据_0801/原始390划出的测试集116/'
    save_dir = './results/'

    """数据集划分"""
    # train:test:valid -> 6:3:1
    train_list = []
    valid_list = []
    test_list = []
    folder_list = os.listdir(root_dir)
    if platform.system() == 'Darwin':
        file_remove(folder_list)
    file_list = sorted(folder_list)

    seed = 51
    random.seed(seed)
    random.shuffle(file_list)

    _train_list = file_list[:int(0.6 * len(file_list))]
    _valid_list = file_list[int(0.6 * len(file_list)):int(0.7 * len(file_list))]
    _test_list = file_list[int(0.7 * len(file_list)):]

    train_list.extend(_train_list)
    valid_list.extend(_valid_list)
    test_list.extend(_test_list)

    if True:
        test_list = file_list

    """数据加载"""
    print(f'Creating dataset...')
    batch_size = 1
    num_workers = 2
    num_class = 3
    img_size = (384, 320)  # x, y
    test_set = CsDataset(root_dir, test_list, img_size=img_size, num_class=num_class, transform="one")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    """硬件配置"""
    device_num = "0"
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    print(f'Now using {device}')

    """模型配置"""
    encoder_name = 'timm-efficientnet-b0'
    model = smp.Unet(in_channels=3, classes=num_class, activation='softmax2d', encoder_name=encoder_name)
    model.to(device)

    """模型权重载入"""
    weight_path = "/Users/WangHao/工作/实习相关/微创卜算子医疗科技有限公司/陈嘉懿组/代码/横切/results/weight/efficientnet-b0_best.pt"
    try:
        model, updata_infos = update_model_weight(model, torch.load(weight_path))
        print('Last model params:{}, current model params:{}, matched params:{}'.format(updata_infos[0],
                                                                                        updata_infos[1],
                                                                                        updata_infos[2]))
    except FileNotFoundError as e:
        print('Can not load last weight from {} for model {}'.format(weight_path, type(model).__name__))
        print('The parameters of model is initialized by method in model set')
    model = torch.load("/Users/WangHao/工作/实习相关/微创卜算子医疗科技有限公司/陈嘉懿组/代码/横切/results/trained_model/efficientnet-b0_best.pt",
                       map_location=torch.device('cpu'))

    """可视化记录"""
    visual = Visualizers(f"{save_dir}/log")

    """测试模型"""
    test_loss, test_stenosis_rate = test_process(model, root_dir, test_loader, visual, device)
    visual.close_vis()
