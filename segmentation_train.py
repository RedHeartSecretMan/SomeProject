import os
import time
import platform
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from models.rvm import MattingNetwork
from utils.dataset_util import CsDataset, CsDatasetVideo01, CsDatasetVideo02
from utils.visualizer_util import Visualizers, analysis_profile, analysis_graph, analysis_interpreter
from utils.imager_util import file_remove, update_model_weight, visual_results, rate_stenosis


def train_process(md, tald, cri_d, cri_j, opt, vis, epo, dev):
    md.train()
    train_loss_list = []
    stenosis_rate_dict = {}
    for i, data in enumerate(tald):
        trans_image, multilabel, image_label, file_name = data
        trans_image, multilabel = trans_image.float().to(dev), multilabel.long().to(dev)

        opt.zero_grad()
        rec = [None] * 4
        outputs, *rec = md(trans_image, *rec)
        loss_d = cri_d(outputs[..., 1:, :, :], multilabel[..., 1:, :, :])
        loss_d.backward()
        opt.step()
        loss_j = cri_j(outputs[..., 1:, :, :], multilabel[..., 1:, :, :])
        train_loss_list.append([loss_d.item(), loss_j.item()])

        # img = image_label["image"]
        # img = img.detach().cpu().numpy()  # B3HW
        # pred = outputs.detach().cpu().numpy()
        # tru = multilabel.detach().cpu().numpy()
        # results = visual_results(img, pred, tru)
        # vis.vis_images('train', results, epo, i)
        #
        # pred_stenosis_rate, lab_stenosis_rate = rate_stenosis(pred, tru, 1)
        # stenosis_rate_dict[file_name] = [pred_stenosis_rate, lab_stenosis_rate]

    train_loss_list = np.mean(train_loss_list, axis=0)
    print(f'The [{epo}] train loss dice: {train_loss_list[0]} train loss iou: {train_loss_list[1]}')

    return train_loss_list, stenosis_rate_dict


def valid_process(md, vlld, cri_d, cri_j, vis, epo, dev):
    md.eval()
    valid_loss_list = []
    stenosis_rate_dict = {}
    with torch.no_grad():
        for i, data in enumerate(vlld):
            trans_image, multilabel, image_label, file_name = data
            trans_image, multilabel = trans_image.to(dev).float(), multilabel.to(dev)

            rec = [None] * 4
            outputs, *rec = md(trans_image, *rec)
            loss_d = cri_d(outputs[..., 1:, :, :], multilabel[..., 1:, :, :])
            loss_j = cri_j(outputs[..., 1:, :, :], multilabel[..., 1:, :, :])
            valid_loss_list.append([loss_d.item(), loss_j.item()])

            # img = image_label["image"]
            # img = img.detach().cpu().numpy()  # B3HW
            # pred = outputs.detach().cpu().numpy()
            # tru = multilabel.detach().cpu().numpy()

            # results = visual_results(img, pred, tru)
            # vis.vis_images('valid', results, epo, i)
            #
            # pred_stenosis_rate, lab_stenosis_rate = rate_stenosis(pred, tru, 1)
            # stenosis_rate_dict[file_name] = [pred_stenosis_rate, lab_stenosis_rate]

    valid_loss_list = np.mean(valid_loss_list, axis=0)
    print(f'The [{epo}] valid loss dice: {valid_loss_list[0]} valid loss iou: {valid_loss_list[1]}')

    return valid_loss_list, stenosis_rate_dict


if __name__ == '__main__':
    """????????????"""
    root_dir = '/Users/WangHao/??????/????????????/???????????????????????????????????????/????????????/??????/????????????_0722/????????????_0722/image_crop/'
    save_dir = './results/'

    """???????????????"""
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

    """????????????"""
    print(f'Creating dataset...')
    batch_size = 1
    num_workers = 2
    num_class = 3
    img_size = (256, 256)  # x, y
    train_set = CsDatasetVideo02(root_dir, train_list, img_size=img_size, num_class=num_class, transform="norm")
    valid_set = CsDatasetVideo02(root_dir, valid_list, img_size=img_size, num_class=num_class, transform="norm")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    """????????????"""
    device_num = "0"
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    print(f'Now using {device}')

    """????????????"""
    # encoder_name = 'timm-efficientnet-b0'
    # model = smp.Unet(in_channels=3, classes=num_class, activation='softmax2d', encoder_name=encoder_name)
    model = MattingNetwork(variant='mobilenetv3', refiner="null")
    model.to(device)

    """??????????????????"""
    weight_path = "/Users/WangHao/??????/????????????/???????????????????????????????????????/????????????/??????/??????/results/weight/cs_roi_MattingNetwork_0.164412_30_0.001_20220727031049.pth"
    try:
        model, updata_infos = update_model_weight(model, torch.load(weight_path))
        print('Last model params:{}, current model params:{}, matched model params:{}'.format(updata_infos[0],
                                                                                        updata_infos[1],
                                                                                        updata_infos[2]))
    except FileNotFoundError as e:
        print('Can not load last weight from {} for model {}'.format(weight_path,
                                                                     type(model).__name__))
        print('The parameters of model is initialized by method in model set')

    """???????????????"""
    visual = Visualizers(f"{save_dir}/log")

    """????????????"""
    criterion_d = smp.utils.losses.DiceLoss()
    criterion_j = smp.utils.losses.JaccardLoss()
    criterion_c = smp.utils.losses.CrossEntropyLoss()
    best_loss = float('Inf')

    """?????????"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    """???????????????"""
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    start = time.time()
    """???????????????????????????"""
    max_epoch = 35
    for epoch in range(1, max_epoch+1):
        # ????????????
        train_loss, train_stenosis_rate = train_process(model, train_loader, criterion_d, criterion_j, optimizer, visual, epoch,
                                                        device)

        # ????????????
        valid_loss, valid_stenosis_rate = valid_process(model, valid_loader, criterion_d, criterion_j, visual, epoch, device)

        # ????????????
        visual.vis_write('train', {
            'train_loss_dice': train_loss[0],
            'train_loss_iou': train_loss[1],
        }, epoch)
        visual.vis_write('valid', {
            'valid_loss_dice': valid_loss[0],
            'valid_loss_iou': valid_loss[1],
        }, epoch)

        # ????????????
        if valid_loss[0] < best_loss:
            best_loss = valid_loss[0]
            lr = optimizer.param_groups[0]['lr']
            timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
            # ????????????-[????????????]_[????????????]_[????????????]_[??????]_[?????????]_[?????????]
            torch.save(model.state_dict(), f'{save_dir}/weight/cs_roi_{type(model).__name__}_{best_loss:.6f}_{epoch}_{lr}'
                                           f'_{timestamp}.pth')
            # torch.save(model, f'{save_dir}/cs_roi_{encoder_name}_{best_loss:.4f}_{epoch}_{lr}_{timestamp}.pth')
            print(f'Current best rvm at epoch {epoch}')

        # ???????????????
        scheduler.step(valid_loss[0])

    visual.close_vis()
    print('Finished training! Total cost time: ', time.time() - start)
