import sys
import av
import os
import pims
import platform
import torch
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from imager_util import file_remove


class VideoReader(Dataset):
    def __init__(self, path, transformers=None):
        self.video = pims.PyAVVideoReader(path)
        self.rate = self.video.frame_rate
        self.transformers = transformers

    @property
    def frame_rate(self):
        return self.rate

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        frame = self.video[idx]
        frame = Image.fromarray(np.asarray(frame))
        if self.transformers is not None:
            frame = self.transformers(frame)
        return frame


class VideoWriter:
    def __init__(self, path, frame_rate, bit_rate=1000000):
        self.container = av.open(path, mode='w')
        self.stream = self.container.add_stream('h264', rate=round(frame_rate))
        self.stream.pix_fmt = 'yuv420p'
        self.stream.bit_rate = bit_rate

    def write(self, frames):
        # frames: [T, C, H, W]
        self.stream.width = frames.size(3)
        self.stream.height = frames.size(2)
        if frames.size(1) == 1:
            frames = frames.repeat(1, 3, 1, 1)  # convert grayscale to RGB
        frames = frames.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
        for t in range(frames.shape[0]):
            frame = frames[t]
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            self.container.mux(self.stream.encode(frame))

    def close(self):
        self.container.mux(self.stream.encode())
        self.container.close()


class ImageSequenceReader(Dataset):
    def __init__(self, path, extension, transformers=None):
        self.path = path
        self.extension = extension
        self.files = sorted(glob(f"{path}/*.{extension}"))
        self.transformers = transformers

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(os.path.join(self.files[idx])) as img:
            img.load()
        if self.transformers is not None:
            return self.transformers(img)
        return img


class ImageSequenceWriter:
    def __init__(self, path, extension='jpg'):
        self.path = path
        self.extension = extension
        self.counter = 0
        os.makedirs(path, exist_ok=True)

    def write(self, frames):
        # frames: [T, C, H, W]
        if len(frames) == 4:
            for t in range(frames.shape[0]):
                to_pil_image(frames[t]).save(os.path.join(
                    self.path, str(self.counter).zfill(4) + '.' + self.extension))
                self.counter += 1
        # frames: [C, H, W]/[H, W]
        elif 2 <= len(frames) < 4:
            to_pil_image(frames).save(os.path.join(
                self.path, str(self.counter).zfill(4) + '.' + self.extension))
            self.counter += 1
        else:
            print("Data format error")
            sys.exit()

    def close(self):
        pass


if __name__ == '__main__':
    mode = "read_image_sequence"
    if mode == "read_video":
        # 读取路径
        video_path = "/Users/WangHao/Desktop/TODO/Data/颈总动脉视频截取/*.avi"
        video_list = glob(video_path)
        if platform.system() == 'Darwin':
            file_remove(video_list)
        video_list = sorted(video_list)

        # 读取参数
        input_resize = None
        if input_resize is not None:
            transform = transforms.Compose([
                transforms.Resize(input_resize[::-1]),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.ToTensor()

        for path_name in video_list:
            # 读取数据
            source = VideoReader(path_name, transform)

            # 保存路径
            save_path = path_name.replace('.avi', '')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            # 保存参数
            output_type = "image_sequence"
            output_video_mbps = None
            output_composition = save_path
            output_foreground = None
            output_alpha = None
            assert any([output_composition, output_foreground, output_alpha]), 'Must provide at least one output.'

            if (output_composition is not None) and (output_type == 'video'):
                bgr = torch.tensor([120, 255, 155]).div(255).view(1, 1, 3, 1, 1)

            if output_type == "video":
                f_rate = source.frame_rate if isinstance(source, VideoReader) else 30
                output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
                if output_composition is not None:
                    writer_com = VideoWriter(
                        path=output_composition,
                        frame_rate=f_rate,
                        bit_rate=int(output_video_mbps * 1000000))
                if output_foreground is not None:
                    writer_fgr = VideoWriter(
                        path=output_foreground,
                        frame_rate=f_rate,
                        bit_rate=int(output_video_mbps * 1000000))
                if output_alpha is not None:
                    writer_pha = VideoWriter(
                        path=output_alpha,
                        frame_rate=f_rate,
                        bit_rate=int(output_video_mbps * 1000000))
            elif output_type == "image_sequence":
                if output_composition is not None:
                    writer_com = ImageSequenceWriter(output_composition, 'png')
                    for data in source:
                        writer_com.write(data)
                if output_foreground is not None:
                    writer_fgr = ImageSequenceWriter(output_foreground, 'png')
                    for data in source:
                        writer_fgr.write(data)
                if output_alpha is not None:
                    writer_pha = ImageSequenceWriter(output_alpha, 'png')
                    for data in source:
                        writer_pha.write(data)
    elif mode == "read_image_sequence":
        # 读取路径
        image_sequence_path = "/Users/WangHao/工作/实习相关/微创卜算子医疗科技有限公司/陈嘉懿组/数据/佳文数据_0722/短轴视频_0722"
        image_sequence_list = os.listdir(image_sequence_path)
        if platform.system() == 'Darwin':
            file_remove(image_sequence_list)
        image_sequence_list = sorted(image_sequence_list)

        # 读取参数
        input_resize = None
        if input_resize is not None:
            transform = transforms.Compose([
                transforms.Resize(input_resize[::-1]),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.ToTensor()

        for path_name in image_sequence_list:
            # 读取数据
            ext = "png"
            source = ImageSequenceReader(os.path.join(image_sequence_path, path_name), ext, transform)

            # 保存路径
            save_path = path_name
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            # 保存参数
            output_type = "npy"
            output_video_mbps = None
            output_composition = save_path  # com = fgr * pha + bgr * (1 - pha)
            output_foreground = None  # fgr
            output_alpha = None  # pha
            assert any([output_composition, output_foreground, output_alpha]), 'Must provide at least one output.'

            if (output_composition is not None) and (output_type == 'video'):
                bgr = torch.tensor([120, 255, 155]).div(255).view(1, 1, 3, 1, 1)

            if output_type == "video":
                f_rate = source.frame_rate if isinstance(source, VideoReader) else 30
                output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
                if output_composition is not None:
                    writer_com = VideoWriter(
                        path=output_composition,
                        frame_rate=f_rate,
                        bit_rate=int(output_video_mbps * 1000000))
                    com = []
                    for data in source:
                        com.append(data)
                    writer_com.write(torch.tensor(com))
                if output_foreground is not None:
                    writer_fgr = VideoWriter(
                        path=output_foreground,
                        frame_rate=f_rate,
                        bit_rate=int(output_video_mbps * 1000000))
                if output_alpha is not None:
                    writer_pha = VideoWriter(
                        path=output_alpha,
                        frame_rate=f_rate,
                        bit_rate=int(output_video_mbps * 1000000))
            elif output_type == "image_sequence":
                if output_composition is not None:
                    writer_com = ImageSequenceWriter(output_composition, 'png')
                    for data in source:
                        writer_com.write(data)
                if output_alpha is not None:
                    writer_pha = ImageSequenceWriter(output_alpha, 'png')
                    for data in source:
                        writer_pha.write(data)
                if output_foreground is not None:
                    writer_fgr = ImageSequenceWriter(output_foreground, 'png')
                    for data in source:
                        writer_fgr.write(data)
            elif output_type == "npy":
                com = []
                for data in source:
                    data = (data.cpu().numpy() * 255).astype(np.uint8)
                    com.append(data)
                np.save(os.path.join(image_sequence_path, path_name, f"{path_name}"), np.stack(com))








