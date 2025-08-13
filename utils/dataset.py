import os
import decord
import numpy as np
import random
import json
import torchvision
import torchvision.transforms as T
import torch

from glob import glob
from PIL import Image
from itertools import islice
from pathlib import Path
from .bucketing import sensible_buckets

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange, repeat

import requests
import io

def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids

    return prompt_ids

def read_caption_file(caption_file):
        with open(caption_file, 'r', encoding="utf8") as t:
            return t.read()

def get_text_prompt(
        text_prompt: str = '', 
        fallback_prompt: str= '',
        file_path:str = '', 
        ext_types=['.mp4'],
        use_caption=False
    ):
    try:
        if use_caption:
            if len(text_prompt) > 1: return text_prompt
            caption_file = ''
            # Use caption on per-video basis (One caption PER video)
            for ext in ext_types:
                maybe_file = file_path.replace(ext, '.txt')
                if maybe_file.endswith(ext_types): continue
                if os.path.exists(maybe_file): 
                    caption_file = maybe_file
                    break

            if os.path.exists(caption_file):
                return read_caption_file(caption_file)
            
            # Return fallback prompt if no conditions are met.
            return fallback_prompt

        return text_prompt
    except:
        print(f"Couldn't read prompt caption for {file_path}. Using fallback.")
        return fallback_prompt

# 按原始lvdm的方式，从第一帧起取连续k帧
# def get_video_frames(vr, start_idx, sample_rate=1, max_frames=24):
#     #! 获取video指定位置的frames
#     max_range = len(vr)
#     frame_number = sorted((0, start_idx, max_range))[1]

#     frame_range = range(frame_number, max_range, sample_rate)
#     frame_range_indices = list(frame_range)[:max_frames]
#     import pdb; pdb.set_trace()
#     return frame_range_indices

# 按原control-a-video+blip caption的方式，从中间帧n起取[n,n+k]连续k帧
def get_video_frames(vr, start_idx, sample_rate=1, max_frames=24):
    #! 获取video指定位置的frames
    max_range = len(vr)
    frame_number = sorted((0, start_idx, max_range))[1]

    frame_range = range(frame_number, max_range, sample_rate)
    # frame_range_indices = list(frame_range)[:max_frames]
    # import pdb; pdb.set_trace()

    real_max_range = len(frame_range)
    # early_frame_num = max_frames//2
    later_frame_num = max_frames
    init_frame_ind = real_max_range//2 + 1

    # frame_range_indices = list(frame_range)[init_frame_ind - early_frame_num : init_frame_ind + later_frame_num]    # 改为从中间帧n起取[n-k/2,n+k/2]连续k帧
    frame_range_indices = list(frame_range)[init_frame_ind : init_frame_ind + later_frame_num]    # 改为从中间帧n起取[n,n+k]连续k帧
    
    # import pdb; pdb.set_trace()
    return frame_range_indices


def process_video(vid_path, use_bucketing, w, h, get_frame_buckets, get_frame_batch):
    if use_bucketing:
        vr = decord.VideoReader(vid_path)
        # resize = get_frame_buckets(vr)
        center_crop, resize = get_frame_buckets(vr)
        # video = get_frame_batch(vr, resize=resize)
        video = get_frame_batch(vr, crop=center_crop, resize=resize)
    else:
        vr = decord.VideoReader(vid_path, width=w, height=h)
        video = get_frame_batch(vr) # no resize

    # import pdb; pdb.set_trace()
    return video, vr


# https://github.com/ExponentialML/Video-BLIP2-Preprocessor
class VideoJsonDataset(Dataset):
    def __init__(
            self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 10,
            sample_start_idx: int = 1,
            frame_step: int = 1,
            path: str = None,
            json_path: str ="",
            json_data = None,
            vid_data_key: str = "video_path",
            preprocessed: bool = False,
            use_bucketing: bool = False,
            data_mode: str = "train",   #! 用于判断是取train还是val/test数据
            fetch_num = 300000,
            **kwargs
    ):
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.use_bucketing = use_bucketing
        self.tokenizer = tokenizer
        self.preprocessed = preprocessed
        
        self.vid_data_key = vid_data_key

        if isinstance(json_path, str):
            self.train_data = self.load_from_json(json_path, json_data)
        else:
            #! 此处修改为可以load多个video annots文件
            json_path = list(json_path)
            self.train_data = []
            for sub_json_path in json_path:
                self.train_data_sub = self.load_from_json(sub_json_path, json_data)         
                self.train_data.extend(self.train_data_sub)
        
        # 发现数据量过大时会报错，此处限制数量
        if len(self.train_data) > 300000:
            self.train_data = self.train_data[:fetch_num]   # 只取一部分数据

        #! 用于排查破损视频影响
        # self.train_data_aa = []
        # self.train_data_aa.extend(self.train_data[:3])
        # for alldata in self.train_data:
        #     dir_video = alldata['video_path'].split("/")[-2]
        #     if dir_video == "4761950":
        #         self.train_data_aa.append(alldata)
        # self.train_data = self.train_data_aa

        # import pdb; pdb.set_trace()
        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.frame_step = frame_step

        self.video_dir = path      #! set where to read video from 
        # import pdb; pdb.set_trace() 
        self.data_mode = data_mode  #! 用于判断是取train还是val/test数据
        self.fps = 8

    def build_json(self, json_data):
        extended_data = []
        for data in json_data['data']:
            for nested_data in data['data']:
                self.build_json_dict(
                    data, 
                    nested_data, 
                    extended_data
                )
        json_data = extended_data
        return json_data

    def build_json_dict(self, data, nested_data, extended_data):
        clip_path = nested_data['clip_path'] if 'clip_path' in nested_data else None
        
        extended_data.append({
            self.vid_data_key: data[self.vid_data_key],
            'frame_index': nested_data['frame_index'],
            'prompt': nested_data['prompt'],
            'clip_path': clip_path
        })
        
    def load_from_json(self, path, json_data):
        try:
            with open(path) as jpath:
                print(f"Loading JSON from {path}")
                json_data = json.load(jpath)

                return self.build_json(json_data)

        except:
            self.train_data = []
            print("Non-existant JSON path. Skipping.")
            
    def validate_json(self, base_path, path):
        return os.path.exists(f"{base_path}/{path}")

    def get_frame_range(self, vr):
        # import pdb; pdb.set_trace()
        return get_video_frames(
            vr, 
            self.sample_start_idx, 
            self.frame_step, 
            self.n_sample_frames
        )
    
    def get_vid_idx(self, vr, vid_data=None):
        frames = self.n_sample_frames
        if vid_data is not None:
            idx = vid_data['frame_index']
        else:
            idx = self.sample_start_idx

        return idx

    def get_frame_buckets(self, vr):
        h, w, _  = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w) 

        #! 这里需要加center-crop    
        crop_length = min(h, w)  # 找最短的边
        center_crop = T.transforms.CenterCrop((crop_length, crop_length))  # crop

        resize = T.transforms.Resize((self.height, self.width), antialias=True)   # resize
        # import pdb; pdb.set_trace()
        # return resize
        return center_crop, resize
        
    def get_frame_batch(self, vr, crop=None, resize=None):
        #! 根据指定fps抽帧，不用设置frame_step(默认为1)
        n_sample_frames = self.n_sample_frames
        native_fps = vr.get_avg_fps()   # vila-fps=30
        every_nth_frame = max(1, round(native_fps / self.fps))
        every_nth_frame = min(len(vr), every_nth_frame)
        effective_length = len(vr) // every_nth_frame
        if effective_length < n_sample_frames:
            n_sample_frames = effective_length
        init_idx = random.randint(0, (effective_length - n_sample_frames))  # 可避免取帧的index越界
        # init_idx = effective_length // 2    # 配合blip，从clip的中间帧开始
        # init_idx = 0    
        
        idxs = every_nth_frame * np.arange(init_idx, init_idx + n_sample_frames)
        frames = vr.get_batch(idxs)
        # import pdb; pdb.set_trace()

        # frame_range = self.get_frame_range(vr)
        # frames = vr.get_batch(frame_range)
        video = rearrange(frames, "f h w c -> f c h w")

        if crop is not None: video = crop(video)    # center-crop
        if resize is not None: video = resize(video)
        return video

    def process_video_wrapper(self, vid_path):
        # import pdb; pdb.set_trace()
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        
        return video, vr 

    def train_data_batch(self, index):
        # 用于取train数据

        # If we are training on individual clips.
        if 'clip_path' in self.train_data[index] and \
            self.train_data[index]['clip_path'] is not None:
            vid_data = self.train_data[index]
                  
            # 1) hdVila100M 数据
            if "hdVila100M_" in vid_data['clip_path']:
                clip_path = vid_data['clip_path'] 
                clip_name = clip_path.split("/")[-1]     # 为载入conf文件中的video_dir路径，只用video path中包含video名字的部分
                # a) load local data
                # clip_path = os.path.join(self.video_dir, clip_name)      
                # b) load mounted-bos data
                # example: /root/ttv_data/hdVila100M/slice_video/2bG33NemyYk/2bG33NemyYk.12.mp4
                clip_name = clip_name.split("hdVila100M_")[-1]  
                video_dir_name = clip_name.split(".")[0]        # 取clip_name, 去掉suffix
                clip_path = os.path.join(self.video_dir, video_dir_name, clip_name)    
                
            # 2) pexels数据
            elif "/pexels/" in vid_data['clip_path']:
                clip_path = vid_data['clip_path'] 
                # if "/root/ttv_data/." not in clip_path:
                if "/root/ttv_data/" not in clip_path:
                    clip_path = self.video_dir + clip_path
            
            # import pdb; pdb.set_trace()
            # Get video prompt      
            prompt = vid_data['prompt']     
            video, _ = self.process_video_wrapper(clip_path)    
            prompt_ids = prompt_ids = get_prompt_ids(prompt, self.tokenizer)    

            return video, prompt, prompt_ids    

        # Assign train data
        train_data = self.train_data[index]
        # import pdb; pdb.set_trace()
        
        # Get the frame of the current index.
        self.sample_start_idx = train_data['frame_index']
        
        # Initialize resize
        resize = None
        video, vr = self.process_video_wrapper(train_data[self.vid_data_key])

        # Get video prompt
        prompt = train_data['prompt']
        vr.seek(0)

        prompt_ids = get_prompt_ids(prompt, self.tokenizer)
        # import pdb; pdb.set_trace()
        return video, prompt, prompt_ids
        
    def test_data_batch(self, index):
        #! 用于取val/test数据
        
        # If we are training on individual clips.
        if 'clip_path' in self.train_data[index] and \
            self.train_data[index]['clip_path'] is not None:
            vid_data = self.train_data[index]
            # 1) hdVila100M 数据
            if "hdVila100M_" in vid_data['clip_path']:
                clip_path = vid_data['clip_path'] 
                clip_name = clip_path.split("/")[-1]     # 为载入conf文件中的video_dir路径，只用video path中包含video名字的部分
                # a) load local data
                # clip_path = os.path.join(self.video_dir, clip_name)      
                # b) load mounted-bos data
                # example: /root/ttv_data/hdVila100M/slice_video/2bG33NemyYk/2bG33NemyYk.12.mp4
                clip_name = clip_name.split("hdVila100M_")[-1]  
                video_dir_name = clip_name.split(".")[0]        # 取clip_name, 去掉suffix
                clip_path = os.path.join(self.video_dir, video_dir_name, clip_name)    
            
            # 2) pexels 数据
            elif "/pexels/" in vid_data['clip_path']:
                clip_path = vid_data['clip_path'] 
                if "/root/ttv_data/." not in clip_path:
                    clip_path = self.video_dir + clip_path

            # Get video prompt      
            prompt = vid_data['prompt']     
            video, _ = self.process_video_wrapper(clip_path)    

            prompt_ids = prompt_ids = get_prompt_ids(prompt, self.tokenizer)    
            # import pdb; pdb.set_trace()
            return video, prompt, prompt_ids    

        # Assign train data
        train_data = self.train_data[index]
        # import pdb; pdb.set_trace()
        
        # Get the frame of the current index.
        self.sample_start_idx = train_data['frame_index']
        
        # Initialize resize
        resize = None

        video, vr = self.process_video_wrapper(train_data[self.vid_data_key])

        # Get video prompt
        prompt = train_data['prompt']
        vr.seek(0)
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)
        
        # import pdb; pdb.set_trace()
        return video, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'json'

    def __len__(self):
        if self.train_data is not None:
            return len(self.train_data)
        else: 
            return 0

    def __getitem__(self, index):
        # print("VideoJsonDataset ...")
        # Initialize variables
        video = None
        prompt = None
        prompt_ids = None

        # Use default JSON training
        if self.data_mode == "train":
            try:
                if self.train_data is not None:
                    video, prompt, prompt_ids = self.train_data_batch(index)
                if video.shape[0] < self.n_sample_frames:
                    return self.__getitem__((index + 1) % len(self))
            except:
                video, prompt, prompt_ids = self.train_data_batch(0)

        elif self.data_mode == "val":
            try:
                if self.train_data is not None:
                    video, prompt, prompt_ids = self.test_data_batch(index)     
            except:
                video, prompt, prompt_ids = self.test_data_batch(0)
            print(f'video shape: {video.shape}')
            # import pdb; pdb.set_trace()

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt,
            'dataset': self.__getname__()
        }
        # import pdb; pdb.set_trace()
        return example


class SingleVideoDataset(Dataset):
    def __init__(
        self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 4,
            frame_step: int = 1,
            single_video_path: str = "",
            single_video_prompt: str = "",
            use_caption: bool = False,
            use_bucketing: bool = False,
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing
        self.frames = []
        self.index = 1

        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.n_sample_frames = n_sample_frames
        self.frame_step = frame_step

        self.single_video_path = single_video_path
        self.single_video_prompt = single_video_prompt

        self.width = width
        self.height = height
        
    def create_video_chunks(self):
        # Create a list of frames separated by sample frames
        # [(1,2,3), (4,5,6), ...]
        vr = decord.VideoReader(self.single_video_path)
        vr_range = range(1, len(vr), self.frame_step)

        self.frames = list(self.chunk(vr_range, self.n_sample_frames))

        # Delete any list that contains an out of range index.
        for i, inner_frame_nums in enumerate(self.frames):
            for frame_num in inner_frame_nums:
                if frame_num > len(vr):
                    print(f"Removing out of range index list at position: {i}...")
                    del self.frames[i]

        return self.frames

    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def get_frame_batch(self, vr, resize=None):
        index = self.index
        frames = vr.get_batch(self.frames[self.index])
        video = rearrange(frames, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video

    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize
    
    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        
        return video, vr 

    def single_video_batch(self, index):
        train_data = self.single_video_path
        self.index = index

        if train_data.endswith(self.vid_types):
            video, _ = self.process_video_wrapper(train_data)

            prompt = self.single_video_prompt
            prompt_ids = get_prompt_ids(prompt, self.tokenizer)

            return video, prompt, prompt_ids
        else:
            raise ValueError(f"Single video is not a video type. Types: {self.vid_types}")
    
    @staticmethod
    def __getname__(): return 'single_video'

    def __len__(self):
        
        return len(self.create_video_chunks())

    def __getitem__(self, index):

        video, prompt, prompt_ids = self.single_video_batch(index)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt,
            'dataset': self.__getname__()
        }

        return example
    

class ImageDataset(Dataset):    
    def __init__(
        self,
        tokenizer = None,
        width: int = 256,
        height: int = 256,
        base_width: int = 256,
        base_height: int = 256,
        use_caption:     bool = False,
        image_dir: str = '',
        single_img_prompt: str = '',
        use_bucketing: bool = False,
        fallback_prompt: str = '',
        total_img_num: int = 50000,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.img_types = (".png", ".jpg", ".jpeg", '.bmp')
        self.use_bucketing = use_bucketing
        self.total_img_num = total_img_num
        
        #! self.image_dir = self.get_images_list(image_dir)

        self.image_dir, self.len = self.get_images_list_online(image_dir)
        self.fallback_prompt = fallback_prompt

        self.use_caption = use_caption
        self.single_img_prompt = single_img_prompt

        self.width = width
        self.height = height
        # import pdb; pdb.set_trace()
        
    """"""
    def get_images_list_online(self, image_dir):
        full_img_dir = []
        if os.path.exists(image_dir):
            lines = open(image_dir).readlines()
            random.shuffle(lines)
            # lines = lines[:self.total_img_num]  # 只取指定数量的图像数(default 50000)
            # import pdb; pdb.set_trace()
            for line in lines:
                # laion_aes_v2_ffff0ac23cba9009fe7258ea9d47cf23.jpg       Spider web and dew drops        url
                # name, prompt, url = line.strip().split('\t')
                items = line.strip().split('\t')
                name = items[0]
                prompt = items[1]
                url =  items[-1]
                full_img_dir.append(
                    {
                        "name": name,
                        "prompt": prompt,
                        "url": url
                    }
                )
            # return sorted(full_img_dir)
            return full_img_dir, len(full_img_dir)
        return ['']

    
    def get_images_list(self, image_dir):
        if os.path.exists(image_dir):
            imgs = [x for x in os.listdir(image_dir) if x.endswith(self.img_types)]
            full_img_dir = []

            for img in imgs: 
                full_img_dir.append(f"{image_dir}/{img}")

            return sorted(full_img_dir)

        return ['']

    def image_batch(self, index):
        train_data = self.image_dir[index]
        #! img = train_data
        img_url = train_data['url']
        prompt = train_data['prompt']
        """
        try:
            img = torchvision.io.read_image(img, mode=torchvision.io.ImageReadMode.RGB)
        except:
            img = T.transforms.PILToTensor()(Image.open(img).convert("RGB"))
        """
        try:
            res = requests.get(img_url)
            img = T.transforms.PILToTensor()(Image.open(io.BytesIO(res.content)).convert("RGB"))
        except:
            res = requests.get("http://bj.bcebos.com/kg-aigc-spicloud/laion-aes-v2/img/ffff0ac23cba9009fe7258ea9d47cf23.jpg?authorization=bce-auth-v1%2F934209d6f01a45a99664c67019d775dc%2F2022-11-12T00%3A25%3A30Z%2F-1%2F%2Fb1ff3b410e809dc2c56891687c3581f37580fce62c6ecfeed86a4c6c014a2469")
            img = T.transforms.PILToTensor()(Image.open(io.BytesIO(res.content)).convert("RGB"))
            prompt = "Spider web and dew drops"
        width = self.width
        height = self.height

        if self.use_bucketing:
            _, h, w = img.shape
            # import pdb; pdb.set_trace()
            # width, height = sensible_buckets(width, height, w, h)   # orig setting
            width, height = self.width, self.height
              
        # v0.2b之前的处理方式
        # resize = T.transforms.Resize((height, width), antialias=True); img = resize(img) 

        # v0.2b之后的处理方式
        original_h, original_w = img.shape[-2:]
        crop_scale = min(original_h/height, original_w/width)
        crop_size = (int(crop_scale*height), int(crop_scale*width))
        transform = T.Compose([
            T.transforms.CenterCrop(crop_size),
            T.Resize([height,width])
        ]) # height, width 是目标 h w
        img = transform(img)
        # import pdb; pdb.set_trace()

        img = repeat(img, 'c h w -> f c h w', f=1)
        """
        prompt = get_text_prompt(
            file_path=train_data,
            text_prompt=self.single_img_prompt,
            fallback_prompt=self.fallback_prompt,
            ext_types=self.img_types,  
            use_caption=True
        )
        """
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return img, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'image'
    
    def __len__(self):
        # Image directory
        return self.len
        """
        if os.path.exists(self.image_dir[0]):
            return len(self.image_dir)
        else:
            return 0
        """
    def __getitem__(self, index):
        # print("ImageDataset ...")
        try:
            img, prompt, prompt_ids = self.image_batch(index)
        except:
            img, prompt, prompt_ids = self.image_batch(0)
        print(f'image shape: {img.shape}')
        example = {
            "pixel_values": (img / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt, 
            'dataset': self.__getname__()
        }

        return example


class VideoFolderDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        width: int = 256,
        height: int = 256,
        n_sample_frames: int = 16,
        fps: int = 8,
        path: str = "./data",
        fallback_prompt: str = "",
        use_bucketing: bool = False,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing

        self.fallback_prompt = fallback_prompt

        self.video_files = glob(f"{path}/*.mp4")

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.fps = fps

    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize

    def get_frame_batch(self, vr, resize=None):
        n_sample_frames = self.n_sample_frames
        native_fps = vr.get_avg_fps()
        
        every_nth_frame = max(1, round(native_fps / self.fps))
        every_nth_frame = min(len(vr), every_nth_frame)
        
        effective_length = len(vr) // every_nth_frame
        if effective_length < n_sample_frames:
            n_sample_frames = effective_length

        effective_idx = random.randint(0, (effective_length - n_sample_frames))
        idxs = every_nth_frame * np.arange(effective_idx, effective_idx + n_sample_frames)

        video = vr.get_batch(idxs)
        video = rearrange(video, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video, vr
        
    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        return video, vr
    
    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    @staticmethod
    def __getname__(): return 'folder'

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):

        video, _ = self.process_video_wrapper(self.video_files[index])

        if os.path.exists(self.video_files[index].replace(".mp4", ".txt")):
            with open(self.video_files[index].replace(".mp4", ".txt"), "r") as f:
                prompt = f.read()
        else:
            prompt = self.fallback_prompt

        prompt_ids = self.get_prompt_ids(prompt)

        return {"pixel_values": (video[0] / 127.5 - 1.0), "prompt_ids": prompt_ids[0], "text_prompt": prompt, 'dataset': self.__getname__()}


class CachedDataset(Dataset):
    def __init__(self,cache_dir: str = ''):
        self.cache_dir = cache_dir
        self.cached_data_list = self.get_files_list()

    def get_files_list(self):
        tensors_list = [f"{self.cache_dir}/{x}" for x in os.listdir(self.cache_dir) if x.endswith('.pt')]
        return sorted(tensors_list)

    def __len__(self):
        return len(self.cached_data_list)

    def __getitem__(self, index):
        cached_latent = torch.load(self.cached_data_list[index], map_location='cuda:0')
        return cached_latent


class CombineDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.len_video = 0
        self.len_image = 0
        self.video_dataset = None
        self.image_dataset = None
        for dataset in self.datasets:
            if dataset[0]['pixel_values'].shape[0] > 1:
                self.len_video = len(dataset)
                self.video_dataset = dataset
            else:
                self.len_image = len(dataset)
                self.image_dataset = dataset
        # import pdb; pdb.set_trace()
        print(f"video num:{self.len_video}  image num:{self.len_image}")
    
    def __len__(self):
        # return self.len_video + self.len_image
        # return self.len_video + int(self.len_video*0.25)  #! 手动设置只取有限个image
        return self.len_video
        
    def __getitem__(self, index):
        p_data = random.random()
        # import pdb; pdb.set_trace()
        if p_data < 1:
        # if p_data < 0.8:
            print("get video data")
            return self.video_dataset[index % self.len_video]
        else:
            print("get image data")
            return self.image_dataset[index % self.len_image]