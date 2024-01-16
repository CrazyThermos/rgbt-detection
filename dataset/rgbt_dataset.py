import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import random
import glob
import psutil
import math

from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from utils.general import *

class RGBTDataloader(Dataset):
    # RGB-T train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = RGBTAlbumentations(size=img_size) if augment else None

        
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            # yuhang: Read RGB images and infrared images separately according to the suffix
            self.rgb_files = sorted(x.replace('/', os.sep) for x in f if '_rgb' in x.split('.')[0]) 
            self.t_files = sorted(x.replace('/', os.sep) for x in f if '_t' in x.split('.')[0]) 
            
            
            # self.im_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.rgb_files, f'{prefix}No rgb images found'
            assert self.t_files, f'{prefix}No t images found'
            # yuhang: Check if the images are paired
            assert len(self.rgb_files) == len(self.t_files), f'{prefix}rgb images number is not equal t images {len(self.rgb_files)} != {len(self.t_files)}'
            for index in range(len(self.rgb_files)):
               assert self.rgb_files[index].split('_rgb')[0] == self.t_files[index].split('_t')[0], f'{prefix} index:{index}'
        
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\n') from e

        # Check cache
        self.label_files = img2label_paths(self.t_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.rgb_files + self.t_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training. '

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training. '
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        # yuhang: Check if the images in cache are paired
        self.rgb_files = [x for x in cache.keys() if '_rgb' in x.split('/')[-1]]  # update
        self.t_files = [x for x in cache.keys() if '_t' in x.split('/')[-1]]
        self.label_files = img2label_paths([x for x in cache.keys() if '_t' in x.split('/')[-1]])  # update
        for index in range(len(self.rgb_files)):
            assert self.rgb_files[index].split('_rgb')[0] == self.t_files[index].split('_t')[0], f'{prefix} index:{index}'
            
        # Filter images
        # yuhang: Filter RGB and infrared images separately
        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f'{prefix}{n - len(include)}/{n} images filtered from dataset')
            self.rgb_files = [self.rgb_files[i] for i in include]
            self.t_files = [self.t_files[i] for i in include]     
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh

        # Create indices
        n = len(self.shapes)//2  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Rectangular Training
        # yuhang: Rectangular RGB and infrared images separately
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes[0:n]  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.rgb_files = [self.rgb_files[i] for i in irect]
            self.t_files = [self.t_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into RAM/disk for faster training
        if cache_images == 'ram' and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.rgb_files] + [Path(f).with_suffix('.npy') for f in self.t_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({b / gb:.1f}GB {cache_images})'
            pbar.close()

    def check_cache_ram(self, safety_margin=0.1, prefix=''):
        # Check image caching requirements vs available memory
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.n, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.t_files))  # sample image
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.n / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(f'{prefix}{mem_required / gb:.1f}GB RAM required, '
                        f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{prefix}Scanning {path.parent / path.stem}...'

        # yuhang: Verify RGB and infrared images separately
        with Pool(NUM_THREADS) as pool:
            rgb_pbar = tqdm(pool.imap(verify_image_label, zip(self.rgb_files, self.label_files, repeat(prefix))),
                        desc=desc,
                        total=len(self.t_files),
                        bar_format=TQDM_BAR_FORMAT)
            for rgb_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in rgb_pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if rgb_file:
                    x[rgb_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                rgb_pbar.desc = f'{desc} {nf} rgb images, {nm + ne} backgrounds, {nc} corrupt'
        rgb_pbar.close()

        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        with Pool(NUM_THREADS) as pool:
            t_pbar = tqdm(pool.imap(verify_image_label, zip(self.t_files, self.label_files, repeat(prefix))),
                        desc=desc,
                        total=len(self.t_files),
                        bar_format=TQDM_BAR_FORMAT)
            for t_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in t_pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if t_file:
                    x[t_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                t_pbar.desc = f'{desc} {nf} t images, {nm + ne} backgrounds, {nc} corrupt'
        t_pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING ⚠️ No labels found in {path}. ')
        x['hash'] = get_hash(self.label_files + self.rgb_files + self.t_files)
        x['results'] = nf, nm, ne, nc, len(self.t_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        return len(self.t_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']

        # yuhang: In data augmentation, RGB and infrared images need to be enhanced in pairs
        if mosaic:
            # Load mosaic
            img_rgb, img_t, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img_rgb, img_t, labels = rgbt_mixup(img_rgb, img_t, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img_rgb, _, _ = self.load_image(index, imtype='rgb')
            img_t, (h0, w0), (h, w) = self.load_image(index, imtype='t')


            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img_rgb, _, _ = letterbox(img_rgb, shape, auto=False, scaleup=self.augment)
            img_t, ratio, pad = letterbox(img_t, shape, auto=False, scaleup=self.augment)
            
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img_rgb, img_t, labels = rgbt_random_perspective(img_rgb,
                                                                img_t,
                                                                labels,
                                                                degrees=hyp['degrees'],
                                                                translate=hyp['translate'],
                                                                scale=hyp['scale'],
                                                                shear=hyp['shear'],
                                                                perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img_t.shape[1], h=img_t.shape[0], clip=True, eps=1E-3)

        # yuhang: In data augmentation, RGB and infrared images need to be enhanced in pairs
        if self.augment:         
            # Albumentations
            img_rgb, img_t, labels = self.albumentations(img_rgb, img_t, labels)
            
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img_rgb, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            augment_hsv(img_t, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img_rgb = np.flipud(img_rgb)
                img_t = np.flipud(img_t)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img_rgb = np.fliplr(img_rgb)
                img_t = np.fliplr(img_t)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img_rgb = img_rgb.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_rgb = np.ascontiguousarray(img_rgb)
        img_t = img_t.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_t = np.ascontiguousarray(img_t)

        return torch.from_numpy(img_rgb), torch.from_numpy(img_t), labels_out, self.rgb_files[index], self.t_files[index], shapes


    def load_image(self, i, imtype='t'):
        # Loads 1 image from specify the type of dataset index 'i', returns (im, original hw, resized hw)
        # imtype mean the type of dataset, rgb/t
        if imtype == 't': 
            im, f, fn = self.ims[i], self.t_files[i], self.npy_files[i],
        elif imtype == 'rgb':
            im, f, fn = self.ims[i], self.rgb_files[i], self.npy_files[i],
        else :
            raise Exception(f'imtype only can be rgb/t!\n') 
        
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.rgb_files[i]))
            np.save(f.as_posix(), cv2.imread(self.t_files[i]))


    def load_mosaic(self, index):
        # RGBT 4-mosaic loader. 
        # Loads 1 RGB image + 3 random RGB images into a 4-image RGB mosaic. 
        # Loads 1 T image + 3 random T images into a 4-image T mosaic
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img_rgb, _, (_, _) = self.load_image(index, imtype='rgb')
            img_t, _, (h, w) = self.load_image(index, imtype='t')

            # place img in img4
            if i == 0:  # top left
                img_rgb4 = np.full((s * 2, s * 2, img_rgb.shape[2]), 114, dtype=np.uint8)  # base RGB image with 4 tiles
                img_t4 = np.full((s * 2, s * 2, img_t.shape[2]), 114, dtype=np.uint8)  # base T image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img_rgb4[y1a:y2a, x1a:x2a] = img_rgb[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            img_t4[y1a:y2a, x1a:x2a] = img_t[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img_rgb4, img_t4, labels4, segments4 = rgbt_copy_paste(img_rgb4, img_t4, labels4, segments4, p=self.hyp['copy_paste'])
        img_rgb4, img_t4, labels4 = rgbt_random_perspective(img_rgb4, 
                                                            img_t4,
                                                            labels4,
                                                            segments4,
                                                            degrees=self.hyp['degrees'],
                                                            translate=self.hyp['translate'],
                                                            scale=self.hyp['scale'],
                                                            shear=self.hyp['shear'],
                                                            perspective=self.hyp['perspective'],
                                                            border=self.mosaic_border)  # border to remove

        return img_rgb4, img_t4, labels4

    def load_mosaic9(self, index):
        # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9, labels9, segments9 = copy_paste(img9, labels9, segments9, p=self.hyp['copy_paste'])
        img9, labels9 = random_perspective(img9,
                                           labels9,
                                           segments9,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img9, labels9

    @staticmethod
    def collate_fn(batch):
        im_rgb, im_t, label, rgb_path, t_path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im_rgb, 0), torch.stack(im_t, 0), torch.cat(label, 0), rgb_path, t_path, shapes

    @staticmethod
    def collate_fn4(batch):
        im_rgb, im_t, label, rgb_path, t_path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4_rgb, im4_t, label4, path4, shapes4 = [], [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im1 = F.interpolate(im_rgb[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
                                    align_corners=False)[0].type(im_rgb[i].type())
                im2 = F.interpolate(im_t[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
                                    align_corners=False)[0].type(im_t[i].type())
                lb = label[i]
            else:
                im1 = torch.cat((torch.cat((im_rgb[i], im_rgb[i + 1]), 1), torch.cat((im_rgb[i + 2], im_rgb[i + 3]), 1)), 2)
                im2 = torch.cat((torch.cat((im_t[i], im_t[i + 1]), 1), torch.cat((im_t[i + 2], im_t[i + 3]), 1)), 2)

                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4_rgb.append(im1)
            im4_t.append(im2)

            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4_rgb, 0), torch.stack(im4_rgb, 0), torch.cat(label4, 0), path4, shapes4

class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)
            


def create_rgbtdataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False,
                      seed=0):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = RGBTDataloader(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=RGBTDataloader.collate_fn4 if quad else RGBTDataloader.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset

