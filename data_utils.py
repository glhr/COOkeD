import ast
import io
import logging
import os
import pickle

import torch
from PIL import Image, ImageFile

import logging
import random
import traceback

from torch.utils.data import Dataset
import torchvision.transforms as tvs_trans

from openood.preprocessors.transform import normalization_dict, Convert, interpolation_modes

CLIP_NORM_PARAMS = [[0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]]

def get_img_normalizer(setting):
    if setting in normalization_dict.keys():
        norm_mean = normalization_dict[setting][0]
        norm_std = normalization_dict[setting][1]
    elif setting == "clip":
        norm_mean = CLIP_NORM_PARAMS[0]
        norm_std = CLIP_NORM_PARAMS[1]
    else:
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
    return tvs_trans.Normalize(mean=norm_mean, std=norm_std)

def get_preprocessor(crop_mode="centercrop", image_size=224, norm_params=None):
    if crop_mode == "resize":
        tf_list = [
            Convert('RGB'),
            tvs_trans.Resize((image_size,image_size),
                                interpolation=interpolation_modes["bilinear"]),
            tvs_trans.ToTensor()
        ]
        
    elif crop_mode == "centercrop":
        tf_list = [
            Convert('RGB'),
            tvs_trans.Resize(image_size,
                                interpolation=interpolation_modes["bilinear"]),
            #tvs_trans.RandomCrop(image_size, padding=4),
            tvs_trans.CenterCrop(image_size),
            tvs_trans.ToTensor()
        ]
    if norm_params is not None:
        tf_list.append(tvs_trans.Normalize(mean=norm_params[0], std=norm_params[1]))
    return tvs_trans.Compose(tf_list)
        
def preprocess_image_for_clip(image, size=224):
    norm_params = CLIP_NORM_PARAMS
    tf = get_preprocessor(crop_mode="resize", image_size=size, norm_params=norm_params)
    return tf(image).unsqueeze(0)  # add batch dimension

def preprocess_image_for_cls(image, size=224, id_name='imagenet'):
    norm_params = normalization_dict[id_name]
    tf = get_preprocessor(crop_mode="resize", image_size=size, norm_params=norm_params)
    return tf(image).unsqueeze(0)  # add batch dimension


class BaseDataset(Dataset):
    def __init__(self, pseudo_index=-1, skip_broken=False, new_index='next'):
        super(BaseDataset, self).__init__()
        self.pseudo_index = pseudo_index
        self.skip_broken = skip_broken
        self.new_index = new_index
        if new_index not in ('next', 'rand'):
            raise ValueError('new_index not one of ("next", "rand")')

    def __getitem__(self, index):
        # in some pytorch versions, input index will be torch.Tensor
        index = int(index)

        # if sampler produce pseudo_index,
        # randomly sample an index, and mark it as pseudo
        if index == self.pseudo_index:
            index = random.randrange(len(self))
            pseudo = 1
        else:
            pseudo = 0

        while True:
            try:
                sample = self.getitem(index)
                break
            except Exception as e:
                if self.skip_broken and not isinstance(e, NotImplementedError):
                    if self.new_index == 'next':
                        new_index = (index + 1) % len(self)
                    else:
                        new_index = random.randrange(len(self))
                    logging.warn(
                        'skip broken index [{}], use next index [{}]'.format(
                            index, new_index))
                    index = new_index
                else:
                    logging.error('index [{}] broken'.format(index))
                    traceback.print_exc()
                    logging.error(e)
                    raise e

        sample['index'] = index
        sample['pseudo'] = pseudo
        return sample

    def getitem(self, index):
        raise NotImplementedError


DATA_INFO = {
    'cifar100': {
        'num_classes': 100,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/train_cifar100.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_cifar100.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/test_cifar100.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar10', 'tin'],
                'cifar10': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_cifar10.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_places365.txt'
                }
            },
        }
    },
    'cifar100n_noisyfine': {
        'num_classes': 100,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/train_cifar100n_noisyfine.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_cifar100.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/test_cifar100.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar10', 'tin'],
                'cifar10': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_cifar10.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_places365.txt'
                }
            },
        }
    },

    'imagenet200': {
        'num_classes': 200,
        'id': {
            'train': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/train_imagenet200.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_imagenet200.txt'
            },
            'test': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_v2.txt'
            },
            'imagenet_c': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_c.txt'
            },
            'imagenet_r': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_r.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir':
                    'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_openimage_o.txt'
                },
            },
        }
    },
    'imagenet': {
        'num_classes': 1000,
        'id': {
            'train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/train_imagenet.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/val_imagenet.txt'
            },
            'test': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/test_imagenet.txt'
            }
        },
        'csid': {
            'datasets':
            ['imagenet_v2', 'imagenet_c', 'imagenet_r', 'imagenet_es'],
            'imagenet_v2': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_v2.txt'
            },
            'imagenet_c': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_c.txt'
            },
            'imagenet_r': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_r.txt'
            },
            'imagenet_es': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_es.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/imagenet/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_openimage_o.txt'
                },
            },
        }
    },
}



# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True

oodb_class_num = {
    "patternnet": 19,
    "dtd": 23
}

for variant in range(3):
    for oodb_dataset_name in ["patternnet", "dtd"]:
        oodb_dataset_name_ood = "dtd" if oodb_dataset_name == "patternnet" else "patternnet"
        DATA_INFO[f"ooddb_{oodb_dataset_name}_{variant}"] = {
            'num_classes': oodb_class_num[oodb_dataset_name],
            'id': {
                'train': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': f"benchmark_imglist/ooddb/ooddb_{oodb_dataset_name}_id_{variant}_train.txt"
                },
                'val': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': f"benchmark_imglist/ooddb/ooddb_{oodb_dataset_name}_id_{variant}_test.txt"
                },
                'test': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': f"benchmark_imglist/ooddb/ooddb_{oodb_dataset_name}_id_{variant}_test.txt"
                }
            },
            'ood': {
                'near': {
                    f"ooddb_{oodb_dataset_name}_{variant}": {
                        'data_dir': 'images_largescale/',
                        'imglist_path': f"benchmark_imglist/ooddb/ooddb_{oodb_dataset_name}_ood_{variant}_test.txt"
                    },
                },
                'far': {
                    'cifar10': {
                        'data_dir': 'images_classic/',
                        'imglist_path':
                        'benchmark_imglist/cifar100/test_cifar10.txt'
                    },
                    'tin': {
                        'data_dir': 'images_classic/',
                        'imglist_path': 'benchmark_imglist/cifar100/test_tin.txt'
                    },
                    f"ooddb_{oodb_dataset_name_ood}_{variant}": {
                        'data_dir': 'images_largescale/',
                        'imglist_path': f"benchmark_imglist/ooddb/ooddb_{oodb_dataset_name_ood}_ood_{variant}_test.txt"
                    },
                    'mnist': {
                        'data_dir': 'images_classic/',
                        'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                    },
                    'svhn': {
                        'data_dir': 'images_classic/',
                        'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
                    },
                    'texture': {
                        'data_dir': 'images_classic/',
                        'imglist_path':
                        'benchmark_imglist/cifar100/test_texture.txt'
                    },
                    'places365': {
                        'data_dir': 'images_classic/',
                        'imglist_path':
                        'benchmark_imglist/cifar100/test_places365.txt'
                    },
                }
            }
        }

class ImglistDataset(BaseDataset):
    def __init__(self,
                 name,
                 imglist_pth,
                 data_dir,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(ImglistDataset, self).__init__(**kwargs)

        self.name = name
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}

        try:
            with open(path, 'rb') as f:
                content = f.read()
            filebytes = content
            buff = io.BytesIO(filebytes)
            
            image = Image.open(buff).convert('RGB')
            sample['data'] = self.transform_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)
            # Generate Soft Label
            # soft_label = torch.Tensor(self.num_classes)
            # if sample['label'] < 0:
            #     soft_label.fill_(1.0 / self.num_classes)
            # else:
            #     soft_label.fill_(0)
            #     soft_label[sample['label']] = 1
            # sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample


class ImglistDataset_CLIPWds(BaseDataset):
    def __init__(self,
                 name,
                 imglist_pth,
                 data_dir,
                 wds_index_json,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(ImglistDataset_CLIPWds, self).__init__(**kwargs)

        import wids

        self.name = name
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen

        self.wds_index_json = wds_index_json
        self.wds_dataset = wids.ShardListDataset(self.wds_index_json, cache_dir=f'./cache/wids/{name}', keep=True)
        print(f"Loaded {self.wds_index_json} with {len(self.wds_dataset)} samples")
        self.load_imgs = True

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}

        try:
            if self.load_imgs:
                with open(path, 'rb') as f:
                    content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                sample['data'] = self.transform_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)
            
            wds_sample = self.wds_dataset[index]
            sample["clip_feat"] = pickle.loads(wds_sample['.clip_feat.pyd'].read())
            #print(f"clip_feat.shape: {clip_feat.shape}")

        except Exception as e:
            print(self.wds_index_json, index, len(self.wds_dataset))
            logging.error('[{}] broken'.format(path))
            raise e
        return sample


def get_dataloader(dataset, loader_kwargs):
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)

def get_id_dataset(id_name, split, preprocessor, data_root='./data', args=None):
    data_info = DATA_INFO[id_name]
    totensor_tf = tvs_trans.ToTensor()

    if args is None or args.data_mode == "standard":
        return ImglistDataset(
        name='_'.join((id_name, split)),
        imglist_pth=os.path.join(data_root,
                                    data_info['id'][split]['imglist_path']),
        data_dir=os.path.join(data_root,
                                data_info['id'][split]['data_dir']),
        num_classes=data_info['num_classes'],
        preprocessor=preprocessor,
        data_aux_preprocessor=totensor_tf)
    elif args.data_mode == "webdataset":

        dataset_folder_name = {
            "ooddb_patternnet_0": "PatternNet",
            "ooddb_dtd_0": "DTD",
            "imagenet200": "imagenet200",
            "imagenet": "imagenet_1k",
            "cifar100n_noisyfine": "cifar100"
        }
        preprocessor_eval_str = "" if args.preprocessor_eval == "resize" else f"{args.preprocessor_eval}_"
        wds_index_filename = f"{split}_clipfeat_{args.clip_architecture}-{args.clip_pretrained}-{args.clip_dim}_{preprocessor_eval_str}index.json"
        return ImglistDataset_CLIPWds(
        name='_'.join((id_name, split)),
        imglist_pth=os.path.join(data_root,
                                    data_info['id'][split]['imglist_path']),
        data_dir=os.path.join(data_root,
                                data_info['id'][split]['data_dir']),
        wds_index_json=os.path.join(data_root,data_info['id'][split]['data_dir'],dataset_folder_name.get(id_name,id_name),wds_index_filename), # TODO: fix data folder
        num_classes=data_info['num_classes'],
        preprocessor=preprocessor,
        data_aux_preprocessor=totensor_tf)

def get_csid_dataset(id_name, csid_name, preprocessor, data_root='./data', args=None):
    data_info = DATA_INFO[id_name]
    totensor_tf = tvs_trans.ToTensor()

    if args is None or args.data_mode =="standard":
        return ImglistDataset(
        name='_'.join((id_name, csid_name)),
        imglist_pth=os.path.join(data_root,
                                    data_info['csid'][csid_name]['imglist_path']),
        data_dir=os.path.join(data_root,
                                data_info['csid'][csid_name]['data_dir']),
        num_classes=data_info['num_classes'],
        preprocessor=preprocessor,
        data_aux_preprocessor=totensor_tf)
    elif args.data_mode == "webdataset":
        dataset_folder_name = {
            "imagenet200": "imagenet200",
            "imagenet": "imagenet_1k"
        }
        preprocessor_eval_str = "" if args.preprocessor_eval == "resize" else f"{args.preprocessor_eval}_"
        wds_index_filename = f"{csid_name}_clipfeat_{args.clip_architecture}-{args.clip_pretrained}-{args.clip_dim}_{preprocessor_eval_str}index.json"
        return ImglistDataset_CLIPWds(
            name='_'.join((csid_name, id_name)),
            imglist_pth=os.path.join(data_root,
                                    data_info['csid'][csid_name]['imglist_path']),
            data_dir=os.path.join(data_root,
                                data_info['csid'][csid_name]['data_dir']),
            num_classes=data_info['num_classes'],
            preprocessor=preprocessor,
            data_aux_preprocessor=totensor_tf,
            wds_index_json=os.path.join(data_root,data_info['id']["test"]['data_dir'],dataset_folder_name.get(id_name,id_name),wds_index_filename), # TODO: fix data folder
        )

def get_ood_dataset(id_name, ood_name, preprocessor, split="near", data_root='./data', args=None):
    data_info = DATA_INFO[id_name]
    totensor_tf = tvs_trans.ToTensor()

    if args is None or args.data_mode =="standard":
        return ImglistDataset(
        name='_'.join((id_name, ood_name)),
        imglist_pth=os.path.join(data_root,
                                    data_info['ood'][split][ood_name]['imglist_path']),
        data_dir=os.path.join(data_root,
                                data_info['ood'][split][ood_name]['data_dir']),
        num_classes=data_info['num_classes'],
        preprocessor=preprocessor,
        data_aux_preprocessor=totensor_tf)
    elif args.data_mode == "webdataset":
        dataset_folder_name = {
            "ooddb_patternnet_0": "PatternNet",
            "ooddb_dtd_0": "DTD",
            "imagenet200": "imagenet200",
            "imagenet": "imagenet_1k",
            "textures": "texture",
            "cifar100n_noisyfine": "cifar100"
        }
        update_id_name = {
            "cifar100n_noisyfine": "cifar100"
        }
        preprocessor_eval_str = "" if args.preprocessor_eval == "resize" else f"{args.preprocessor_eval}_"
        wds_index_filename = f"{update_id_name.get(id_name,id_name)}_clipfeat_{args.clip_architecture}-{args.clip_pretrained}-{args.clip_dim}_{preprocessor_eval_str}index.json"
        return ImglistDataset_CLIPWds(
            name='_'.join((ood_name, id_name)),
            imglist_pth=os.path.join(data_root,
                                    data_info['ood'][split][ood_name]['imglist_path']),
            data_dir=os.path.join(data_root,
                                data_info['ood'][split][ood_name]['data_dir']),
            num_classes=data_info['num_classes'],
            preprocessor=preprocessor,
            data_aux_preprocessor=totensor_tf,
            wds_index_json=os.path.join(data_root,data_info['ood'][split][ood_name]['data_dir'],dataset_folder_name.get(ood_name,ood_name),wds_index_filename), # TODO: fix data folder
        )

def get_csid_dict(id_name):
    if "imagenet" in id_name:
        csid_datasets = {
            'imagenet_v2': 'csid',
            'imagenet_c': 'csid',
            'imagenet_r': 'csid',
        }
    return csid_datasets

def get_ood_dict(id_name):
    if "cifar" in id_name:
        ood_datasets = {
            'svhn': 'far',
            'texture': 'far',
            'tin': 'near',
            'places365': 'far',
            'mnist': 'far',
            'cifar100' if id_name == 'cifar10' else 'cifar10': 'near'
        }
    elif "imagenet" in id_name:
        ood_datasets = {
            'inaturalist': 'far',
            'textures': 'far',
            'openimage_o': 'far',
            'ninco': 'near',
            'ssb_hard': 'near',
        }
    elif "ooddb" in id_name:
        ood_datasets = {
           id_name: 'near',
           'svhn': 'far',
            'ooddb_patternnet_0' if 'dtd' in id_name else 'ooddb_dtd_0': 'far',
            'tin': 'far',
            'places365': 'far',
            'mnist': 'far',
            'cifar10': 'far'
        }
    return ood_datasets

cifar100_labels = [
    'apple',  # id 0
    'aquarium_fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer_keyboard',
    'lamp',
    'lawn_mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple_tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak_tree',
    'orange',
    'orchid',
    'otter',
    'palm_tree',
    'pear',
    'pickup_truck',
    'pine_tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet_pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow_tree',
    'wolf',
    'woman',
    'worm',
]


# make a mapping from imagenet200 class idx to class name

# first, load the txt file containing path and label: data/benchmark_imglist/imagenet200/train_imagenet200.txt
# then, split the line by space and get the label

with open('data/benchmark_imglist/imagenet200/train_imagenet200.txt') as f:
    lines_imagenet200 = f.readlines()

imagenet200_label_to_wordnet_id = {}


for line in lines_imagenet200:
    path = line.split(' ')[0] # format imagenet_1k/train/n01443537/n01443537_10007.JPEG
    label = int(line.split(' ')[1].strip()) # format 0
    wordnet_id = path.split('/')[2] # format n01443537
    imagenet200_label_to_wordnet_id[label] = wordnet_id

imagenet1k_label_to_wordnet_id = {}

with open('data/benchmark_imglist/imagenet/train_imagenet.txt') as f:
    lines_imagenet1k = f.readlines()

for line in lines_imagenet1k:
    path = line.split(' ')[0] # format imagenet_1k/train/n01443537/n01443537_10007.JPEG
    label = int(line.split(' ')[1].strip()) # format 0
    wordnet_id = path.split('/')[2] # format n01443537
    imagenet1k_label_to_wordnet_id[label] = wordnet_id

#print(imagenet200_label_to_wordnet_id)

wordnet_id_to_class_name = {}
wordnet_id_to_class_idx = {}

# download https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
import json
import requests
import os

from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES

if not os.path.exists('imagenet_class_index.json'):
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    response = requests.get(url)
    with open('imagenet_class_index.json', 'wb') as f:
        f.write(response.content)

with open('imagenet_class_index.json') as f:
    class_index = json.load(f)

for key, value in class_index.items():
    wordnet_id = value[0]
    class_name = value[1]
    wordnet_id_to_class_name[wordnet_id] = class_name
    wordnet_id_to_class_idx[wordnet_id] = int(key)

#print(wordnet_id_to_class_name)

imagenet200_label_to_class_idx = {}
for label, wordnet_id in imagenet200_label_to_wordnet_id.items():
    class_idx = wordnet_id_to_class_idx[wordnet_id]
    imagenet200_label_to_class_idx[label] = class_idx

imagenet1k_label_to_class_idx = {}
for label, wordnet_id in imagenet1k_label_to_wordnet_id.items():
    class_idx = wordnet_id_to_class_idx[wordnet_id]
    imagenet1k_label_to_class_idx[label] = class_idx

imagenet200_label_to_class_name = {}
for orig_idx, in_idx in imagenet200_label_to_class_idx.items():
    imagenet200_label_to_class_name[orig_idx] = IMAGENET_CLASSNAMES[in_idx]

imagenet1k_label_to_class_name = {}
for orig_idx, in_idx in imagenet1k_label_to_class_idx.items():
    imagenet1k_label_to_class_name[orig_idx] = IMAGENET_CLASSNAMES[in_idx]
#print(imagenet200_label_to_class_name)

def get_label_to_class_mapping(id_name, ooddb_split="train",sanity_check=True):
    
    if "cifar100" in id_name:
        mapping = dict(zip(range(len(cifar100_labels)),cifar100_labels))
        
    elif id_name == "imagenet200":
        mapping = imagenet200_label_to_class_name
    elif id_name == "imagenet":
        mapping = imagenet1k_label_to_class_name
    elif id_name == "ooddb_patternnet_0":
        from OODDB.utils import get_dataset_split_info
        _,_, class_idx_to_name = get_dataset_split_info(
            dataset="patternnet",
            split=ooddb_split,
            data_order=0,
        )
        mapping = class_idx_to_name
    elif id_name == "ooddb_dtd_0":
        from OODDB.utils import get_dataset_split_info
        _,_, class_idx_to_name = get_dataset_split_info(
            dataset="dtd",
            split=ooddb_split,
            data_order=0,
        )
        print(class_idx_to_name)
        mapping = class_idx_to_name

    mapping ={k:v.replace("_"," ") for k,v in mapping.items()}

    if sanity_check:
        assert len(mapping) == DATA_INFO[id_name]["num_classes"], \
        f"Number of classes in mapping ({len(mapping)}) does not match number of classes in dataset ({DATA_INFO[id_name]['num_classes']})"
    return mapping
    
if __name__ == "__main__":
    
    print("--patternnet")
    id_classes = get_label_to_class_mapping("ooddb_patternnet_0", ooddb_split="train")
    all_classes = get_label_to_class_mapping("ooddb_patternnet_0", ooddb_split="test", sanity_check=False)

    ood_classes = {k:v for k,v in all_classes.items() if k not in id_classes}

    print(f"ID classes: {id_classes}")
    print(f"OOD classes: {ood_classes}")

    print("-dtd")
    id_classes = get_label_to_class_mapping("ooddb_dtd_0", ooddb_split="train")
    all_classes = get_label_to_class_mapping("ooddb_dtd_0", ooddb_split="test", sanity_check=False)

    ood_classes = {k:v for k,v in all_classes.items() if k not in id_classes}

    print(f"ID classes: {id_classes}")
    print(f"OOD classes: {ood_classes}")

    for id_name in ["cifar100","imagenet200","imagenet"]:
        print(f"------------",id_name)
        id_classes = get_label_to_class_mapping(id_name)
        print(id_classes)