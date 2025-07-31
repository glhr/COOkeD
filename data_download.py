# script adapted from https://github.com/Jingkang50/OpenOOD/blob/main/scripts/download/download.py

import argparse
import os
import zipfile

import gdown
import tarfile

benchmarks_dict = {
    'cifar-100':
    ['cifar100', 'cifar10', 'tin', 'svhn', 'texture', 'places365', 'mnist'],
    'imagenet-200': [
        'imagenet_1k','ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o'
    ],
    'imagenet-1k': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o',
        'imagenet_v2',
        'imagenet_c',
        'imagenet_r'
    ],
    'ooddb_patternnet': ['PatternNet'],
    'ooddb_dtd': ['DTD'],
}

dir_dict = {
    'images_classic/': [
        'cifar100', 'tin', 'svhn', 'mnist', 'cifar10', 'places365','texture'
    ],
    'images_largescale/': [
        'imagenet_1k',
        'ssb_hard',
        'ninco',
        'inaturalist',
        'openimage_o',
        'imagenet_v2',
        'imagenet_c',
        'imagenet_r',
        'DTD',
        'PatternNet'
    ],
}

download_dataset_dict = {
    'cifar100': '1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_',
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    'cifar10c': '170DU_ficWWmbh6O2wqELxK9jxRiGhlJH',
    'svhn': '1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI',
    'mnist': '1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb',
    'tin': '1PZ-ixyx52U989IKsMA2OT-24fToTrelC',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'places365': '1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay',
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
    'ssb_hard': '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
    'ninco': '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
    'imagenet_v2': '1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho',
    'imagenet_r': '1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU',
    'imagenet_c': '1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt',
    'imagenet_o': '1S9cFV7fGvJCcka220-pIO9JPZL1p1V8w',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    "DTD": '17Z_-3wgrLIljLzFJ8VhnCTR6rEpw51c4', # originally downloaded from https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
    "PatternNet": "1QhVD6Wi1glRCBpNlcjElCNiICpzXx2qA" # originally downloaded from https://nuisteducn1-my.sharepoint.com/:u:/g/personal/zhouwx_nuist_edu_cn/EYSPYqBztbBBqS27B7uM_mEB3R9maNJze8M1Qg9Q6cnPBQ?e=MSf977
}

download_checkpoint_dict = {
    # TODO: 
}

def require_download(filename, path):
    for item in os.listdir(path):
        if item.startswith(filename) or filename.startswith(
                item) or path.endswith(filename):
            return False

    else:
        print(filename + ' needs download:')
        return True


def download_dataset(dataset, args, use_gdown=True):
    print(f'----> dataset: {dataset}')
    #if "imagenet" in dataset: return
    try:
        for key in dir_dict.keys():
            if dataset in dir_dict[key]:
                store_path = os.path.join(args.save_dir[0], key, dataset)
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
                break
        else:
            print(dir_dict.keys())
            print('Invalid dataset detected {}'.format(dataset))
            return

        if require_download(dataset, store_path):
            print(store_path)
            if not store_path.endswith('/'):
                store_path = store_path + '/'
            gdown.download(id=download_dataset_dict[dataset], output=store_path)
            file_path = os.path.join(store_path, dataset + '.zip')

            if file_path.endswith('.tar.gz'):
                with tarfile.open(file_path, 'r:gz') as tar_file:
                    tar_file.extractall(store_path)
            elif file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_file:
                    zip_file.extractall(store_path)
            os.remove(file_path)
    except Exception as e:
        print(f"Error downloading {dataset}: {e}")
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download datasets and checkpoints')
    parser.add_argument('--contents',
                        nargs='+',
                        default=['datasets'])
    parser.add_argument('--datasets', nargs='+', default=['all'])
    parser.add_argument('--checkpoints', nargs='+', default=['all'])
    parser.add_argument('--save_dir',
                        nargs='+',
                        default=['./data', './checkpoints'])
    args = parser.parse_args()

    if args.datasets[0] == 'all':
        args.datasets = [
            'imagenet-1k', 'imagenet-200', 'cifar-100', 'ooddb_patternnet', 'ooddb_dtd'
        ]

    if args.checkpoints[0] == 'all':
        args.checkpoints = [
            'imagenet-1k', 'imagenet-200', 'cifar-100', 'ooddb_patternnet', 'ooddb_dtd'
        ]

    for content in args.contents:
        if content == 'datasets':
            store_path = args.save_dir[0]
            if not store_path.endswith('/'):
                store_path = store_path + '/'

            for benchmark in args.datasets:
                print(f'Downloading data for benchmark: {benchmark}')
                for dataset in benchmarks_dict[benchmark]:
                    download_dataset(dataset, args)

        elif content == 'checkpoints':
            store_path = os.path.join(args.save_dir[1], ' classifiers/')
            if not os.path.exists(store_path):
                os.makedirs(store_path)

            if not store_path.endswith('/'):
                store_path = store_path + '/'

            for checkpoint in args.checkpoints:
                if require_download(checkpoint, store_path):
                    gdown.download(id=download_checkpoint_dict[checkpoint],
                                   output=store_path)
                    file_path = os.path.join(store_path, checkpoint + '.zip')
                    with zipfile.ZipFile(file_path, 'r') as zip_file:
                        zip_file.extractall(store_path)
                    os.remove(file_path)