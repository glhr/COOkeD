import torch.nn as nn
import torch.nn.functional as F
import torch
import glob

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.networks.resnet18_224x224 import ResNet18_224x224
from openood.networks.resnet50 import ResNet50
from openood.networks.vit_b_16 import ViT_B_16

from data_utils import DATA_INFO

clip_architecture_to_dim = {
        'ViT-B-32': 512,
        'ViT-B-16': 512,
        'ViT-H-14-quickgelu': 1024,
        'ViT-H-14': 1024,
        'ViT-L-14': 768,
    }

clip_architecture_to_quickgelu = {
        'ViT-B-32': True,
        'ViT-B-16': True,
        'ViT-H-14': False,
        'ViT-L-14': True,
    }

# classification checkpoints per dataset
classifiers = {
    "imagenet": {
        "resnet18-ft": "resnet18-f37072fd.pth",
        "resnet50-ft": "resnet50-0676ba61.pth",
        "vit-b-16-ft": "vit_b_16-c867db91.pth",
    },
    "imagenet200": {
        "resnet18-scratch": "imagenet200_resnet18_224x224_e100_lr0.1_loss_cls_mixnone_20250211-093940",
        "resnet18-ft": "imagenet200_resnet18_224x224_e100_lr0.1_loss_cls_mixnone_20250207-231839",
        "resnet50-ft": "imagenet200_resnet50_224x224_e100_lr0.01_loss_cls_mixnone_20250222-034614",
        "vit-b-16-ft": "imagenet200_vit-b-16_224x224_e100_lr0.01_loss_cls_mixnone_20250221-211308"
    },
    "cifar100": {
        "resnet18-scratch": "cifar100_resnet18_224x224_e100_lr0.1_loss_cls_mixnone_20250204-175808",
        "resnet18-ft": "cifar100_resnet18_224x224_e100_lr0.01_loss_cls_mixnone_20250208-064731",
        "resnet50-ft": "cifar100_resnet50_224x224_e100_lr0.01_loss_cls_mixnone_20250215-012046",
        "vit-b-16-ft": "cifar100_vit-b-16_224x224_e100_lr0.01_loss_cls_mixnone_20250218-134736"
    },
    "ooddb_dtd_0": {
        "resnet18-scratch": "ooddb_dtd_0_resnet18_224x224_e100_lr0.1_loss_cls_mixnone_20250228-104720",
        "resnet18-ft": "ooddb_dtd_0_resnet18_224x224_e100_lr0.01_loss_cls_mixnone_20250228-113449",
        "resnet50-ft": "ooddb_dtd_0_resnet50_224x224_e100_lr0.01_loss_cls_mixnone_20250228-110835",
        "vit-b-16-ft": "ooddb_dtd_0_vit-b-16_224x224_e100_lr0.01_loss_cls_mixnone_ViT-B-32openai_20250301-111629"
    },
    "ooddb_patternnet_0": {
        "resnet18-scratch": "ooddb_patternnet_0_resnet18_224x224_e100_lr0.1_loss_cls_mixnone_20250228-102145",
        "resnet18-ft": "ooddb_patternnet_0_resnet18_224x224_e100_lr0.01_loss_cls_mixnone_20250228-110811",
        "resnet50-ft": "ooddb_patternnet_0_resnet50_224x224_e100_lr0.01_loss_cls_mixnone_20250228-113157",
        "vit-b-16-ft": "ooddb_patternnet_0_vit-b-16_224x224_e100_lr0.01_loss_cls_mixnone_ViT-B-32openai_20250301-110254"
    },
    "cifar100n_noisyfine": {
        "resnet18-scratch": "cifar100n_noisyfine_resnet18_224x224_e100_lr0.1_loss_cls_mixnone_20250303-095457",
        "resnet18-ft": "cifar100n_noisyfine_resnet18_224x224_e100_lr0.01_loss_cls_mixnone_20250303-095058",
        "resnet50-ft": "cifar100n_noisyfine_resnet50_224x224_e100_lr0.01_loss_cls_mixnone_20250303-112908",
        "vit-b-16-ft": "cifar100n_noisyfine_vit-b-16_224x224_e100_lr0.01_loss_cls_mixnone_20250305-092442",
        
}
}

# linear probe checkpoints per dataset
probes = {
"imagenet": {
    "ViT-B-32+openai": "imagenet_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_ViT-B-32openai_20250223-164520",
    "ViT-B-16+openai": "imagenet_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_ViT-B-16openai_20250223-164912",
    "ViT-L-14+openai": "imagenet_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_ViT-L-14openai_20250223-172735",
    "ViT-H-14+laion2b_s32b_b79k": "imagenet_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_ViT-H-14laion2b_s32b_b79k_20250224-110743",
},
"imagenet200": {
    "ViT-B-32+openai": "imagenet200_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250220-193837",
    "ViT-L-14+openai": "imagenet200_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250220-165004",
    "ViT-H-14+laion2b_s32b_b79k": "imagenet200_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250221-081433",
    "ViT-B-16+openai": "imagenet200_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250222-171621"
},
"cifar100": {
    "ViT-B-32+openai": "cifar100_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250220-145402",
    "ViT-L-14+openai": "cifar100_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250220-150924",
    "ViT-H-14+laion2b_s32b_b79k": "cifar100_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250221-112741",
    "ViT-B-16+openai": "cifar100_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250222-211559"
},
"ooddb_dtd_0": {
    "ViT-B-32+openai": "ooddb_dtd_0_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250228-123248",
    "ViT-L-14+openai": "ooddb_dtd_0_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250228-123416",
    "ViT-H-14+laion2b_s32b_b79k": "ooddb_dtd_0_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250228-133806",
    "ViT-B-16+openai": "ooddb_dtd_0_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250228-123119",
},
"ooddb_patternnet_0": {
    "ViT-B-32+openai": "ooddb_patternnet_0_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250228-122411",
    "ViT-L-14+openai": "ooddb_patternnet_0_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250228-122741",
    "ViT-H-14+laion2b_s32b_b79k": "ooddb_patternnet_0_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250228-133431",
    "ViT-B-16+openai": "ooddb_patternnet_0_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250228-120037"
},
"cifar100n_noisyfine": {
    "ViT-B-32+openai": "cifar100n_noisyfine_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250303-113751",
    "ViT-B-16+openai": "cifar100n_noisyfine_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250303-114642",
    "ViT-L-14+openai": "cifar100n_noisyfine_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250303-120007",
    "ViT-H-14+laion2b_s32b_b79k": "cifar100n_noisyfine_none_224x224_e20_lr0.1_loss_cls_probe_mixnone_20250303-120750",
}
}


def get_classifier_model(id_name, classifier_variant, is_torchvision_ckpt=False, checkpoint_folder="checkpoints/", device='cuda'):
    checkpoint = classifiers[id_name][classifier_variant]
    assert id_name in DATA_INFO, f"Dataset {id_name} not found in DATA_INFO"
    assert id_name in classifiers, f"Dataset {id_name} not found in classifiers {classifiers}"
    assert classifier_variant in classifiers[id_name], f"Classifier variant {classifier_variant} not found in DATA_INFO for {id_name}"
    if is_torchvision_ckpt:
        checkpoint_path = f"{checkpoint_folder}/{checkpoint}"
        
        # if it doesn't exist, download it from https://download.pytorch.org/models/{checkpoint}
        if not glob.glob(checkpoint_path):
            import os
            from urllib.request import urlretrieve
            
            url = f"https://download.pytorch.org/models/{checkpoint}"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            urlretrieve(url, checkpoint_path)
        
        id_name = "imagenet"
        num_classes = DATA_INFO[id_name]['num_classes']
        if "resnet18" in checkpoint:
            backbone = ResNet18_224x224(num_classes=num_classes)
            backbone_type = "resnet"
        elif "resnet50" in checkpoint:
            backbone = ResNet50(num_classes=num_classes)
            backbone_type = "resnet"
        elif "vit_b_16" in checkpoint:
            backbone = ViT_B_16(num_classes=num_classes)
            backbone_type = "vit"
        else:
            raise NotImplementedError
        backbone.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)

        model = COOkeDNet(backbone=backbone, num_classes=num_classes, aux_dim=None, backbone_type=backbone_type, replace_fc=False).to(device)
    else:

        assert "loss_cls" in checkpoint, f"Checkpoint {checkpoint} is not a classifier checkpoint"
        assert id_name in DATA_INFO, f"Dataset {id_name} not found in DATA_INFO"
        assert id_name in checkpoint, f"Dataset {id_name} not found in checkpoint {checkpoint}"
        num_classes = DATA_INFO[id_name]['num_classes']

        checkpoint = get_checkpoint_path(checkpoint, checkpoint_folder)

        if "resnet18_224x224" in checkpoint:
            backbone = ResNet18_224x224(num_classes=num_classes)
            backbone_type = "resnet"
        elif "resnet18_32x32" in checkpoint:
            backbone = ResNet18_32x32(num_classes=num_classes)
            backbone_type = "resnet"
        elif "resnet50" in checkpoint:
            backbone = ResNet50(num_classes=num_classes)
            backbone_type = "resnet"
        elif "vit-b-16" in checkpoint:
            backbone = ViT_B_16(num_classes=num_classes)
            backbone_type = "vit"
        else:
            raise NotImplementedError
        
        model = COOkeDNet(backbone=backbone, num_classes=num_classes, aux_dim=None, backbone_type=backbone_type).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)

    return model

def get_probe_model(id_name, clip_variant, checkpoint_folder="checkpoints/",
                    device='cuda'):
    
    checkpoint = probes[id_name][clip_variant]

    assert "loss_cls_probe" in checkpoint, f"Checkpoint {checkpoint} is not a classifier checkpoint"
    assert id_name in DATA_INFO, f"Dataset {id_name} not found in DATA_INFO"
    assert id_name in checkpoint, f"Dataset {id_name} not found in checkpoint {checkpoint}"
    num_classes = DATA_INFO[id_name]['num_classes']

    clip_arch, clip_pretrained = clip_variant.split("+")
    assert clip_arch in clip_architecture_to_dim
    checkpoint = get_checkpoint_path(checkpoint, checkpoint_folder)

    clip_feat_dim = clip_architecture_to_dim[clip_arch]
    model = COOkeDNet(backbone=None, num_classes=num_classes, aux_dim=None, backbone_type="none", feature_size=clip_feat_dim).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)

    return model

def get_clip_model(clip_variant, device='cuda'):
    import open_clip
    clip_arch, clip_pretrained = clip_variant.split("+")
    assert clip_arch in clip_architecture_to_dim

    quickgelu=clip_architecture_to_quickgelu[clip_arch]
    secondary_net, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_pretrained, require_pretrained=True, force_quick_gelu=quickgelu, cache_dir="./cache/")
        
    logit_scale = secondary_net.logit_scale.exp().item()
    print(f"Logit scale: {logit_scale}")
    tokenizer = open_clip.get_tokenizer(clip_arch)

    return secondary_net.to(device), tokenizer, logit_scale

def get_checkpoint_path(checkpoint, checkpoint_folder="checkpoint_folder/"):
        # get exact checkpoint path
    if "ckpt" not in checkpoint:
        checkpoint_glob = f"{checkpoint_folder}/{checkpoint}*last_epoch*.ckpt"
        assert len(glob.glob(checkpoint_glob)) == 1, f"Found {len(glob.glob(checkpoint_glob))} checkpoints for {checkpoint_glob}"
        checkpoint = glob.glob(checkpoint_glob)[0]

    return checkpoint

class COOkeDNet(nn.Module):
    def __init__(self, backbone, num_classes, aux_dim=1, replace_fc=True, backbone_type="resnet", feature_size=None):
        super(COOkeDNet, self).__init__()

        self.backbone = backbone

        if backbone_type == "vit":
            feature_size = backbone.heads.head.in_features

            if replace_fc:
                self.fc = nn.Linear(feature_size, num_classes)
                if hasattr(self.backbone.heads, 'head'): self.backbone.heads.head = nn.Identity()
            else:
                self.fc = backbone.heads.head
                self.backbone.heads.head = nn.Identity()

        elif backbone_type == "resnet":

            # get the last layer and check .in_features
            feature_size = backbone.fc.in_features

            if replace_fc:
                self.fc = nn.Linear(feature_size, num_classes)
                if hasattr(self.backbone, 'fc'): self.backbone.fc = nn.Identity()
            else:
                self.fc = backbone.fc
                self.backbone.fc = nn.Identity()

        elif backbone_type == "none":
            assert feature_size is not None
            self.fc = nn.Linear(feature_size, num_classes)
            self.backbone = lambda x, return_feature: (None, x)

        if aux_dim is not None:
            if isinstance(aux_dim, int):
                self.aux_fc = nn.Linear(feature_size, aux_dim)
            elif isinstance(aux_dim, list):
                # define multiple layers with relu in between
                layers = []
                for i in range(len(aux_dim)):
                    layers.append(nn.Linear(feature_size, aux_dim[i]))
                    if i<len(aux_dim)-1: layers.append(nn.ReLU())
                self.aux_fc = nn.Sequential(*layers)
                print(self.aux_fc)
        else:
            self.aux_fc = nn.Identity()

    def forward(self, x, return_aux_logits=False, return_feature=False, **kwargs):
        _, feature = self.backbone(x, return_feature=True, **kwargs)
        
        logits = self.fc(feature)
        aux_logits = self.aux_fc(feature)

        if return_aux_logits and return_feature:
            return logits, aux_logits, feature
        elif return_aux_logits:
            return logits, aux_logits
        elif return_feature:
            return logits, feature
        else:
            return logits

    def forward_threshold(self, x, threshold):
        feature = self.backbone.forward_threshold(x, threshold)
        logits = self.fc(feature)
        return logits
    
    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()