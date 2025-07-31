from PIL import Image
import torch
from model_utils import get_classifier_model, get_clip_model, get_probe_model
from data_utils import preprocess_image_for_clip, preprocess_image_for_cls, get_label_to_class_mapping
import glob
# load trained models
device = "cuda" # or "cpu"
clip_variant = "ViT-B-16+openai" # or ViT-B-16+openai, ViT-L-14+openai, ViT-H-14+laion2b_s32b_b79k
classifier = get_classifier_model("imagenet","resnet18-ft", is_torchvision_ckpt=True, device=device)
probe = get_probe_model("imagenet", clip_variant, device=device)
clip, clip_tokenizer, clip_logit_scale = get_clip_model(clip_variant, device=device)

clip.eval() # pre-trained CLIP model from open_clip
probe.eval() # linear probe trained on CLIP image features from ID dataset
classifier.eval() # Resnet18 trained on ID dataset

# define ID classes and encode prompts
class_mapping = get_label_to_class_mapping("imagenet")
prompts = ["a photo of a [cls]".replace("[cls]",f"{class_mapping[idx]}") for idx in range(len(class_mapping))]
with torch.no_grad():
    prompt_features = clip.encode_text(clip_tokenizer(prompts).to(device))
    prompt_features_normed = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

image_paths = glob.glob("illustrations/*") 

ood_scoring = lambda softmax_probs: torch.distributions.Categorical(probs=softmax_probs).entropy().item() # entropy as OOD score
#ood_scoring = lambda softmax_probs: torch.max(softmax_probs, dim=1).values.item() # maximum softmax probability (MSP) as OOD score

for image_path in image_paths:
    print(f"---------------{image_path}-------------------")
    image = Image.open(image_path).convert("RGB")

    # note: different normalization for CLIP image encoder vs. standard classifier
    image_normalized_clip = preprocess_image_for_clip(image).to(device)
    image_normalized_cls = preprocess_image_for_cls(image).to(device)

    with torch.no_grad():
        # 1. get zero-shot CLIP prediction
        clip_image_features = clip.encode_image(image_normalized_clip)
        clip_image_features_normed = clip_image_features / clip_image_features.norm(dim=-1, keepdim=True)
        text_sim = (clip_image_features_normed @ prompt_features_normed.T)
        softmax_clip_t100 = (clip_logit_scale * text_sim).softmax(dim=1)

        # 2. get probe CLIP prediction
        softmax_probe = probe(clip_image_features).softmax(dim=1)

        # 3. get classifier prediction
        softmax_classifier = classifier(image_normalized_cls).softmax(dim=1)

    # 4. combined prediction
    softmax_ensemble = torch.stack([softmax_clip_t100, softmax_probe, softmax_classifier]).mean(0)

    # class prediction and OOD scores
    pred = softmax_ensemble.argmax(dim=1)
    ood_score = ood_scoring(softmax_ensemble)

    print("CLIP prediction:", class_mapping[softmax_clip_t100.argmax(dim=1).item()], f"(OOD score: {ood_scoring(softmax_clip_t100):.2f})")
    print("Probe prediction:", class_mapping[softmax_probe.argmax(dim=1).item()], f"(OOD score: {ood_scoring(softmax_probe):.2f})")
    print("Classifier prediction:", class_mapping[softmax_classifier.argmax(dim=1).item()], f"(OOD score: {ood_scoring(softmax_classifier):.2f})")
    print("---> COOkeD prediction:", class_mapping[pred.item()] , f"(OOD score: {ood_score:.2f})")
    
    print(f"--------------------------------------------------------------------------------------------------------------")