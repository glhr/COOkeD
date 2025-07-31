import torch
from tqdm import tqdm
import numpy as np
import pickle

from eval_utils import softmax_entropy, auroc_nancheck, fpr_and_fdr_at_recall
from data_utils import get_id_dataset, get_ood_dataset, get_csid_dataset, get_csid_dict, get_ood_dict, get_label_to_class_mapping, DATA_INFO, get_dataloader, get_preprocessor, get_img_normalizer
from model_utils import get_classifier_model, get_probe_model, get_clip_model, clip_architecture_to_dim, probes, classifiers

def ood_eval(test_dataloader, classifier=None, probe=None, clip=None, clip_prompt_templates_normed=None,
             clip_logit_scale=None, img_normalizers=None):
    if classifier is not None: classifier.eval()
    if probe is not None: probe.eval()
    if clip is not None: clip.eval()
    assert clip is not None or probe is not None or classifier is not None, f"At least one of classifier, probe or clip must be provided"

    with torch.no_grad():
        n_batches = 0

        pred_dicts = []
        conf_dicts = []
        labels_cls = []
        
        for batch in tqdm(test_dataloader, desc='Extracting preds from test: ', position=0, leave=True):
            n_batches += 1
            labels = batch['label'].cuda()
            labels_cls.append(labels)

            input_img_cls = img_normalizers["classifier"](batch['data'].cuda())
            input_img_clip = img_normalizers["clip"](batch['data'].cuda())

            conf_dict = {}
            pred_dict = {}

            if clip is not None:
                if 'clip_feat' in batch:
                    clip_feat = batch['clip_feat'].cuda()
                else:
                    clip_feat = clip.encode_image(input_img_clip)

                clip_feat_normed = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
                text_sim = (clip_feat_normed @ clip_prompt_templates_normed.T)
                        
                logits_clip_t100 = 100 * text_sim
                pred_clip = logits_clip_t100.argmax(dim=1)

                softmax_clip_t100 = logits_clip_t100.softmax(1)
                
                msp_clip_t100 = softmax_clip_t100.max(dim=1)[0]
                entropy_clip_t100 = softmax_entropy(softmax_clip_t100)


                if int(clip_logit_scale) != 100:
                    logits_clip_learned = clip_logit_scale * text_sim
                    softmax_clip_learned = logits_clip_learned.softmax(1)
                    msp_clip_learned = softmax_clip_learned.max(dim=1)[0]
                    entropy_clip_learned = softmax_entropy(softmax_clip_learned)
                    
                    conf_dict[f"msp_clip_{clip_logit_scale:.2f}"] = msp_clip_learned
                    conf_dict[f"entropy_clip_{clip_logit_scale:.2f}"] = entropy_clip_learned

                conf_dict["msp_clip_t100"] = msp_clip_t100
                conf_dict["entropy_clip_t100"] = entropy_clip_t100

                pred_dict["pred_clip"] = pred_clip

                if int(clip_logit_scale) != 100:
                    clip_zip = list(zip(
                                ["clip_t100", f"{clip_logit_scale:.2f}"],
                                [softmax_clip_t100, softmax_clip_learned],
                                [msp_clip_t100, msp_clip_learned],
                                [entropy_clip_t100, entropy_clip_learned],
                        ))
                else:
                    clip_zip = list(zip(
                                ["clip_t100"],
                                [softmax_clip_t100],
                                [msp_clip_t100],
                                [entropy_clip_t100],
                        ))

            if probe is not None:
                logits_probe = probe(clip_feat)
                softmax_probe = logits_probe.softmax(dim=1)
                msp_probe, pred_probe = softmax_probe.max(dim=1)
                entropy_probe = softmax_entropy(softmax_probe)

                conf_dict["msp_probe"] = msp_probe
                conf_dict["entropy_probe"] = entropy_probe
                pred_dict["pred_probe"] = pred_probe

            if classifier is not None:
                logits_classifier = classifier(input_img_cls)
                pred_cls = logits_classifier.argmax(dim=1)
                softmax_classifier = logits_classifier.softmax(dim=1)
                msp_classifier, _ = softmax_classifier.max(dim=1)
                entropy_classifier = softmax_entropy(softmax_classifier)

                conf_dict["msp_classifier"] = msp_classifier
                conf_dict["entropy_classifier"] = entropy_classifier
                pred_dict["pred_classifier"] = pred_cls

            assert classifier is not None and probe is not None and clip is not None

            if classifier is not None and probe is not None and clip is not None:

                for temp_str, softmax_clip_t, msp_clip_t, entropy_clip_t in clip_zip:
                    
                    softmax_avg_classifier_probe_clip_t = torch.stack([softmax_classifier, softmax_probe, softmax_clip_t], dim=0).mean(0)

                    msp_avg_classifier_probe_clip_t, pred_avg_classifier_probe_clip_t = softmax_avg_classifier_probe_clip_t.max(1)

                    entropy_avg_classifier_probe_clip_t = softmax_entropy(softmax_avg_classifier_probe_clip_t)

                    conf_dict[f"msp_avg_classifier+probe+{temp_str}"] = msp_avg_classifier_probe_clip_t
                    pred_dict[f"pred_avg_classifier+probe+{temp_str}"] = pred_avg_classifier_probe_clip_t
                    conf_dict[f"entropy_avg_classifier+probe+{temp_str}"] = entropy_avg_classifier_probe_clip_t
                    
            pred_dicts.append(pred_dict)
            conf_dicts.append(conf_dict)

        pred_dict = {}
        conf_dict = {}
        if len(pred_dicts):
            # merge dicts into one
            for key in pred_dicts[0].keys():
                pred_dict[key] = torch.cat([d[key] for d in pred_dicts]).cpu()
            for key in conf_dicts[0].keys():
                conf_dict[key] = torch.cat([d[key] for d in conf_dicts]).cpu()

        labels_cls = torch.cat(labels_cls).cpu()

        return pred_dict, conf_dict, labels_cls
    
if __name__ == "__main__":
    import argparse
    import os

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_mode', default='standard', type=str, help='webdataset or standard. standard will load image files directly, webdataset will load pre-extracted CLIP image features.')
    parser.add_argument('--preprocessor_eval', default='centercrop', type=str, choices=['centercrop', 'resize'], help='preprocessor for evaluation')
    parser.add_argument('--id_name', default="cifar100n_noisyfine", type=str, choices=DATA_INFO.keys(), help='ID dataset name')
    parser.add_argument('--classifier_variant', default="resnet18-ft", type=str, help='resnet18-ft')
    parser.add_argument('--clip_variant', default="ViT-B-16+openai", type=str, choices=probes["imagenet"].keys(), help='clip architecture and pretrained variant, e.g. ViT-B-16+openai')
    parser.add_argument('--prompt_template', default="a photo of a [cls]", type=str, help='something [cls].')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing results')
    parser.add_argument('--csid', action='store_true', help='evaluate on covariate shifted ID dataset')
    parser.add_argument('--save_pkl', action='store_true', help='save pkl with preds and confs in results/pkls/')
    parser.add_argument('--count_params', action='store_true', help='print number of parameters')
    args = parser.parse_args()

    args.image_size = 224
    args.num_classes = DATA_INFO[args.id_name]['num_classes']

    args.clip_architecture, args.clip_pretrained = args.clip_variant.split("+")
    assert args.clip_architecture in clip_architecture_to_dim
    args.clip_dim = clip_architecture_to_dim[args.clip_architecture]

    # check if results already exist
    eval_folder = "results/summary_csvs/"
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)
    pkl_folder = "results/pkls/"
    if not os.path.exists(pkl_folder):
        os.makedirs(pkl_folder)

    csv_path = f"{eval_folder}{args.id_name}_{args.classifier_variant}_{args.clip_variant}_{args.prompt_template}.csv"

    if args.csid:
        csv_path = csv_path.replace(".csv", f"_csid.csv")

    if not args.save_pkl and (os.path.exists(csv_path) and not args.overwrite):
        print(f"Results already exist at {csv_path}. Exiting...")
        exit()
    elif args.save_pkl and (os.path.exists(csv_path.replace(".csv", "_id.pkl").replace(eval_folder, pkl_folder)) and not args.overwrite):
        print(f"Results already exist at {csv_path.replace('.csv', '_id.pkl').replace(eval_folder, pkl_folder)}. Exiting...")
        exit()

    if args.id_name in classifiers:
        classifier = get_classifier_model(args.id_name, args.classifier_variant, is_torchvision_ckpt=args.id_name=="imagenet")
    else:
        classifier = None
        print(f"No known classifier for {args.id_name}")

    if args.id_name in probes:
        probe = get_probe_model(args.id_name, clip_variant=args.clip_variant)
    else:
        probe = None
        print(f"No known probe for {args.id_name} {args.clip_variant}")
        #raise Exception
    
    clip, clip_tokenizer, clip_logit_scale = get_clip_model(args.clip_variant)
    
    preprocessor_eval = get_preprocessor(crop_mode=args.preprocessor_eval, image_size=args.image_size, norm_params=None)
    img_normalizers = { # important: CLIP image encoder and classifier were trained with different normalization
        "classifier": get_img_normalizer(setting=args.id_name),
        "clip": get_img_normalizer(setting="clip")
    }

    eval_loader_kwargs = {
            'batch_size': 128,
            'shuffle': False,
            'num_workers': 4,
            'drop_last': False,
            'pin_memory': True
        }


    test_data = get_id_dataset(args.id_name, 'test', preprocessor=preprocessor_eval, args=args)
    test_loader = get_dataloader(test_data, eval_loader_kwargs)
                                
    with torch.no_grad():
        label_to_class_mapping = get_label_to_class_mapping(args.id_name)
        prompt_template = args.prompt_template
        prompts = [prompt_template.replace("[cls]",f"{label_to_class_mapping[label].replace('_',' ')}")  for label in range(len(label_to_class_mapping))]
        assert len(prompts) == args.num_classes
        #print(prompts)
        # get logits
        prompt_tokenized = clip_tokenizer(prompts).cuda()
        prompt_features = clip.encode_text(prompt_tokenized)
        prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

    print("Running evaluation on ID dataset...")
    pred_dict_id, conf_dict_id, cls_labels_id = ood_eval(test_loader, classifier=classifier, probe=probe,
                                                        clip=clip, clip_prompt_templates_normed=prompt_features,
                                                        clip_logit_scale=clip_logit_scale, img_normalizers=img_normalizers)
    

    if args.save_pkl:
        pkl_path = csv_path.replace(".csv", "_id.pkl").replace(eval_folder, pkl_folder)
        with open(pkl_path, 'wb') as f:
            pickle.dump({"pred_dict":pred_dict_id, "conf_dict":conf_dict_id, "cls_labels":cls_labels_id}, f)

    if args.csid:
        print("Running evaluation on covariate shifted ID dataset...")
        csid_datasets = get_csid_dict(args.id_name)
        for csid_name, _ in csid_datasets.items():

            csid_data = get_csid_dataset(args.id_name, csid_name, preprocessor_eval, args=args)
            csid_loader = get_dataloader(csid_data, eval_loader_kwargs)
            
            pred_dict_csid, conf_dict_csid, cls_labels_csid = ood_eval(csid_loader, classifier=classifier, probe=probe,
                                                    clip=clip, clip_prompt_templates_normed=prompt_features,
                                                    clip_logit_scale=clip_logit_scale, img_normalizers=img_normalizers)

            # merge dicts with id
            for key in pred_dict_csid.keys():
                pred_dict_id[key] = torch.cat([pred_dict_id[key], pred_dict_csid[key]])
            for key in conf_dict_csid.keys():
                conf_dict_id[key] = torch.cat([conf_dict_id[key], conf_dict_csid[key]])
            cls_labels_id = torch.cat([cls_labels_id, cls_labels_csid])

        if args.save_pkl:
            pkl_path = csv_path.replace(".csv", "_csid.pkl").replace(eval_folder, pkl_folder)
            with open(pkl_path, 'wb') as f:
                pickle.dump({"pred_dict":pred_dict_id, "conf_dict":conf_dict_id, "cls_labels":cls_labels_id}, f)
    #print(pred_dict_id)
    #print(conf_dict_id)

    ood_datasets = get_ood_dict(args.id_name)

    perf_dict = {}

    for key, value in pred_dict_id.items():
        iscorrect = value.eq(cls_labels_id).float()
        accuracy = iscorrect.mean().item()
        if "avg" in key:
            print(f"Accuracy COOkeD: {accuracy*100:.2f}%")
        else:
            print(f"Accuracy {key}: {accuracy*100:.2f}%")
        perf_dict[f"classification/{key}/test_accuracy"] = accuracy

    for ood_name, ood_type in ood_datasets.items():
        print(f"Running evaluation on OOD dataset {ood_name} ({ood_type})...")
        ood_data = get_ood_dataset(args.id_name, ood_name, preprocessor_eval, split=ood_type, args=args)
        ood_loader = get_dataloader(ood_data, eval_loader_kwargs)
        
        pred_dict_ood, conf_dict_ood, cls_labels_ood = ood_eval(ood_loader, classifier=classifier, probe=probe,
                                                clip=clip, clip_prompt_templates_normed=prompt_features,
                                                clip_logit_scale=clip_logit_scale, img_normalizers=img_normalizers)
        

        if args.save_pkl:
            pkl_path = csv_path.replace(".csv", f"_{ood_type}ood_{ood_name}.pkl").replace(eval_folder, pkl_folder)
            with open(pkl_path, 'wb') as f:
                pickle.dump({"pred_dict":pred_dict_ood, "conf_dict":conf_dict_ood, "cls_labels":cls_labels_ood}, f)


        if len(conf_dict_ood):
            for key, value in conf_dict_ood.items():
                mult = -1 if ("msp" in key) else 1
                conf_id = mult*conf_dict_id[key]
                conf_ood = mult*value
                labels = torch.cat([torch.zeros_like(conf_id), torch.ones_like(conf_ood)])
                confs = torch.cat([conf_id, conf_ood])
                auroc = auroc_nancheck(labels, confs)
                fpr95 = fpr_and_fdr_at_recall(1-labels.numpy(), -confs.numpy(), recall_level=0.95)
                try:
                    perf_dict[f"ood_ablation/{ood_type}/{key}/auroc"].append(auroc)
                    perf_dict[f"ood_ablation/{ood_type}/{key}/fpr95"].append(fpr95)
                except KeyError:
                    perf_dict[f"ood_ablation/{ood_type}/{key}/auroc"] = [auroc]
                    perf_dict[f"ood_ablation/{ood_type}/{key}/fpr95"] = [fpr95]

                perf_dict[f"ood_ablation/{ood_name}/{key}/auroc"] = auroc
                perf_dict[f"ood_ablation/{ood_name}/{key}/fpr95"] = fpr95

                if key == "entropy_avg_classifier+probe+clip_t100":
                    print(f"COOkeD OOD detection: {ood_name} AUROC: {auroc*100:.2f}%")

    for key in perf_dict:
        try:
            perf_dict[key] = np.mean(perf_dict[key])
        except:
            perf_dict[key] = None
            print(f"Failed to compute mean for {key}")

    for key in conf_dict_ood.keys():
        # key format: f"/oodood_ablation_type/ce/auroc"
        for metric in ["auroc","fpr95"]:
            near_perf = perf_dict[f"ood_ablation/near/{key}/{metric}"]
            far_perf = perf_dict[f"ood_ablation/far/{key}/{metric}"]
            assert isinstance(near_perf, float) or near_perf is None
            assert isinstance(far_perf, float) or far_perf is None
            #print(key,[near_perf, far_perf])
            if near_perf is not None and far_perf is not None:
                try:
                    avg_perf = np.mean([near_perf, far_perf])
                except Exception as e:
                    print(near_perf, far_perf)
                    raise e
            else:
                avg_perf = None
            perf_dict[f"ood_ablation/avg/{key}/{metric}"] = avg_perf
            
            if key == "entropy_avg_classifier+probe+clip_t100" and metric == "auroc":
                print(f"COOkeD OOD detection: {metric} near: {near_perf*100:.2f}% far: {far_perf*100:.2f}% avg: {avg_perf*100:.2f}%")

    for arg in vars(args):
        perf_dict[arg] = getattr(args, arg)

    # save dict as pandas dataframe
    import pandas as pd
    #print(perf_dict)
    df = pd.DataFrame(perf_dict.items(), columns=["metric", "value"])
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")