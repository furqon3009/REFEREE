import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from classifier import Classifier
from config import *
from data import get_subset_office31, get_multitask_experiment, get_subset_officeHome, get_subset_visda, get_subset_domainnet
from tqdm import tqdm
import random
import copy
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms.functional import to_pil_image
from utils import denorm_for_clip
from clip_labels import get_clip_class_names, build_clip_prompts, template_for_domain


def seed_everything(seed=0):
    
    cudnn_deterministic = True
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def compute_accuracy(model, test_loader, device):
    """
    Computes the accuracy for the model on the test dataset.
    Args:
        model (Classifier): The classifier model to evaluate.
        test_df (pd.DataFrame): DataFrame containing the test dataset.
    Returns:
        float: The accuracy of the model on the test dataset.
    """
    model.backbone.eval()
    stat = 'test'
    X_test, y_test = model.get_embeddings(test_loader, stat)

    # Predict one sample at a time to avoid dimension mismatch
    y_pred = []
    for xi in tqdm(X_test, desc="Predicting"):
        xi_tensor = torch.from_numpy(xi).float().to(device)
        pred = model.predict(xi_tensor)
        # pred might be a numpy array or scalar
        if isinstance(pred, np.ndarray):
            y_pred.append(pred.item())
        else:
            y_pred.append(pred)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_pred == y_test)

    return accuracy

def compute_accuracy_dual_cil(
    model,
    test_loader,
    device,
    clip_model,
    clip_processor,
    class_names,
):
    """
    Dual-branch evaluation in CIL mode (classes_seen only).
    - Fuses ViT-KLDA and CLIP per sample with label-free, confidence-based weights.
    - No test-label tuning/leakage for alpha/beta.
    - Assumes KLDA classes align with integer indices into `class_names`.

    Args:
        model: your target classifier with `backbone` and `model.models[0]` = KLDA
        test_loader: DataLoader yielding (images, labels)
        device: torch.device
        clip_model: CLIP model (eval/frozen)
        clip_processor: CLIP processor
        class_names: list[str], global class name list; indices must match labels

    Returns:
        float: accuracy in [0,1]
    """
    model.backbone.eval()
    clip_model.eval()

    # Build CLIP text features once for all classes
    with torch.no_grad():
        text_inputs = clip_processor(
            text=[f"a photo of a {w}" for w in class_names],
            return_tensors="pt",
            padding=True,
        ).to(device)
        text_feats = F.normalize(clip_model.get_text_features(**text_inputs), dim=1)  # [C_all, D]

    # CIL: restrict scoring to classes that KLDA has seen so far
    class_list = sorted(model.model.models[0].classes_seen)
    if len(class_list) == 0:
        print("[Dual CIL] No classes_seen in KLDA; returning 0.0 accuracy.")
        return 0.0

    all_targets, all_preds = [], []

    with torch.no_grad():
        for imgs, y in test_loader:
            all_targets.extend(y.numpy().tolist())
            imgs = imgs.to(device, non_blocking=True)

            # ViT features (target model backbone)
            vit_feats = F.normalize(model.backbone(imgs), p=2, dim=1)  # [B, d]
            pil_batch = [to_pil_image(denorm_for_clip(img).cpu()) for img in imgs]
            img_inputs = clip_processor(images=pil_batch, return_tensors="pt", padding=True).to(device)
            img_feats = F.normalize(clip_model.get_image_features(**img_inputs), dim=1)  # [B, D]

            # Per sample fusion over classes_seen
            klda0 = model.model.models[0]  # primary KLDA
            for i in range(imgs.size(0)):
                # KLDA probabilities over classes_seen
                p_k = F.softmax(klda0.get_logits(vit_feats[i]), dim=0)  # [K_seen]

                # CLIP probabilities over ALL classes, then reindex to classes_seen
                p_c_all = F.softmax(img_feats[i] @ text_feats.T, dim=0)  # [C_all]
                p_c = torch.stack([p_c_all[c] for c in class_list])      # [K_seen]

                # Label-free confidence fusion weights (per sample)
                ck = float(p_k.max().item())
                cc = float(p_c.max().item())
                a = ck / (ck + cc + 1e-8)
                b = 1.0 - a

                # Fuse and pick argmax over classes_seen
                p = a * p_k + b * p_c
                pred_idx = torch.argmax(p).item()
                all_preds.append(class_list[pred_idx])

    acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    print(f"[Dual CIL] Accuracy (confidence-weighted fusion): {acc*100:.2f}%")
    return acc

def save_results(dataset_name, model_name, seed, accuracies, sigma=None, D=None, final_avg=None, forgetting=None, mode='append'):
    """
    Save results to a file in the desired format.
    """
    results_dir = os.path.join("results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    file_name = model_name.replace("/", "-")
    results_file = os.path.join(results_dir, f"{file_name}.txt")

    # If mode is 'init', overwrite file and write sigma & D
    if mode == 'init':
        with open(results_file, "a") as f:
            f.write(f"sigma: {sigma}\n")
            f.write(f"D: {D}\n")

    # If mode is 'task', append per-task accuracies
    elif mode == 'task':
        with open(results_file, "a") as f:
            acc_list_str = ", ".join([f"{x:.4f}" for x in accuracies])
            f.write(f"Seed: {seed}, Accuracies: [{acc_list_str}]\n")

    # If mode is 'final', append final summary
    elif mode == 'final':
        with open(results_file, "a") as f:
            f.write(f"Final Average Accuracy: {final_avg * 100:.2f}%\n")
            f.write(f"Forgetting per task: {forgetting[1]}\n")
            f.write(f"Average Forgetting: {forgetting[0]:.4f}\n")
            f.write(f" ")


def compute_forgetting(results):
    """
    Compute average forgetting across tasks.
    results[i, j] = accuracy on task j after training on task i.
    """
    num_tasks = results.shape[0]
    forgetting = []

    for t in range(num_tasks - 1):  # exclude last task
        max_acc = np.max(results[: , t])   # best accuracy on task t
        final_acc = results[-1, t]         # accuracy after all tasks
        forgetting.append(max_acc - final_acc)

    return np.mean(forgetting), forgetting


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VLM Branch
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # print(f'DatasetsType.office31:{DatasetsType.office31}')

    if args.source == DatasetsType.office31:
        src_list = Office31_src
        tgt_list = Office31_tgt
    elif args.source == DatasetsType.officehome:
        src_list = OfficeHome_src
        tgt_list = OfficeHome_tgt
    elif args.source == DatasetsType.mnist or args.source == DatasetsType.usps:
        src_list = [args.source]
        tgt_list = [args.target]
    elif args.source == DatasetsType.visda:
        src_list = [args.source]
        tgt_list = [args.target]
    elif args.source == DatasetsType.domainnet:
        src_list = DomainNet_src
        tgt_list = DomainNet_tgt

    for s,t in zip(src_list, tgt_list):

        # Load dataset
        seed_everything(args.seed)
        print('Preparing the data...')
        if args.source == DatasetsType.office31:
            split_src = get_subset_office31(s, args.scenario)
            src_tr = split_src.train_datasets
            src_te = split_src.test_datasets
            split_tgt = get_subset_office31(t, args.scenario)
            tgt_tr = split_tgt.train_datasets
            tgt_te = split_tgt.test_datasets

            config = {'size': 224, 'channels': 3, 'classes': 31}
            classes_per_task = [6,6,6,6,7]
            args.tasks = 5
            transform = src_tr[0].transforms
        elif args.source == DatasetsType.officehome:
            print(f'{args.source}')
            split_src = get_subset_officeHome(s, args.scenario)
            src_tr = split_src.train_datasets
            src_te = split_src.test_datasets
            split_tgt = get_subset_officeHome(t, args.scenario)
            tgt_tr = split_tgt.train_datasets
            tgt_te = split_tgt.test_datasets

            config = {'size': 224, 'channels': 3, 'classes': 65}
            classes_per_task = 5
            args.tasks = 13
            transform = src_tr[0].transforms
        elif args.source == "MNIST" or args.source == "USPS":
            (src_tr, src_te), config, classes_per_task, transform, permutation = get_multitask_experiment(
                    dataset_name=args.source, scenario=args.scenario, tasks=args.tasks, data_dir=args.d_dir,    
                    verbose=True, exception=True if args.seed==0 else False, permutation=None)
            (tgt_tr, tgt_te), config, classes_per_task, transform, permutation = get_multitask_experiment(
                    dataset_name=args.target, scenario=args.scenario, tasks=args.tasks, data_dir=args.d_dir,    
                    verbose=True, exception=True if args.seed==0 else False, permutation=permutation)
        elif args.source.lower() == "visda":
            split_src = get_subset_visda("train", args.scenario)
            src_tr = split_src.train_datasets
            src_te = split_src.test_datasets
            split_tgt = get_subset_visda("validation", args.scenario)
            tgt_tr = split_tgt.train_datasets
            tgt_te = split_tgt.test_datasets

            config = {'size': 224, 'channels': 3, 'classes': 12}
            classes_per_task = 3
            args.tasks = 4
            transform = src_tr[0].transforms
        elif args.source.lower() == "domainnet":
            src_split = get_subset_domainnet(s, args.scenario)
            src_tr = src_split.train_datasets
            src_te = src_split.test_datasets
            tgt_split = get_subset_domainnet(t, args.scenario)
            tgt_tr = tgt_split.train_datasets
            tgt_te = tgt_split.test_datasets

            config = {'size': 224, 'channels': 3, 'classes': 345}
            classes_per_task = 23
            args.tasks = 15
            transform = src_tr[0].transforms

        # CLIP prompt labels 
        order_ref = src_tr[0] if hasattr(src_tr[0], "class_to_idx") else None
        class_names = get_clip_class_names(args.source, dataset_for_order=order_ref)

        # domain-aware CLIP templates for DomainNet
        if str(args.source).lower() == "domainnet":
            tmpl_src = template_for_domain(s) 
            tmpl_tgt = template_for_domain(t)
        else:
            tmpl_src = tmpl_tgt = "a photo of a {}"

        clip_text_prompts = build_clip_prompts(class_names, template=tmpl_tgt)
        
        src_train_loader2 = []
        src_test_loader2 = []
        tgt_train_loader2 = []
        tgt_test_loader2 = []
        for i in range(args.tasks):
            src_train_loader2.append(DataLoader(src_tr[i], batch_size=args.batch_size, shuffle=False, num_workers=4))
            src_test_loader2.append(DataLoader(src_te[i], batch_size=args.batch_size, shuffle=False, num_workers=4))
            tgt_train_loader2.append(DataLoader(tgt_tr[i], batch_size=args.batch_size, shuffle=False, num_workers=4))
            tgt_test_loader2.append(DataLoader(tgt_te[i], batch_size=args.batch_size, shuffle=False, num_workers=4))

            # Number of samples per task
            print(f"[Task {i}] Source train samples: {len(src_tr[i])}, Target train samples: {len(tgt_tr[i])}")

        # create models
        model_src = Classifier(device, D=args.D, sigma=args.sigma, num_ensembles=args.num_ensembles, seed=args.seed, model_name=args.model_name)
        model_tgt = Classifier(device, D=args.D, sigma=args.sigma, num_ensembles=args.num_ensembles, seed=args.seed, model_name=args.model_name)

        results = np.zeros((args.tasks, args.tasks))
        results_tgt = np.zeros((args.tasks, args.tasks))

        save_results(args.source, args.model_name, args.seed,
                     accuracies=[], sigma=args.sigma, D=args.D, mode='init')

        for i in range(args.tasks):
            print(f"\n--- Training on Task {i} ---")
            # source domain training
            model_src.fit(src_train_loader2[i])

            # Warm-start target KLDA for the new classes of task i 
            model_tgt._warm_start_new_classes_from_source(
                model_src=model_src,
                task_id=i,
                classes_per_task=classes_per_task
            )

            # Adapt target model 
            model_tgt.fit_target_dual_weighted(
                tgt_loader=tgt_train_loader2[i],
                model_src=model_src,
                clip_model=clip_model,
                clip_processor=clip_processor,
                class_names=class_names,
                task_id=i,
                classes_per_task=classes_per_task,
                use_freq_aug=True
            )
            
            for j in range(i+1):
                # Source evaluation (CL)
                acc_src = compute_accuracy(model_src, src_test_loader2[j], device)
                results[i, j] = acc_src
                print(f"Source Accuracy on Task {j}: {acc_src * 100:.2f}%")

                # evaluate target (ViT + CLIP) --> CIL
                acc_tgt_dual = compute_accuracy_dual_cil(
                    model=model_tgt, 
                    test_loader=tgt_test_loader2[j], 
                    device=device, 
                    clip_model=clip_model, 
                    clip_processor=clip_processor, 
                    class_names=class_names
                )
                results_tgt[i, j] = acc_tgt_dual
                print(f"Target Accuracy on Task {j} (ViT+CLIP): {acc_tgt_dual * 100:.2f}%")

            # Save results for current task
            save_results(args.source, args.model_name, args.seed,
                         accuracies=results_tgt[i, :], mode='task')

        print("\nFinal Source Accuracy Matrix:\n", results)
        print("\nFinal Target Accuracy Matrix:\n", results_tgt)

        final_avg_accuracy_src = np.mean(results[-1, :])
        final_avg_accuracy_tgt = np.mean(results_tgt[-1, :])
        print(f"\nFinal Average Source Accuracy: {final_avg_accuracy_src * 100:.2f}%")
        print(f"Final Average Target Accuracy: {final_avg_accuracy_tgt * 100:.2f}%")
        # Compute forgetting
        avg_forgetting_src, forgetting_per_task_src = compute_forgetting(results)
        avg_forgetting_tgt, forgetting_per_task_tgt = compute_forgetting(results_tgt)
        print(f"\nSource Forgetting per task: {forgetting_per_task_src}")
        print(f"Target Forgetting per task: {forgetting_per_task_tgt}")
        print(f"Average Source Forgetting: {avg_forgetting_src:.4f}")
        print(f"Average Target Forgetting: {avg_forgetting_tgt:.4f}")

        # Save final summary
        save_results(args.source, args.model_name, args.seed,
                     accuracies=[], final_avg=final_avg_accuracy_tgt,
                     forgetting=(avg_forgetting_tgt, forgetting_per_task_tgt), mode='final')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help="path to download/load MNIST data")
    parser.add_argument('--save_dir', default='experiments', type=str,
                        help='Directory containing all experiments')
    parser.add_argument('--batch_size', type=int, default=128, help="batch size for embedding extraction")
    parser.add_argument('--num_workers', type=int, default=4, help="number of dataloader workers")
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help="timm ViT model name")
    parser.add_argument('--D', type=int, default=6000, help="RFF feature dimension for KLDA")
    parser.add_argument('--sigma', type=float, default=1e-4, help="bandwidth for RBF kernel")
    parser.add_argument('--num_ensembles', type=int, default=1, help="number of ensemble models (use 1 for single) ")
    parser.add_argument('--runs', type=int, default=3, help='how often to repeat?')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--dataset_name', type=str, default='OfficeHome', help="Name of the dataset to use (MNIST, USPS, Office31, OfficeHome).")
    parser.add_argument('--source', type=str, default='VisDA', help="Name of the dataset to use (MNIST, USPS, Office31, OfficeHome, VisDA, DomainNet).")
    parser.add_argument('--target', type=str, default='VisDA', help="Name of the dataset to use (MNIST, USPS, Office31, OfficeHome, VisDA, DomainNet).")
    parser.add_argument('--scenario', type=str, default='class', choices=['task', 'class', 'domain'], help="only class incremental is available now")
    parser.add_argument('--tasks', default=5, type=int, help='number of tasks')
    parser.add_argument('--data-dir', type=str, default='./data', dest='d_dir', help='default: %(default)s')
    args = parser.parse_args()
    base_seed = args.seed
    if args.runs <= 1:
        main(args)
    else:
        for run_idx in range(args.runs):
            args.seed = base_seed + run_idx
            print(f"\n========== Run {run_idx+1}/{args.runs} (seed={args.seed}) ==========")
            main(args)
