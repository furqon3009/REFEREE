import numpy as np
import torch
import torch.nn.functional as F
import timm
from KLDA import KLDA_E
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import os
import math
from utils import denorm_for_clip
import pywt 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Classifier:
    def __init__(self, device, D, sigma, num_ensembles, seed=0, model_name='vit_base_patch16_224'):
        """
        Args:
            D (int): Dimension of Random Fourier Features (RFF).
            sigma (float): Bandwidth parameter for the RBF kernel.
            num_ensembles (int): Number of models in the ensemble.
            seed (int, optional): Seed for reproducibility. Defaults to 0.
            model_name (str, optional): Pre-trained model name for SentenceTransformer. Defaults to 'facebook/bart-base'.
        """
        self.device = device
        self.backbone = timm.create_model(model_name, pretrained=True)
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.reset_classifier(0)
        self.backbone.eval()
        self.backbone = self.backbone.to(device)
        
        self.D = D
        self.sigma = sigma
        self.num_ensembles = num_ensembles
        self.seed = seed
        self.model = None
        self.labels = []

        # For (de)normalizing images around DWT
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        self.img_std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)

    # ---------- FREQ-AUG HELPERS ----------
    def _denorm01(self, x):
        """x: float tensor in ImageNet norm -> clamp to [0,1]."""
        x = x * self.img_std + self.img_mean
        return torch.clamp(x, 0.0, 1.0)

    def _renorm(self, x01):
        """x01: float tensor [0,1] -> ImageNet norm."""
        x = (x01 - self.img_mean) / self.img_std
        return x

    def _dwt_replace_hf_single(self, img_chw, wavelet='haar', mode='zeros'):
        """
        img_chw: (3,H,W) tensor in [0,1]
        mode: 'zeros' or 'randn'
        """
        c, h, w = img_chw.shape
        out = []
        for ci in range(c):
            a = img_chw[ci].detach().cpu().numpy()
            LL, (LH, HL, HH) = pywt.dwt2(a, wavelet=wavelet)

            if mode == 'zeros':
                LHm = np.zeros_like(LH); HLm = np.zeros_like(HL); HHm = np.zeros_like(HH)
            else:  # 'randn'
                LHm = np.random.randn(*LH.shape).astype(a.dtype)
                HLm = np.random.randn(*HL.shape).astype(a.dtype)
                HHm = np.random.randn(*HH.shape).astype(a.dtype)

            rec = pywt.idwt2((LL, (LHm, HLm, HHm)), wavelet=wavelet)
            # PyWavelets preserves shape for even sizes; clip & pad/crop as safety
            rec = np.clip(rec, 0.0, 1.0)
            if rec.shape != a.shape:
                # center-crop or pad to match (rare)
                hh, ww = rec.shape
                rec_t = torch.from_numpy(rec)
                rec_t = rec_t[:h, :w]
                pad_h = max(0, h - rec_t.shape[0]); pad_w = max(0, w - rec_t.shape[1])
                rec_t = torch.nn.functional.pad(rec_t, (0,pad_w,0,pad_h), mode='replicate')
            else:
                rec_t = torch.from_numpy(rec)

            out.append(rec_t)
        out = torch.stack(out, dim=0)  # (3,H,W)
        return out

    def _freq_augment_batch(self, images, wavelet='haar', modes=('zeros','randn'), prob=1.0):
        """
        images: (B,3,H,W) normalized; returns list [aug_zeros, aug_randn] (if prob triggers).
        """
        if prob <= 0.0 or not modes:
            return []
        with torch.no_grad():
            x01 = self._denorm01(images.clone())
            outs = []
            for m in modes:
                augs = []
                for i in range(x01.size(0)):
                    if torch.rand(1, device=self.device).item() <= prob:
                        aug = self._dwt_replace_hf_single(x01[i], wavelet=wavelet, mode=m).to(self.device)
                    else:
                        aug = x01[i]  # pass-through
                    augs.append(aug)
                augs = torch.stack(augs, dim=0)  # (B,3,H,W) in [0,1]
                outs.append(self._renorm(augs))
            return outs  # list of tensors, each (B,3,H,W)

    def _ensure_klda_from_source(self, model_src):
        """Create an empty KLDA_E in the target model, mirroring the source, if needed."""
        if getattr(self, "model", None) is not None and self.model is not None:
            return  # already initialized

        # Mirror source ensemble shape & hyperparams
        src_e = model_src.model
        src0 = src_e.models[0]
        num_ensembles = len(src_e.models)

        self.model = KLDA_E(
            num_classes=src0.num_classes,
            d=src0.d,                  # backbone feature dim
            D=src0.D,
            sigma=src0.sigma,
            num_ensembles=num_ensembles,
            seed=self.seed,
            device=self.device
        )

    def _warm_start_new_classes_from_source(self, model_src, task_id, classes_per_task):
        """Copy source class means/counts for the new classes of task_id into the target KLDA."""
        self._ensure_klda_from_source(model_src)

        # Compute which classes are new for this task
        if isinstance(classes_per_task, int):
            start = task_id * classes_per_task
            new_classes = list(range(start, start + classes_per_task))
        else:
            start = sum(classes_per_task[:task_id])
            new_classes = list(range(start, start + classes_per_task[task_id]))

        copied_any = False
        for tgt_m, src_m in zip(self.model.models, model_src.model.models):
            copied = []
            for c in new_classes:
                if c in src_m.class_means:
                    tgt_m.class_means[c] = src_m.class_means[c].clone()
                    # keep a reasonable (non-zero) count so incremental averaging works
                    tgt_m.class_counts[c] = 25.0  # small prior so target updates have effect
                    if c not in tgt_m.classes_seen:
                        tgt_m.classes_seen.append(c)
                    copied.append(c)
            # rebuild matrices so get_logits works immediately
            if tgt_m.classes_seen:
                tgt_m.class_mean_matrix = torch.stack(
                    [tgt_m.class_means[c] for c in sorted(tgt_m.classes_seen)]
                ).to(tgt_m.device)
                if tgt_m.sigma_inv is None:
                    tgt_m.sigma_inv = torch.pinverse(
                        tgt_m.sigma_matrix + 1e-6 * torch.eye(tgt_m.D, device=tgt_m.device)
                    )
            if copied:
                copied_any = True

        if copied_any:
            print(f"[WarmStart] Copied source means for task {task_id} classes: {new_classes}")
        else:
            print(f"[WarmStart] No source means found for task {task_id} classes: {new_classes}")


    def fit(self, train_loader):
        """
        Trains the ensemble model using the provided dataframe.
        Args:
            df (pd.DataFrame): DataFrame containing 'text' and 'label' columns.
        """
        
        stat = 'train'
        X_train, y_train = self.get_embeddings(train_loader, stat)
        num_classes = len(np.unique(y_train))
        feature_dim = X_train.shape[1]

        # Initialize KLDA_E only if it doesn't exist
        if self.model is None:
            self.model = KLDA_E(
                num_classes=num_classes,
                d=feature_dim,
                D=self.D,
                sigma=self.sigma,
                num_ensembles=self.num_ensembles,
                seed=self.seed,
                device=self.device
            )
        
        # Convert data to torch tensors
        X_tensor = torch.from_numpy(X_train).float().to(self.device)
        y_tensor = torch.from_numpy(y_train).long().to(self.device)

        # Update the model with all samples at once
        self.model.batch_update(X_tensor, y_tensor)
        self.model.fit()

    def fit_target_dual_weighted(
        self,
        tgt_loader,
        model_src,
        clip_model,
        clip_processor,
        class_names,
        task_id,
        classes_per_task,
        use_freq_aug=False,
        wavelet='haar',
        aug_modes=('zeros','randn'),
        aug_prob=1, 
        conf_thr=0.0,           # keep sample if fused confidence ≥ conf_thr OR branches agree
        clip_text_template: str = "a photo of a {}"
    ):
        """
        REFEREE (corrected) dual-branch target adaptation:
          - Teacher = source KLDA + CLIP, run on CLEAN inputs (no augs).
          - Per-sample, label-free α/β from branch confidences (top1–top2 margins).
          - Probability-space fusion: p_hat = a * p_k + (1-a) * p_c.
          - Consensus gating: keep if fused_conf ≥ conf_thr OR (argmax_k == argmax_c).
          - Student updates on student features only (+ freq augs),
            reusing teacher pseudo-labels and inverse-entropy sample weights.

        Notes:
          * No ground-truth labels are used here (no leakage).
          * Ensures class masking to the current task's valid classes.
        """

        device = self.device
        clip_model.eval()
        self.backbone.eval()

        # Determine the current task's classes ----
        if isinstance(classes_per_task, list):
            start = sum(classes_per_task[:task_id])
            end   = sum(classes_per_task[:task_id + 1])
        else:
            start = task_id * classes_per_task
            end   = (task_id + 1) * classes_per_task

        valid_classes = model_src.model.models[0].classes_seen[start:end]
        if not valid_classes:
            print("[Dual Weighted] No valid classes for this task; check classes_per_task/task_id.")
            return

        # Global class list & indices for masking KLDA logits
        class_list   = sorted(model_src.model.models[0].classes_seen)
        class_to_idx = {c: i for i, c in enumerate(class_list)}
        idxs_task    = [class_to_idx[c] for c in valid_classes if c in class_to_idx]
        C_task       = len(idxs_task)

        # Build CLIP text features for this task's classes ----
        with torch.no_grad():
            text_inputs = clip_processor(
                # text=[f"a photo of a {class_names[c]}" for c in valid_classes],
                text=[clip_text_template.format(class_names[c]) for c in valid_classes],
                return_tensors="pt", padding=True
            ).to(device)
            text_feats = F.normalize(clip_model.get_text_features(**text_inputs), dim=1)  # [C_task, d_txt]

        # Precompute TEACHER ViT embeddings once (source backbone) ----
        # (We only use these to obtain KLDA teacher logits; student features are computed later.)
        X_tgt, _ = model_src.get_embeddings(tgt_loader, stat="adapt")
        X_tgt = torch.from_numpy(X_tgt).float().to(device)

        # Init target KLDA if needed (dimension from teacher features is fine; student is same ViT arch)
        d = X_tgt.shape[1]
        if self.model is None:
            self.model = KLDA_E(
                num_classes=len(class_names),
                d=d, D=self.D, sigma=self.sigma,
                num_ensembles=self.num_ensembles, seed=self.seed, device=device
            )

        # TEACHER pass: per-sample α/β from confidence margins; produce labels/weights ----
        fused_labels = []   # global label ids (in valid_classes' space, mapped back to global ids)
        fused_weights = []  # entropy weighting in [0,1]
        keep_mask = []      # bool per sample (gating)
        fused_confs = []    # fused max prob (for logging)
        global_idx = 0

        with torch.no_grad():
            for images, _ in tgt_loader:
                images = images.to(device)

                # CLIP image features on CLEAN images
                pil_batch = [to_pil_image(denorm_for_clip(img).cpu()) for img in images]
                img_inputs = clip_processor(images=pil_batch, return_tensors="pt", padding=True).to(device)
                img_feats = F.normalize(clip_model.get_image_features(**img_inputs), dim=1)  # [B, d_txt]

                for i in range(images.size(0)):
                    # --- KLDA teacher logits masked to task classes
                    x_teacher = X_tgt[global_idx]; global_idx += 1
                    logits_all = model_src.model.models[0].get_logits(x_teacher)        # [|seen|]
                    logits_k   = logits_all[idxs_task]                                   # [C_task]
                    p_k        = F.softmax(logits_k, dim=0)                              # [C_task]

                    # --- CLIP teacher probs on task classes
                    logits_c_all = img_feats[i] @ text_feats.T                           # [C_task]
                    p_c          = F.softmax(logits_c_all, dim=0)                        # [C_task]

                    # --- per-sample confidences via top1–top2 margins (more stable than raw max-prob)
                    top2_k, _ = torch.topk(p_k, 2)
                    top2_c, _ = torch.topk(p_c, 2)
                    mk = float((top2_k[0] - top2_k[1]).item())
                    mc = float((top2_c[0] - top2_c[1]).item())
                    denom = (mk + mc) if (mk + mc) > 1e-8 else 1e-8
                    a = mk / denom
                    # clamp α to avoid dominance by one branch due to calibration quirks
                    a = float(max(0.2, min(0.8, a)))
                    b = 1.0 - a

                    # --- probability-space fusion
                    p_hat = a * p_k + b * p_c                                            # [C_task]

                    # fused confidence + agreement gate
                    conf, idx = torch.max(p_hat, dim=0)
                    agree = (int(p_k.argmax().item()) == int(p_c.argmax().item()))
                    keep  = bool(conf.item() >= conf_thr or agree)

                    # entropy weight in [0,1]
                    H = -(p_hat * torch.log(p_hat.clamp_min(1e-12))).sum()
                    w = (1.0 - (H / math.log(C_task))).item()
                    w = max(0.2, min(1.0, w))  # floor=0.2 avoids ESS collapse

                    fused_labels.append(valid_classes[idx.item()])  # map task index -> global class id
                    fused_weights.append(w)
                    fused_confs.append(conf.item())
                    keep_mask.append(keep)

        kept = sum(1 for k in keep_mask if k)
        print(f"[Dual/Teacher] kept {kept}/{len(keep_mask)} samples "
              f"({100.0*kept/len(keep_mask):.1f}%), mean fused conf={np.mean(fused_confs):.3f}")

        # STUDENT updates: + freq-aug features ----
        X_list, y_list, w_list = [], [], []
        idx_ptr = 0

        with torch.no_grad():
            for images, _ in tgt_loader:
                images = images.to(device)
                bsz = images.size(0)

                # slice teacher outputs for this batch
                yb = fused_labels[idx_ptr:idx_ptr+bsz]
                wb = fused_weights[idx_ptr:idx_ptr+bsz]
                mb = keep_mask[idx_ptr:idx_ptr+bsz]   # booleans
                idx_ptr += bsz

                if any(mb):
                    # student features
                    f_clean = self.backbone(images)
                    f_clean = F.normalize(f_clean, dim=1)

                    # select kept samples
                    mb_t = torch.tensor(mb, device=device, dtype=torch.bool)
                    yb_t = torch.tensor([y for y,m in zip(yb, mb) if m], device=device, dtype=torch.long)
                    wb_t = torch.tensor([w for w,m in zip(wb, mb) if m], device=device, dtype=torch.float)

                    X_list.append(f_clean[mb_t])
                    y_list.append(yb_t)
                    w_list.append(wb_t)

                    # frequency-aware augmentation on STUDENT path only
                    if use_freq_aug and aug_prob > 0.0:
                        aug_batches = self._freq_augment_batch(images, wavelet=wavelet, modes=aug_modes, prob=aug_prob)
                        for aug in aug_batches:
                            f_aug = self.backbone(aug)
                            f_aug = F.normalize(f_aug, dim=1)
                            X_list.append(f_aug[mb_t])   # reuse same kept indices
                            y_list.append(yb_t)
                            w_list.append(wb_t)

        if not X_list:
            print("[Dual/Student] No confident samples passed gating; skip KLDA update.")
            return

        X_tensor = torch.cat(X_list, dim=0)
        y_tensor = torch.cat(y_list, dim=0)
        w_tensor = torch.cat(w_list, dim=0)

        print(f"[Dual/Student] updating on {len(y_tensor)} samples, "
              f"classes={sorted(set(y_tensor.tolist()))}")

        self.model.batch_update_weighted(X_tensor, y_tensor, sample_weights=w_tensor)
        self.model.fit()

        # Safety: ensure means exist for all task classes (copy from source if missing) ----
        tgt_m = self.model.models[0]
        src_m = model_src.model.models[0]
        missing = set(valid_classes) - set(tgt_m.classes_seen)
        if missing:
            print(f"[Inject Means] Copying class means for: {sorted(missing)}")
            for cls in sorted(missing):
                if cls not in src_m.classes_seen:
                    print(f"[Warn] Class {cls} not in source model; skip.")
                    continue
                tgt_m.class_means[cls]  = src_m.class_means[cls].clone()
                tgt_m.class_counts[cls] = float(src_m.class_counts.get(cls, 25.0))
                if cls not in tgt_m.classes_seen:
                    tgt_m.classes_seen.append(cls)
            tgt_m.class_mean_matrix = torch.stack(
                [tgt_m.class_means[c] for c in sorted(tgt_m.classes_seen)]
            ).to(device)
            print(f"[Inject Means] Updated class_mean_matrix.shape: {tgt_m.class_mean_matrix.shape}")


    def get_embeddings(self, dataloader, stat):
        """
        Extract feature embeddings from the backbone model for all samples in the dataloader.
        Returns:
            embeddings (np.ndarray): Feature matrix of shape (N, D).
            labels (np.ndarray): Corresponding labels of shape (N,).
        """
        self.backbone.eval()
        embeddings = []
        labels = []
        status = "Extracting embeddings " + stat

        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc=status):
                images = images.to(self.device)
                features = self.backbone(images)
                features = F.normalize(features, p=2, dim=1)
                embeddings.append(features.cpu().numpy())
                labels.append(targets.numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        return embeddings, labels

    def predict(self, input_embedding):
        """
        Predicts the label of a given sentence.
        Args:
            sentence (str): The input sentence to classify.
        Returns:
            str: The predicted label.
        """
        pred = self.model.predict(input_embedding)
        return pred