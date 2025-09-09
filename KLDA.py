import torch
from collections import defaultdict
import torch.nn.functional as F
import math

import torch
from collections import defaultdict
import torch.nn.functional as F
import math

class KLDA:
    def __init__(self, num_classes, d, D, sigma, seed, device):
        """
        Args:
            num_classes (int): Total number of classes (e.g., 10 for MNIST).
            d (int): Input feature dimension.
            D (int): RFF dimension.
            sigma (float): Bandwidth parameter for RBF kernel.
            seed (int): Seed for RFF initialization.
            device (torch.device): Device to run the models on.
        """
        self.dtype = torch.float64
        self.num_classes = num_classes
        self.d = d
        self.D = D
        self.sigma = sigma
        self.seed = seed
        self.device = device

        torch.manual_seed(self.seed)
        self.omega = torch.normal(
            0, 
            torch.sqrt(torch.tensor(2 * self.sigma, dtype=self.dtype)),
            (self.d, self.D), dtype=self.dtype, device=self.device
        )
        self.b = (torch.rand(self.D, dtype=self.dtype, device=self.device) * 2 * math.pi)

        # Means and counts for each class
        self.class_means = defaultdict(lambda: torch.zeros(self.D, dtype=self.dtype, device=self.device))
        self.class_counts = defaultdict(float)

        # Covariance matrix
        self.sigma_matrix = torch.zeros((self.D, self.D), dtype=self.dtype, device=self.device)
        self.sigma_inv = None
        self.class_mean_matrix = None

        # Keep track of which classes we've seen
        self.classes_seen = []

    def _compute_rff(self, X):
        """
        Computes the Random Fourier Features for input data X.
        Args:
            X (torch.Tensor): Shape (n_samples, d).
        Returns:
            torch.Tensor: Shape (n_samples, D).
        """
        X = X.to(self.device, dtype=self.dtype)
        scaling_factor = torch.sqrt(torch.tensor(2.0 / self.D, dtype=self.dtype, device=self.device))
        return scaling_factor * torch.cos(X @ self.omega + self.b)

    def batch_update(self, X, y):
        """
        Updates the model with a batch of data.
        Args:
            X (torch.Tensor): Feature tensor (n_samples, d).
            y (int or torch.Tensor): Class label(s).
        """
        X = X.to(self.device)
        # Ensure y is a tensor
        if isinstance(y, int):
            y = torch.tensor([y], device=self.device)
        else:
            y = y.to(self.device)
            if y.dim() == 0:  # Single value
                y = y.unsqueeze(0)

        phi_X = self._compute_rff(X)  # Shape: (n, D)

        for cls in torch.unique(y):
            cls = cls.item()
            mask = (y == cls)
            n_cls = mask.sum().item()
            if n_cls == 0:
                continue

            phi_cls = phi_X[mask]
            phi_cls_mean = torch.mean(phi_cls, dim=0)

            previous_count = self.class_counts[cls]

            print(f'[Source] Updating class {cls} mean. Current count: {previous_count}, New samples: {n_cls}')

            self.class_counts[cls] += n_cls
            self.class_means[cls] = (
                self.class_means[cls] * previous_count + phi_cls_mean * n_cls
            ) / self.class_counts[cls]

            # Update covariance
            centered_phi_cls = phi_cls - self.class_means[cls]
            self.sigma_matrix += centered_phi_cls.t() @ centered_phi_cls

            # Add to classes_seen if new
            if cls not in self.classes_seen:
                self.classes_seen.append(cls)

    def batch_update_weighted(self, X, y, sample_weights=None):
        """
        Updates the model with a batch of data.
        Args:
            X (torch.Tensor): Feature tensor (n_samples, d).
            y (int or torch.Tensor): Class label(s).
        """
        X = X.to(self.device, dtype=self.dtype)

        if isinstance(y, int):
            y = torch.tensor([y], device=self.device)
        else:
            y = y.to(self.device)
            if y.dim() == 0:
                y = y.unsqueeze(0)

        phi_X = self._compute_rff(X)  # (n, D)

        # Per-sample weights
        if sample_weights is None:
            weights = torch.ones(len(y), dtype=self.dtype, device=self.device)
        else:
            weights = sample_weights.to(self.device).to(self.dtype).flatten()
        # clamp weights to avoid zeros/negatives/NaNs
        weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
        weights = torch.clamp(weights, min=1e-8)

        for cls in torch.unique(y):
            cls_i = cls.item()
            mask = (y == cls)
            n_cls = mask.sum().item()
            if n_cls == 0:
                continue

            w = weights[mask]                     # (n_cls,)
            phi_cls = phi_X[mask]                 # (n_cls, D)
            w_sum = torch.clamp(w.sum(), min=torch.tensor(1e-8, dtype=self.dtype, device=self.device))

            # Weighted mean for the incoming batch of this class
            phi_wmean = (w.unsqueeze(1) * phi_cls).sum(dim=0) / w_sum

            previous_count = float(self.class_counts[cls_i])
            new_count = previous_count + float(w_sum.item())

            print(f'[Target] Updating class {cls_i} mean. Current count: {previous_count}, New samples: {n_cls}, New weighted samples: {float(w_sum.item()):.1f}')

            # Incremental weighted mean update
            self.class_means[cls_i] = (
                self.class_means[cls_i] * previous_count + phi_wmean * w_sum
            ) / new_count

            self.class_counts[cls_i] = new_count

            # Weighted covariance update: sum_i w_i * (x_i - mu)(x_i - mu)^T
            centered = phi_cls - self.class_means[cls_i]            # (n_cls, D)
            cw = torch.sqrt(w).unsqueeze(1) * centered              # (n_cls, D)
            self.sigma_matrix += cw.t() @ cw

            if cls_i not in self.classes_seen:
                self.classes_seen.append(cls_i)


    def fit(self):
        """
        Finalizes the model by computing the inverse of covariance matrix
        and stacking all class means for seen classes.
        """
        self.sigma_inv = torch.pinverse(self.sigma_matrix + 1e-6 * torch.eye(self.D, device=self.device))
        self.class_mean_matrix = torch.stack(
            [self.class_means[c] for c in sorted(self.classes_seen)]
        ).to(self.device)

        print(f"Updated class_mean_matrix.shape: {self.class_mean_matrix.shape}, classes_seen: {self.classes_seen}")

    def get_logits(self, x):
        """
        Computes logits for all seen classes.
        Args:
            x (torch.Tensor): Shape (d,).
        Returns:
            torch.Tensor: Logits for each seen class (len(classes_seen),).
        """
        x = x.to(self.device)
        phi_x = self._compute_rff(x.unsqueeze(0))  # Shape: (1, D)
        diff = self.class_mean_matrix - phi_x      # Shape: (n_seen_classes, D)
        logits = -torch.sum((diff @ self.sigma_inv) * diff, dim=1)  # Mahalanobis
        return logits

    def initialize_class_means_from_source(self, source_model):
        """
        Initialize this KLDA model's class_means and class_counts
        using the values from a source KLDA model.
        """
        for cls in source_model.classes_seen:
            if cls in source_model.class_means:
                self.class_means[cls] = source_model.class_means[cls].clone()
                self.class_counts[cls] = source_model.class_counts[cls]
                if cls not in self.classes_seen:
                    self.classes_seen.append(cls)

        # Recompute the class_mean_matrix
        self.class_mean_matrix = torch.stack(
            [self.class_means[c] for c in sorted(self.classes_seen)]
        ).to(self.device)

        print(f"[Init from Source] Copied {len(self.classes_seen)} class means from source.")



class KLDA_E:
    """Ensemble version of KLDA."""
    def __init__(self, num_classes, d, D, sigma, num_ensembles, seed, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_ensembles = num_ensembles
        self.device = device
        self.models = [
            KLDA(num_classes, d, D, sigma, seed=seed+i, device=self.device) for i in range(self.num_ensembles)
        ]

    def batch_update(self, X, y):
        for model in self.models:
            model.batch_update(X, y)

    def batch_update_weighted(self, X, y, sample_weights=None):
        for model in self.models:
            model.batch_update_weighted(X, y, sample_weights=sample_weights)

    def fit(self):
        for model in self.models:
            model.fit()

    def predict(self, x):
        total_probabilities = torch.zeros(len(self.models[0].classes_seen), device=self.device)

        for model in self.models:
            logits = model.get_logits(x)
            probs = torch.softmax(logits, dim=0)
            total_probabilities += probs

        predicted_class_idx = torch.argmax(total_probabilities).item()
        return model.classes_seen[predicted_class_idx]  # Map back to global class

    def predict_target(self, x, valid_classes):
        """
        Ensemble prediction restricted to a subset of valid classes.
        """
        class_list = sorted(self.models[0].classes_seen)
        class_to_idx = {c: i for i, c in enumerate(class_list)}
        logits_sum = torch.zeros(len(class_list), device=self.device)

        for model in self.models:
            logits = model.get_logits(x)
            probs = torch.softmax(logits, dim=0)
            logits_sum += probs

        # Filter to valid class subset
        idxs = [class_to_idx[c] for c in valid_classes if c in class_to_idx]
        if not idxs:
            raise ValueError("None of the valid_classes were seen by the model.")

        idxs_tensor = torch.tensor(idxs, device=self.device)
        logits_valid = logits_sum[idxs_tensor]
        best_idx = torch.argmax(logits_valid).item()

        return valid_classes[best_idx]


