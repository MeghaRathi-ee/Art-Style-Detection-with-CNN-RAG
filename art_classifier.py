"""
art_classifier.py
=================
EfficientNet-B0 fine-tuned for 15-class art style/movement classification.

Usage:
    from art_classifier import ArtStyleClassifier, classify_image, get_transforms
    from PIL import Image

    model = ArtStyleClassifier()
    model.load_state_dict(torch.load("art_classifier.pth", map_location="cpu"))
    model.eval()

    image = Image.open("painting.jpg").convert("RGB")
    style, confidence, top5 = classify_image(model, image)
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Tuple, List

# ── Art styles (subset of WikiArt categories) ────────────────────────────────
ART_STYLES = [
    "Abstract_Expressionism",
    "Action_painting",
    "Analytical_Cubism",
    "Art_Nouveau_Modern",
    "Baroque",
    "Color_Field_Painting",
    "Contemporary_Realism",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Impressionism",
    "Mannerism_Late_Renaissance",
    "Minimalism",
    "Naive_Art_Primitivism",
    "New_Realism",
    "Northern_Renaissance",
    "Pointillism",
    "Pop_Art",
    "Post_Impressionism",
    "Realism",
    "Rococo",
    "Romanticism",
    "Symbolism",
    "Synthetic_Cubism",
    "Ukiyo_e",
]

NUM_CLASSES = len(ART_STYLES)


# ── Image transforms ──────────────────────────────────────────────────────────
def get_transforms(train: bool = False) -> transforms.Compose:
    """
    Return torchvision transforms for training or inference.
    EfficientNet-B0 expects 224×224 images normalised with ImageNet stats.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ── Model ─────────────────────────────────────────────────────────────────────
class ArtStyleClassifier(nn.Module):
    """
    EfficientNet-B0 with a custom classification head.

    Architecture:
        EfficientNet-B0 backbone (pretrained on ImageNet)
            → global average pool  [1280-d features]
            → Dropout(0.4)
            → Linear(1280, 512)
            → GELU
            → Dropout(0.3)
            → Linear(512, NUM_CLASSES)

    The multi-layer head gives the model capacity to learn fine-grained
    style distinctions (e.g. Baroque vs Neoclassicism) that a single
    linear layer often misses.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = False):
        super().__init__()

        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features  # 1280

        # Replace the default classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            # Freeze all layers except the new classifier head
            for name, param in self.backbone.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_all(self) -> None:
        """Unfreeze the entire network (call after initial warm-up epochs)."""
        for param in self.parameters():
            param.requires_grad = True


# ── Inference helper ──────────────────────────────────────────────────────────
@torch.no_grad()
def classify_image(
    model: ArtStyleClassifier,
    image: Image.Image,
    top_k: int = 5,
    device: str = "cpu",
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Classify a PIL Image and return the top prediction + top-k list.

    Args:
        model:   Trained ArtStyleClassifier (in eval mode).
        image:   PIL Image in RGB format.
        top_k:   Number of top predictions to return.
        device:  "cpu" or "cuda".

    Returns:
        (best_style, confidence, top_k_list)
        - best_style:  e.g. "Impressionism"
        - confidence:  float in [0, 1]
        - top_k_list:  [(style_name, probability), ...]
    """
    model.eval()
    transform = get_transforms(train=False)

    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)  # [1, NUM_CLASSES]
    probs = torch.softmax(logits, dim=-1)[0]  # [NUM_CLASSES]

    # Top-k predictions
    top_probs, top_indices = torch.topk(probs, k=min(top_k, NUM_CLASSES))
    top_predictions = [
        (ART_STYLES[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

    best_style, best_confidence = top_predictions[0]
    return best_style, best_confidence, top_predictions