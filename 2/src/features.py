import numpy as np
from PIL import Image


def preprocess_image(image: np.ndarray, image_size: int = 224):
    """Convert a grayscale image to the three-channel tensor expected by ResNet."""
    from torchvision import transforms

    if image.ndim != 2:
        raise ValueError("image must be grayscale")
    pil_image = Image.fromarray(image).convert("RGB").resize((image_size, image_size))
    return transforms.ToTensor()(pil_image)


class ResNet18Backend:
    """Lazy torchvision wrapper; no model weights are touched until init_backend()."""

    def __init__(self, *, device: str = "cpu"):
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        self.device = device
        self._model = None
        self._transform = None

    @property
    def is_initialized(self) -> bool:
        return self._model is not None

    def init_backend(self):
        if self._model is not None:
            return self
        try:
            import torch
            from torchvision import models, transforms
        except ImportError as exc:
            raise RuntimeError("ResNet18 backend requires PyTorch and torchvision") from exc
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self._model = models.resnet18(weights=weights)
        self._model.eval()
        self._model.to(self.device)
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return self

    def extract_patches(self, image, *, patch_size: int, stride: int) -> np.ndarray:
        return self.extract_feature_grid(image, patch_size=patch_size, stride=stride)[0]

    def extract_feature_grid(self, image: np.ndarray, *, patch_size: int = 224, stride: int = 16):
        """Return a 14x14 layer3 grid and raw grayscale for grid-aligned context maps."""
        self.init_backend()
        import torch

        tensor = preprocess_image(image, patch_size)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        grid = {}
        handle = self._model.layer3.register_forward_hook(
            lambda module, inputs, output: grid.setdefault("features", output.detach())
        )
        try:
            with torch.inference_mode():
                self._model(tensor)
        finally:
            handle.remove()
        features = grid["features"].squeeze(0).permute(1, 2, 0).reshape(-1, 256).cpu().numpy()
        return features, (14, 14)


def get_feature_backend(name: str):
    if name == "resnet18":
        return ResNet18Backend()
    raise ValueError(f"unsupported feature backend: {name}")
