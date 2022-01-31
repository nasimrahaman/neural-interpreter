import torch
from einops import rearrange
from torch import nn

from neural_interpreters.utils import ModelOutput
from neural_interpreters.modules import Transformer
from neural_interpreters.baselines.base_model import _BaseModel

MIN_NUM_PATCHES = 16


class MultiViT(_BaseModel):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes_in_dataset,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        num_entities=None,
        input_type: str = "patches",
        use_shared_cls_tokens=False,
        use_shared_prediction_head=False,
        detach_patch_embeddings=False,
    ):
        """
        Notes
        -----
        For example, if we're doing CIFAR-100 and SVHN, we would have:
          `num_classes_in_dataset = {"cifar-100": 100, "svhn": 10}`
        """
        assert input_type in ["patches", "entities"]
        super(MultiViT, self).__init__(
            dim=dim,
            num_classes_in_dataset=num_classes_in_dataset,
            use_shared_cls_tokens=use_shared_cls_tokens,
            use_shared_prediction_head=use_shared_prediction_head,
            num_entities=num_entities,
            input_type=input_type,
            image_input_channels=channels,
            detach_patch_embeddings=detach_patch_embeddings
        )
        # Validate parameters
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, (
            f"your number of patches ({num_patches}) is way too small for "
            f"attention to be effective (at least 16)."
            f" Try decreasing your patch size"
        )

        # Meta info
        self.image_size = image_size
        self.patch_size = patch_size

        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.pool = "cls"
        # Modules
        if input_type == "patches":
            self._init_patching()
        else:
            self._init_entities()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Prediction head business
        self._init_num_classes()
        self._init_cls_tokens()
        self._init_prediction_heads()


class HybridMultiViT(nn.Sequential):
    def __init__(self, backbone_kwargs: dict, multi_vit_kwargs: dict):
        # TODO
        pass
