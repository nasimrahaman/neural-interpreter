import copy
from contextlib import contextmanager
from typing import List, Optional, Dict

import torch
from einops import rearrange, repeat
from torch import nn

from neural_interpreters.utils import ModelOutput, freeze_existing_module_parameters


class _BaseModel(nn.Module):
    def __init__(
        self,
        dim,
        num_classes_in_dataset,
        use_shared_cls_tokens=False,
        use_shared_prediction_head=False,
        patch_size=None,
        image_size=None,
        image_input_channels=3,
        num_entities=None,
        input_type: str = "patches",
        detach_patch_embeddings: bool = False,
    ):
        super().__init__()
        # Privates
        self._evaluate_datasets = None
        # Publics
        self.dim = dim
        self.num_classes_in_dataset = num_classes_in_dataset
        self.use_shared_cls_tokens = use_shared_cls_tokens
        self.use_shared_prediction_head = use_shared_prediction_head
        self.patch_size = patch_size
        self.image_size = image_size
        self.image_input_channels = image_input_channels
        self.num_entities = num_entities
        self.input_type = input_type
        self.detach_patch_embeddings = detach_patch_embeddings

    def _init_num_classes(self):
        # Prediction head business
        if isinstance(self.num_classes_in_dataset, int):
            self.num_classes_in_dataset = {"default": self.num_classes_in_dataset}
        else:
            assert isinstance(self.num_classes_in_dataset, dict)
        self.num_classes = self.num_classes_in_dataset
        if self.use_shared_prediction_head:
            self.num_classes = next(iter(self.num_classes_in_dataset.values()))
            assert all(
                [v == self.num_classes for v in self.num_classes_in_dataset.values()]
            ), "Number of classes should be the same for all datasets."

    def _init_cls_tokens(self):
        # One cls-token per head
        if self.use_shared_cls_tokens:
            self.shared_cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
            self.cls_token = {
                key: self.shared_cls_token for key in self.num_classes_in_dataset
            }
        else:
            self.shared_cls_token = None
            self.cls_token = nn.ParameterDict(
                {
                    key: nn.Parameter(torch.randn(1, 1, self.dim))
                    for key in self.num_classes_in_dataset
                }
            )

    def _init_prediction_heads(self):
        if self.use_shared_prediction_head:
            self.shared_prediction_head = nn.Sequential(
                nn.LayerNorm(self.dim), nn.Linear(self.dim, self.num_classes),
            )
            self.mlp_prediction_head = {
                key: self.shared_prediction_head
                for key, _num_classes in self.num_classes_in_dataset.items()
            }
        else:
            self.shared_prediction_head = None
            self.mlp_prediction_head = nn.ModuleDict(
                {
                    key: nn.Sequential(
                        nn.LayerNorm(self.dim), nn.Linear(self.dim, _num_classes)
                    )
                    for key, _num_classes in self.num_classes_in_dataset.items()
                }
            )

    def _init_patching(self):
        assert None not in [self.image_size, self.patch_size]
        assert self.image_size % self.patch_size == 0, (
            f"Can't {self.patch_size}x patch an image " f"of size {self.image_size}"
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.patch_dim = self.image_input_channels * self.patch_size ** 2
        self.patch_to_embedding = nn.Linear(self.patch_dim, self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        self.dropout = nn.Identity()

    def _init_entities(self):
        assert self.num_entities is not None
        self.entity_to_embedding = nn.Linear(self.image_input_channels, self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_entities, self.dim))
        self.dropout = nn.Identity()

    @contextmanager
    def evaluate_datasets(self, names: List[str]):
        old_evaluate_datasets = self._evaluate_datasets
        self._evaluate_datasets = names
        yield
        self._evaluate_datasets = old_evaluate_datasets

    def get_prediction_head(self, dataset_name: str) -> torch.nn.Module:
        if dataset_name in self.mlp_prediction_head:
            return self.mlp_prediction_head[dataset_name]
        else:
            # The model has not seen the dataset, so we need to make sure that
            # the prediction heads are meant to be shared.
            assert self.shared_prediction_head is not None
            return self.shared_prediction_head

    def get_heads_to_evaluate(self) -> List[str]:
        if self._evaluate_datasets is None:
            # Evaluate all the heads
            return list(self.mlp_prediction_head.keys())
        else:
            # Evaluate only the heads that are requested
            return list(self._evaluate_datasets)

    def get_cls_token(
        self, dataset_name: str, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        if dataset_name in self.cls_token:
            token = self.cls_token[dataset_name]
        else:
            assert self.shared_cls_token is not None
            token = self.shared_cls_token
        if batch_size is None:
            return token
        else:
            return repeat(token, "() n d -> b n d", b=batch_size)

    def get_required_cls_tokens(self) -> List[str]:
        if self._evaluate_datasets is None:
            return list(self.cls_token.keys())
        else:
            return list(self._evaluate_datasets)

    def get_dataset_name_to_token_idx(self) -> Dict[str, int]:
        dataset_name_to_token_idx = {
            key: idx for idx, key in enumerate(self.get_required_cls_tokens())
        }
        return dataset_name_to_token_idx

    def _extract_and_embed_patches(self, img) -> Dict[str, torch.Tensor]:
        p = self.patch_size
        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        assert x.shape[1] == self.num_patches, (
            f"Was expecting {self.num_patches} " f"tokens, got {x.shape[1]} instead."
        )
        x = self.patch_to_embedding(x)
        return {"embedded_patches": x, "embedded_shape": list(x.shape)}

    def _extract_and_embed_entities(self, entities) -> Dict[str, torch.Tensor]:
        # entities.shape = BNC, so no reshaping needed
        x = self.entity_to_embedding(entities)
        return {"embedded_entities": x, "embedded_shape": list(x.shape)}

    def _get_predictions(self, x):
        # TODO: Special code-path for when both prediction heads and cls tokens
        #  are shared.
        dataset_name_to_token_idx = self.get_dataset_name_to_token_idx()
        outputs = {}
        for dataset_name in self.get_heads_to_evaluate():
            pre_prediction = x[:, dataset_name_to_token_idx[dataset_name]]
            prediction_head = self.get_prediction_head(dataset_name)
            outputs[dataset_name] = prediction_head(pre_prediction)
        return outputs

    def _get_cls_tokens(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        required_tokens = self.get_required_cls_tokens()
        cls_tokens = [
            self.get_cls_token(name, batch_size=batch_size) for name in required_tokens
        ]
        return cls_tokens

    def forward(self, img, mask=None):
        if self.input_type == "patches":
            _patches = self._extract_and_embed_patches(img)
            x, (b, n, _) = _patches["embedded_patches"], _patches["embedded_shape"]
        elif self.input_type == "entities":
            _entities = self._extract_and_embed_entities(img)
            x, (b, n, _) = _entities["embedded_entities"], _entities["embedded_shape"]
        else:
            raise ValueError
        if self.detach_patch_embeddings:
            x = x.detach()
        input_patches = x
        # Token bookkeeping
        cls_tokens = self._get_cls_tokens(batch_size=b)
        x = torch.cat(cls_tokens + [x], dim=1)
        x[:, len(cls_tokens) :] += self.pos_embedding

        x = self.dropout(x)
        x = self.transformer(x, mask)

        outputs = self._get_predictions(x)

        return ModelOutput(
            predictions=outputs,
            output_patches=x[:, len(cls_tokens) :],
            input_patches=input_patches,
        )

    def finetune(
        self,
        dataset_name: str,
        num_classes: int,
        deepcopy: bool = False,
        freeze_existing_parameters: bool = False,
    ) -> "_BaseModel":
        """
        Sets the model in fine-tune mode.

        Parameters
        ----------
        dataset_name : str
            Name of the new dataset.
        num_classes : int
            Number of classes in the new dataset.
        deepcopy : bool
            Whether to return a deep copy of the model.
        freeze_existing_parameters : bool
            Whether to have the parameters that are thus far learned be untrainable.
            Note that this does not affect the new parameters that are instantiated
            in this function.
        Returns
        -------
        _BaseModel
        """
        if deepcopy:
            new_model = copy.deepcopy(self)
            return new_model.finetune(
                dataset_name=dataset_name,
                num_classes=num_classes,
                deepcopy=False,
                freeze_existing_parameters=freeze_existing_parameters,
            )
        self.train()
        # Kill gradients to the older parameters if need be
        if freeze_existing_parameters:
            freeze_existing_module_parameters(self)
        # Add new CLS token
        if self.shared_cls_token is not None:
            # In this case, we only have to register the new dataset name
            self.cls_token.update({dataset_name: self.shared_cls_token})
        else:
            # In this case, we'll have to define a new CLS token
            self.cls_token.update(
                {dataset_name: nn.Parameter(torch.randn(1, 1, self.dim))}
            )
        # Add new prediction head
        if self.shared_prediction_head is not None:
            # Again, we only need to register the new dataset name
            assert num_classes == self.num_classes, (
                "The number of classes must match " "when sharing prediction heads."
            )
            self.mlp_prediction_head.update({dataset_name: self.shared_prediction_head})
        else:
            # Make a new prediction head
            self.mlp_prediction_head.update(
                {
                    dataset_name: nn.Sequential(
                        nn.LayerNorm(self.dim), nn.Linear(self.dim, num_classes)
                    )
                }
            )
        return self
