import warnings

import cv2
import numpy as np
import torch
import torchvision


class BiRefNetMattingEngine(torch.nn.Module):
    """BiRefNet-based matting engine matching the StyleMatteEngine interface.

    Uses HuggingFace AutoModelForImageSegmentation to load BiRefNet for
    background removal. Drop-in replacement for StyleMatteEngine.
    """

    def __init__(self, device='cpu', model_id='ZhengPeng7/BiRefNet', use_fp16=True):
        super().__init__()
        self._device = device
        self._model_id = model_id
        self._use_fp16 = use_fp16 and ('cuda' in str(device))
        self._process_resolution = 1024
        self._init_model()

    def _init_model(self):
        from transformers import AutoModelForImageSegmentation

        self.model = AutoModelForImageSegmentation.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )
        self.model.to(self._device)
        self.model.eval()

        if self._use_fp16:
            self.model.half()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                (self._process_resolution, self._process_resolution)
            ),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def forward(self, input_image, return_type='matting', background_rgb=1.0):
        """Run BiRefNet matting on a [C,H,W] tensor in [0,1] range.

        Args:
            input_image: [C,H,W] torch tensor, values in [0,1]
            return_type: 'alpha', 'matting', or 'all'
            background_rgb: background color scalar for compositing

        Returns:
            Depends on return_type:
            - 'alpha': [H,W] alpha mask tensor
            - 'matting': ([C,H,W] composited tensor, [H,W] alpha mask)
            - 'all': ([C,H,W] foreground composite, [C,H,W] background composite)
        """
        if input_image.max() > 2.0:
            warnings.warn('Image should be normalized to [0, 1].')

        _, ori_h, ori_w = input_image.shape
        input_image = input_image.to(self._device).float()

        # Prepare input: resize to process_resolution and normalize
        processed = input_image.clone().unsqueeze(0)  # [1, C, H, W]
        processed = torchvision.transforms.functional.resize(
            processed, (self._process_resolution, self._process_resolution),
            antialias=True,
        )
        processed = torchvision.transforms.functional.normalize(
            processed, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
        )

        if self._use_fp16:
            processed = processed.half()

        # BiRefNet inference - take last output and sigmoid
        preds = self.model(processed)[-1].sigmoid()

        # Resize prediction back to original size
        predict = torchvision.transforms.functional.resize(
            preds, (ori_h, ori_w), antialias=True,
        )  # [1, 1, H, W]
        predict = predict[0]  # [1, H, W]

        if return_type == 'alpha':
            return predict[0]  # [H, W]
        elif return_type == 'matting':
            predict = predict.float().expand(3, -1, -1)  # [3, H, W]
            background = input_image.new_ones(input_image.shape) * background_rgb
            matting_image = input_image * predict + (1 - predict) * background
            return matting_image, predict[0]  # ([C,H,W], [H,W])
        elif return_type == 'all':
            predict = predict.float().expand(3, -1, -1)
            background = input_image.new_ones(input_image.shape) * background_rgb
            foreground_image = input_image * predict + (1 - predict) * background
            background_image = input_image * (1 - predict) + predict * background
            return foreground_image, background_image
        else:
            raise NotImplementedError(f"Unknown return_type: {return_type}")
