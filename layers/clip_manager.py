import sys
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPImageProcessor
from transformers import Blip2Processor, Blip2Model

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from layers.models_mae import *


class VLMManager:
    """
    Manager class to handle different VLM types (CLIP, BLIP2, ViLT).
    """

    def __init__(self, config):
        self.config = config
        # self.vlm_type = config.vlm_type.lower()
        self.device = self._acquire_device()
        self._init_vlm()

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _init_vlm(self):

        self._init_clip()

        self.model.to(self.device)
        learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _init_clip(self):
        CLIP_ARCH = 'openai/clip-vit-base-patch32'
        try:
            print("Trying to load from local cache...")
            self.processor = CLIPImageProcessor.from_pretrained(CLIP_ARCH, local_files_only=True)
            self.model = CLIPVisionModel.from_pretrained(CLIP_ARCH, output_hidden_states=True, local_files_only=True)
            print("Successfully loaded from local cache!")
        except Exception as e:
            print(f"Local cache not found: {e}")
            print("Loading from remote...")
            self.processor = CLIPImageProcessor.from_pretrained(CLIP_ARCH)
            self.model = CLIPVisionModel.from_pretrained(CLIP_ARCH, output_hidden_states=True)
            print("Successfully loaded from remote!")

        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 512
        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 77
        self.fused_feature_len = 9
        self.multimodal_fusion_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        ).to(self.device)

    def _set_requires_grad(self, model, value):
        for param in model.parameters():
            param.requires_grad = value
        for child in model.children():
            self._set_requires_grad(child, value)

    def process_inputs(self, B, images):

        return self._process_clip_inputs(B, images)

    def _process_clip_inputs(self, B, images):
        encoding = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True)
        image_features = outputs.pooler_output  # Shape: [B, hidden_size]
        return image_features

