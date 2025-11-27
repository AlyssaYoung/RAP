import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE
import copy
import traceback
import json
import os
import pdb


class LLaVA_OneVision(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = '<image>'
    IMAGE_TOKEN_INDEX = -200

    def __init__(
            self,
            model_path='lmms-lab/llava-onevision-qwen2-0.5b-ov',
            debug=False,
            is_process_image=False,
            processed_image_path=None,
            max_step=200,
            bias_value=0.2,
            rag_model_path='openbmb/VisRAG-Ret',
            vision_tower_path=None,
            **kwargs
        ):
        assert model_path is not None
        try:
            from llava.model.multimodal_encoder import builder
            original_build_vision_tower = builder.build_vision_tower

            def patched_build_vision_tower(vision_tower_cfg, **kw):
                # Extract vision tower path robustly
                vt_path = vision_tower_path

                if vt_path is None:
                    vt_path = (
                        getattr(vision_tower_cfg, "mm_vision_tower", None)
                        or getattr(vision_tower_cfg, "vision_tower", None)
                        or getattr(
                            getattr(vision_tower_cfg, "vision_config", {}),
                            "model_name_or_path",
                            None,
                        )
                    )

                # SigLIP keyword detection
                if vt_path and "siglip" in vt_path.lower():
                    try:
                        # Try main siglip encoder
                        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
                    except ImportError:
                        # Fallback to lite encoder
                        from llava.model.multimodal_encoder.siglip_encoder_lite import SigLipVisionTower

                    # Pass args correctly
                    return SigLipVisionTower(vt_path, vision_tower_cfg=vision_tower_cfg, **kw)

                # Otherwise fallback to original tower
                return original_build_vision_tower(vision_tower_cfg, **kw)

            # Apply monkey patch
            builder.build_vision_tower = patched_build_vision_tower

            # Also patch in llava_arch if already imported
            try:
                import llava.model.llava_arch as llava_arch_module
                if hasattr(llava_arch_module, 'build_vision_tower'):
                    llava_arch_module.build_vision_tower = patched_build_vision_tower
            except:
                pass

            print("[INFO] Patched build_vision_tower (SigLIP prioritized).")

        except Exception as e:
            warnings.warn(f"Failed to patch build_vision_tower: {e}")
            traceback.print_exc()

        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates
            from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        except ImportError:
            traceback.print_exc()
            warnings.warn('Please install LLaVA-NeXT.')
            return

        super().__init__(
            debug=debug,
            is_process_image=is_process_image,
            processed_image_path=processed_image_path,
            max_step=max_step,
            bias_value=bias_value,
            rag_model_path=rag_model_path
        )

        self.model_path = model_path
        model_name = get_model_name_from_path(model_path)

        # Only override memory config (do NOT write to disk)
        self.override_vision_tower_path = vision_tower_path

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map=None,
            torch_dtype="float16",
            attn_implementation='flash_attention_2',
        )

        model.cuda().eval()
        model.to(torch.float16)
        vt = model.get_vision_tower()
        if vt is not None:
            vt.to(torch.float16)
        model.tie_weights()

        if 'llava' in model_path.lower():
            conv_mode = 'qwen_1_5'
        self.conv_template = conv_mode
        self.conv_templates = conv_templates

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = process_images
        self.device = self.model.device

        self.init_index_yes_no()
        
    def init_index_yes_no(self):
        if len(self.tokenizer("Yes").input_ids) == 1 and len(self.tokenizer("No").input_ids) == 1: 
            self.index_yes = self.tokenizer("Yes").input_ids[0]
            self.index_no = self.tokenizer("No").input_ids[0]
        else:
            assert len(self.tokenizer("Yes").input_ids) == 2 and len(self.tokenizer("No").input_ids) == 2
            self.index_yes = self.tokenizer("Yes").input_ids[1]
            self.index_no = self.tokenizer("No").input_ids[1]
        
    @torch.no_grad()
    def get_confidence_value(self, content, image_list: Image.Image):
        if not isinstance(image_list, list):
            image_list = [image_list]
        image_sizes = [img.size for img in image_list]
        image_tensor = self.process_images(image_list, self.image_processor, self.model.config).to(dtype=torch.float16)
        image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]
        content = self.DEFAULT_IMAGE_TOKEN + '\n' + content
        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda(self.device)
        outputs = self.model(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            modalities=["image"] * len(input_ids),
            return_dict=True
        )
        return self._cal_confidence(outputs)
    
    @torch.no_grad()
    def _cal_confidence(self, outputs):
        logits_yesno = outputs.logits[0, -1, [self.index_yes, self.index_no]]
        confidence = torch.softmax(logits_yesno, dim=-1)[0] 
        confidence = 2 * (confidence.item() - 0.5) # [-1, 1]
        return confidence

    def generate_inner(self, message, dataset=None):
        content, images = '', []
        image_sizes = []  # Store image sizes

        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                if isinstance(msg['value'],str):
                    img = Image.open(msg['value']).convert('RGB')
                else:
                    img = msg['value']
                images.append(img)
                image_sizes.append(img.size)  # Store the size of each image
                content = self.DEFAULT_IMAGE_TOKEN + '\n' + content

        # Process images using the class attribute self.process_images
        image_tensor = self.process_images(images, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(prompt_question,
                                               self.tokenizer,
                                               self.IMAGE_TOKEN_INDEX,
                                               return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda(self.device)

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs
