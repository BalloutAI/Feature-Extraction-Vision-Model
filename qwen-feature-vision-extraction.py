#!/usr/bin/env python3
#SBATCH --job-name="v-7b-2.5-qwen-extraac"
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=qwens2.5-7b-vision-run1-not-normalized.txt

import os
import h5py
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import numpy as np
import torch


# Set random seeds for reproducibility
torch.manual_seed(123)  # 42,123, 2023 Seed for PyTorch
np.random.seed(123)     # Seed for NumPy


class FeatureExtractor:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def extract_features(self, root_dir, output_path):
        all_features = []
        all_labels = []
        
        # Process videos from both folders
        for class_id, class_name in enumerate(['plausible', 'implausible']):
            class_dir = os.path.join(root_dir, class_name)
            video_files = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
            
            for video_file in tqdm(video_files, desc=f"Processing {class_name} videos"):
                video_path = os.path.join(class_dir, video_file)
                
                # Prepare input like in the working example
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 420 * 360,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": "Describe this video"},
                    ],
                }]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                _, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                
                # Move to GPU
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                # Extract features
                with torch.no_grad():
                    print(f"Processing {video_file} - grid_thw shape:", inputs['video_grid_thw'].shape)
                    features = self.model.visual(
                        inputs['pixel_values_videos'],
                        grid_thw=inputs['video_grid_thw']
                    )
                    
                    # Average features across spatial-temporal patches and reshape to 2D
                    features = torch.mean(features, dim=0)  # Shape: [1536]
                    features = features.unsqueeze(0)  # Shape: [1, 1536]
                    
                    # Add L2 normalization
#                    features = torch.nn.functional.normalize(features, p=2, dim=1)
                    
                    print("\nFeature shape after averaging and normalization:", features.shape)
                    print("First feature row (normalized):")
                    print(features[0, :10])  # Print first 10 elements of first row
                    
                    all_features.append(features.cpu().numpy())
                    all_labels.extend([class_id])  # Note: removed * len(features)

        # Save to H5 file
        with h5py.File(output_path, 'w') as f:
            features_array = np.concatenate(all_features, axis=0)  # Will maintain [num_videos, 1536] shape
            labels_array = np.array(all_labels)
            print(f"Saving features shape: {features_array.shape}")  # Should be [num_videos, 1536]
            print(f"Saving labels shape: {labels_array.shape}")  # Should be [num_videos]
            f.create_dataset('features', data=features_array)
            f.create_dataset('labels', data=labels_array)
def main():
    extractor = FeatureExtractor()
    extractor.extract_features(
        root_dir=r"/home/staff/m/mballout/try/aa-video-llava-done/Video-LLaVA-inference/grasp-finetuning-dataset",  # Replace with your dataset path
        output_path="video_features-qwen2-5-7b-run1-not-normalized.h5"
    )

if __name__ == "__main__":
    main()


