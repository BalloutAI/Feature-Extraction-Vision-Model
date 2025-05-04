# Video Feature Extraction with Qwen Vision-Language Models

## Overview

This repository contains two Python scripts to extract video-level features from MP4 files using Qwen multimodal (vision-language) models:

1. **qwen-feature-vision-extraction.py**  
   - Uses `Qwen2.5-VL-7B-Instruct`  
2. **extract-feature-with-text-alignment.py**  
   - Uses `Qwen2-VL-2B-Instruct`  


Both scripts process videos in two classes—`plausible` and `implausible`—and produce an HDF5 file containing feature vectors and class labels.  

---

## Requirements

- Python 3.8 or higher  
- CUDA-enabled GPU (recommended)  
- PyTorch  
- [transformers](https://github.com/huggingface/transformers)  
- [bitsandbytes](https://github.com/facebookresearch/bitsandbytes)  
- [tqdm](https://github.com/tqdm/tqdm)  
- [h5py](https://github.com/h5py/h5py)  
- numpy  

You also need the helper module `qwen_vl_utils` in your `PYTHONPATH`.  

---

## Installation

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/qwen-video-feature-extraction.git
   cd qwen-video-feature-extraction
   ```

2. Create a virtual environment and install dependencies:  
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
   pip install transformers bitsandbytes tqdm h5py numpy
   ```

3. Ensure `qwen_vl_utils.py` is accessible (either in the project root or installed as a package).
