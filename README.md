# APD-VAD

The code related to the paper below:

Xusheng Li, Haotong Du, Binghan Chen, Xianghua Li, Chao Gao, *Asymmetric Privileged Distillation for Weakly Supervised Video Anomaly Detection*, Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM26), 2026.

This paper introduces APD-VAD, an asymmetric privileged knowledge distillation framework for weakly supervised video anomaly detection.

## Abstract

Recent advances in Large Language Models (LLMs) and vision--language models have introduced richer semantic guidance into Weakly Supervised Video Anomaly Detection (WS-VAD). However, video-dependent text generation or encoding during inference introduces substantial computational overhead and complicates deployment in resource-constrained scenarios. We propose APD-VAD, an Asymmetric Privileged Distillation framework that uses LVLM-generated descriptions as privileged information during training and deploys a visual student without dynamic text processing. A Privileged Language Cascade (PLC) supplies video- and snippet-level descriptions to the teacher, and a Background-Debiased Graph (BDG) attenuates scene-dominated relations before temporal graph reasoning. Semantic-distribution, feature, and topology distillation transfer the resulting class, snippet, and relational knowledge to the student. At inference, a dual-branch visual graph and Visual Prototype Memory (VPM) provide category-aware visual context through cached category prototypes and class-routed visual retrieval. Experiments on UCF-Crime and XD-Violence demonstrate competitive anomaly detection and improved fine-grained temporal localization, while deployment profiling confirms efficient inference without recurring LVLM processing.

## Project Structure

The project comprises the following key components:

- `src/model_xd.py` / `src/model_ucf.py`: Model implementations for XD-Violence and UCF-Crime.
- `src/xd_train_teacher_student.py` / `src/ucf_train_teacher_student.py`: Teacher-student training and privileged knowledge distillation.
- `src/xd_test.py` / `src/ucf_test.py`: Evaluation scripts for anomaly detection and temporal localization.
- `src/xd_option.py` / `src/ucf_option.py`: Dataset paths and training configurations.
- `src/clip/`: CLIP ViT-B/16 backbone.
- `src/utils/`: Dataset loading, evaluation, and utility functions.
- `src/crop.py`: Visual preprocessing utilities.
- `scripts/`: Training, testing, and profiling scripts.
- `list/`: Dataset lists and ground-truth annotations.
- `tools/`: Offline text feature extraction utilities.
- `model/`: Optional pretrained weights and generated checkpoints.

## Installation

Clone the repository:

```bash
git clone https://github.com/xushengli501-design/APD-VAD.git
cd APD-VAD
```

Create a virtual environment:

```bash
python -m venv apdvad_env
source apdvad_env/bin/activate
```

Install the required dependencies:

```bash
pip install torch torchvision numpy pandas scipy scikit-learn opencv-python pillow tqdm ftfy regex
```

The CLIP implementation is included under `src/clip/`. The CLIP ViT-B/16 weights are downloaded automatically on first use through:

```python
clip.load("ViT-B/16")
```

## Data Preparation

The repository does not bundle the original videos or extracted visual/text features. Please download the datasets and prepare all required features before training or evaluation.

### Dataset Download

#### UCF-Crime

UCF-Crime can be downloaded from the official project page:

- Official project page:  
  https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/

- Direct dataset archive:  
  https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip

#### XD-Violence

XD-Violence can be downloaded from the official dataset page:

- Official project and download page:  
  https://roc-ng.github.io/XD-Violence/

The official page provides the training videos, test videos, and test annotations.

### Visual Feature Preparation

APD-VAD uses the frozen **CLIP ViT-B/16** visual encoder to construct the visual representations used by the model.

Prepare the CLIP visual features for UCF-Crime and XD-Violence and place the generated `.npy` files at the locations specified by the corresponding CSV files under `list/`.

The feature paths can also be changed through:

```text
src/ucf_option.py
src/xd_option.py
```

### Privileged Text Generation and Feature Extraction

The privileged language information used by APD-VAD is generated offline before training.

For each training video, frames are first sampled at **2 FPS**. A frozen CLIP ViT-B/16 visual encoder is used for semantic deduplication. If the cosine similarity between the current sampled frame and the previously retained frame is greater than `0.92`, the current frame is discarded.

The retained keyframes are then fed into **Qwen3-VL-32B** to generate two levels of textual descriptions:

- **Video-level description**: generated from the retained keyframes of the complete video.
- **Snippet-level description**: generated from the retained keyframes within each temporal snippet.

The prompt used for text generation is:

```text
You are a surveillance event analyst.
Describe the keyframe sequence in one sentence using the schema
[actors / actions / objects / scene].
Do not infer intent or outcome that is not visually grounded.
If a slot is not visible, omit it.
```

Qwen3-VL-32B is used only for generating natural-language descriptions. The generated descriptions are subsequently encoded using the frozen **CLIP text encoder** to obtain the privileged text features.

The complete preprocessing pipeline is:

```text
Video
  -> 2 FPS frame sampling
  -> CLIP semantic deduplication (threshold = 0.92)
  -> retained keyframes
  -> Qwen3-VL-32B
      -> video-level description
      -> snippet-level descriptions
  -> CLIP text encoder
  -> privileged text features
```

The provided text feature extraction utility can be used after the textual descriptions have been generated:

```bash
python tools/extract_text_features.py \
    --input <generated_text.jsonl> \
    --output <output_feature_directory> \
    --mode video
```

For snippet-level features:

```bash
python tools/extract_text_features.py \
    --input <generated_text.jsonl> \
    --output <output_feature_directory> \
    --mode snippet
```

All Qwen3-VL-32B processing and text feature extraction are performed offline before training. Qwen3-VL-32B is not used during student inference.

### Before Running

Before running the training or evaluation commands, make sure the following data have been prepared:

- UCF-Crime or XD-Violence dataset;
- CLIP visual features;
- video-level privileged text features;
- snippet-level privileged text features;
- dataset lists and ground-truth annotations.

The training scripts do **not** generate Qwen3-VL-32B descriptions or extract the required visual/text features online.

Once the required data have been prepared and the corresponding paths are correctly configured, the following commands can be used directly.

## Usage

### XD-Violence

Train APD-VAD:

```bash
bash scripts/train_xd.sh
```

Evaluate the trained model:

```bash
bash scripts/test_xd.sh
```

### UCF-Crime

Train APD-VAD:

```bash
bash scripts/train_ucf.sh
```

Evaluate the trained model:

```bash
bash scripts/test_ucf.sh
```

Training checkpoints are saved under `model/`, and terminal outputs are written to `logs/`.

If pretrained weights are available, they can be placed under `model/` or specified through the corresponding option files.

## Parameters

Training and evaluation parameters can be configured through:

```text
src/xd_option.py
src/ucf_option.py
```

or directly through the scripts under:

```text
scripts/
```

Important configurations include:

- dataset and feature paths;
- batch size;
- learning rate;
- teacher and student training schedules;
- semantic-distribution distillation weight;
- feature distillation weight;
- topology distillation weight;
- temporal regularization parameters.

## Datasets

### UCF-Crime

UCF-Crime is a large-scale real-world surveillance video anomaly detection dataset containing normal videos and 13 anomaly categories.

Download:

https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/

### XD-Violence

XD-Violence is a large-scale multi-scene video dataset containing normal videos and multiple violence-related anomaly categories.

Download:

https://roc-ng.github.io/XD-Violence/

## Evaluation

APD-VAD is evaluated using the standard weakly supervised video anomaly detection metrics:

- **UCF-Crime**: frame-level AUC and temporal detection mAP.
- **XD-Violence**: frame-level AP and temporal detection mAP.

Temporal localization performance is evaluated over multiple temporal IoU thresholds.

## Results

APD-VAD achieves competitive performance on both UCF-Crime and XD-Violence while avoiding recurring LVLM processing during student inference.

## License

This project is licensed under the MIT License.

## Citation

If you use APD-VAD in your research, please cite our paper:

```bibtex
@inproceedings{li2026apdvad,
  title     = {Asymmetric Privileged Distillation for Weakly Supervised Video Anomaly Detection},
  author    = {Li, Xusheng and Du, Haotong and Chen, Binghan and Li, Xianghua and Gao, Chao},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year      = {2026}
}
```
