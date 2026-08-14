# APD-VAD

The code related to the paper below:

Xusheng Li, Haotong Du, Binghan Chen, Xianghua Li, Chao Gao, *Asymmetric Privileged Distillation for Weakly Supervised Video Anomaly Detection*, Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM26), 2026.

This paper introduces APD-VAD, a method that proposes a privileged knowledge distillation framework for weakly-supervised video anomaly detection.

## Abstract

Recent advances in Large Language Models (LLMs) and vision--language models have introduced richer semantic guidance into Weakly Supervised Video Anomaly Detection (WS-VAD). However, video-dependent text generation or encoding during inference introduces substantial computational overhead and complicates deployment in resource-constrained scenarios. We propose APD-VAD, an Asymmetric Privileged Distillation framework that uses LVLM-generated descriptions as privileged information during training and deploys a visual student without dynamic text processing. A Privileged Language Cascade (PLC) supplies video- and snippet-level descriptions to the teacher, and a Background-Debiased Graph (BDG) attenuates scene-dominated relations before temporal graph reasoning. Semantic-distribution, feature, and topology distillation transfer the resulting class, snippet, and relational knowledge to the student. At inference, a dual-branch visual graph and Visual Prototype Memory (VPM) provide category-aware visual context through cached category prototypes and class-routed visual retrieval. Experiments on UCF-Crime and XD-Violence demonstrate competitive anomaly detection and improved fine-grained temporal localization, while deployment profiling confirms efficient inference without recurring LVLM processing.

## Project Structure

The project comprises the following key components:

- `model_xd.py` / `model_ucf.py`: Implementation of the XD-Violence and UCF-Crime models (CLIPVAD), including the CLIP backbone, temporal transformer, debiased causal graph, and the DCSA / DNP modules.
- `xd_train_teacher_student.py` / `ucf_train_teacher_student.py`: Teacher-student training scripts, including teacher training, privileged-knowledge distillation losses, and optimization steps.
- `xd_test.py` / `ucf_test.py`: Evaluation scripts for AUC, AP, and detection mAP.
- `xd_option.py` / `ucf_option.py`: Argument parsing and resolution of feature paths.
- `clip/`: CLIP backbone (ViT-B/16), bundled locally, no external installation required.
- `utils/`: Utilities for dataset loading, detection-MAP computation, and other helpers.
- `crop.py`: Feature cropping with 10 spatial-crop variants.
- `scripts/`: One-click training and testing scripts plus a GPU profiler.
- `list/`: Data lists (CSV) and ground-truth annotations (npy), along with the scripts that generate them.
- `tools/`: Text feature extraction script (CLIP text encoding of the generated descriptions).

## Installation

To set up the environment:

```bash
git clone <this-repository>
cd APD-VAD-repro
```

Create a virtual environment (recommended):

```bash
python -m venv apdvad_env
source apdvad_env/bin/activate
```

Install dependencies:

```bash
pip install torch numpy pandas scipy scikit-learn
```

The CLIP backbone (`src/clip/`) is bundled in this repository, so no separate CLIP package is needed. The CLIP ViT-B/16 weights are downloaded automatically on first use via `clip.load("ViT-B/16")`.

## Data Preparation

The repository does **not** bundle any extracted features. The **privileged text features** (unique to this method) are generated offline into `data/` before training:

| Directory | Content |
|-----------|---------|
| `data/xd_text` | XD-Violence video-level text descriptions |
| `data/ucf_text` | UCF-Crime video-level text descriptions |
| `data/ucf_text_llm_snippets` | UCF-Crime snippet-level LLM text |
| `data/xd_text_v2_snippets` | XD-Violence snippet-level LLM text (v2) |

### Text Generation and Feature Extraction

The privileged text used by APD-VAD is generated offline before training.

For each training video, frames are first sampled at **2 FPS**. A frozen CLIP ViT-B/16 visual encoder is then used for semantic deduplication. If the cosine similarity between the current sampled frame and the previously retained frame is greater than `0.92`, the current frame is discarded.

The retained keyframes are fed into **Qwen3-VL-32B** to generate two levels of textual descriptions:

- **Video-level description**: generated from the retained keyframes of the entire video.
- **Snippet-level description**: generated from the retained keyframes within each temporal snippet.

The prompt used for text generation is:

```text
You are a surveillance event analyst.
Describe the keyframe sequence in one sentence using the schema
[actors / actions / objects / scene].
Do not infer intent or outcome that is not visually grounded.
If a slot is not visible, omit it.
```

Qwen3-VL-32B is used only to generate natural-language descriptions. The generated descriptions are subsequently encoded using the frozen **CLIP text encoder** to obtain the privileged text features used during teacher training.

The complete text preprocessing pipeline is:

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

All text generation and text feature extraction are performed offline before training. Qwen3-VL-32B is not required during student inference.

The **visual and audio features** are not bundled (large size) and must be extracted with common models and placed under `data/`:

| Feature | Extraction model | Output dim | Target path |
|---------|------------------|------------|-------------|
| Visual | CLIP **ViT-B/16** | 512 | `data/UCFClipFeatures` (UCF), `data/XDTrainClipFeatures` and `data/XDTestClipFeatures` (XD) |
| Audio | **VGGish** | 128 | `data/xd_audio_vggish` (XD) |

- **Visual features**: sample one frame per 16-frame snippet, apply the crop in `src/crop.py` (crop type 5: resize 340×256 → center 224×224 → horizontal flip), encode with the CLIP ViT-B/16 visual encoder to 512-d, and save one `.npy` per video with shape `(T, 512)`.
- **Audio features**: extract with VGGish to 128-d and save one `.npy` per video.

The `path` column of `list/*.csv` and the option defaults already point to these locations.

### Before Running

Before running the training or evaluation commands below, please make sure that all required data have been prepared and placed in the corresponding directories described above, including:

- visual features;
- audio features for XD-Violence;
- video-level privileged text features;
- snippet-level privileged text features;
- dataset lists and ground-truth annotations.

The training scripts do **not** run Qwen3-VL-32B or generate privileged text features online. Text generation and CLIP text feature extraction must therefore be completed in advance.

Once all required data are prepared, the commands below can be used directly.

## Usage

### XD-Violence

Train the teacher for 3 epochs, then joint distillation for 12 epochs with DCSA / DNP / audio auxiliary / debiased causal graph.

```bash
bash scripts/train_xd.sh
bash scripts/test_xd.sh
```

### UCF-Crime

Train the teacher for 1 epoch, then the student for 15 epochs with debiased causal graph and Gaussian smoothing (σ=2).

```bash
bash scripts/train_ucf.sh
bash scripts/test_ucf.sh
```

Pretrained weights `model/model_xd.pth` (XD) and `model/model_ucf.pth` (UCF) are optional: if placed in `model/`, they are used to initialize the teacher and reproduce the best results; otherwise training starts from random initialization and still runs, though with degraded performance. Training automatically saves the best checkpoint to `model/`, and all terminal output is written to `logs/`.

## Parameters

Customize the training process by modifying arguments in `xd_option.py` / `ucf_option.py` or in `scripts/train_*.sh`. Key parameters include:

- Dataset: XD-Violence or UCF-Crime.
- Teacher epochs, student epochs, learning rate, and batch size.
- Distillation weights: `--distill-kd-bin-weight`, `--distill-kd-multi-weight`, `--distill-kd-feat-weight`, `--kd-temp`.
- Loss weights: `--dcsa-loss-weight`, `--dnp-loss-weight`, `--loss1-weight`, `--loss2-weight`, `--temporal-consistency-weight`.
- UCF post-processing: `--gaussian-sigma 2`.


## Datasets

- **XD-Violence**: 7 classes, audio-available test subset.
- **UCF-Crime**: 14 classes.

Both datasets use pre-extracted CLIP visual features (ViT-B/16, 512-d), with VGGish audio for XD-Violence.

## Evaluation

APD-VAD is evaluated with the standard weakly-supervised VAD metrics:

- **AUC** and **AP** for frame-level anomaly detection.
- **mAP** for temporal anomaly localization, averaged over multiple IoU thresholds.

## Results

APD-VAD achieves competitive performance against state-of-the-art methods on the XD-Violence and UCF-Crime benchmarks.

## License

This project is licensed under the MIT License.

## Citation

If you use APD-VAD in your research, please cite our paper:

```bibtex
@inproceedings{li2026apdvad,
  title={Asymmetric Privileged Distillation for Weakly Supervised Video Anomaly Detection},
  author={Li, Xusheng and Du, Haotong and Chen, Binghan and Li, Xianghua and Gao, Chao},
  booktitle={Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year={2026}
}
```