# Multimodal Hate and Abusive Content Detection

Pretrained checkpoint is included. No training needed.

- 3-class output: Abusive, Offensive, Non-abusive
- Pretrained model: processed_data/best_novel.pt
- Demo entry point: user_test_fused_lora.py (CLI + optional Gradio UI)
- Python 3.10+ recommended

---

## 1) Install packages

Pick ONE Torch wheel (GPU or CPU), then install the rest.

Colab (recommended)
```bash
# GPU (CUDA 12.4). If it errors, use the CPU line below.
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# CPU fallback:
# pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -U transformers peft gradio easyocr pandas numpy pillow
```

Local (Windows/macOS/Linux)
```bash
# (Optional) Create and activate a virtual env
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# GPU (CUDA 12.4). If it errors, use the CPU line below.
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# CPU fallback:
# pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -U transformers peft gradio easyocr pandas numpy pillow
pip install paddleocr
pip install paddlepaddle
```

If your checkpoint is NOT at processed_data/best_novel.pt, set:
- Windows (CMD): `set CKPT_PATH=D:\path\to\best_novel.pt`
- PowerShell: `$env:CKPT_PATH="D:\path\to\best_novel.pt"`
- macOS/Linux: `export CKPT_PATH=/path/to/best_novel.pt`

---

## 2) Run the demo (CLI)

Quick smoke test (runs a built-in example):
```bash
python user_test_fused_lora.py
```

Custom one-liners (no file edits needed):
- Text only
```bash
python -c "from user_test_fused_lora import user_predict; print(user_predict('I hate you', image=None))"
```

- Text + image (path to a local meme file)
```bash
python -c "from user_test_fused_lora import user_predict; print(user_predict('', image='path/to/meme.jpg'))"
```

What you’ll see:
- final_label: Abusive/Offensive/Non-abusive
- final_probs: class probabilities
- abusive_probability: p(Abusive)+p(Offensive)
- gate_weight_image: how much the image influenced the decision (~0 text-dominant, ~1 image-dominant)
- ocr_text_*: OCR info if an image is provided (EasyOCR by default)

Notes:
- If an image is provided and OCR finds text, that OCR text (normalized) is fed directly to the model.
- If OCR returns nothing, the typed text (if any) is used.

---

## 3) Run the demo (Gradio UI, optional)

1) Open user_test_fused_lora.py
2) Near the bottom, find the line with:
   - `# demo.launch(share=True)`
3) Uncomment it, then run:
```bash
python user_test_fused_lora.py
```
You’ll get a public link. In the UI:
- Type text and/or upload an image (meme)
- Toggle OCR engine (keep EasyOCR for stability)
- Optionally ignore image when OCR text is used

---

## 4) Optional: Evaluate the model (no training needed)

Generates confusion matrix, PR curve, and CSV using the same checkpoint:
```bash
python ds_multimodal_novel_lora_full_pooled.py
```

Outputs (default):
- processed_data/novel_eval_insights/fused_cm.png
- processed_data/novel_eval_insights/abusive_pr_curve.png

Tip for a quick run: open the file and set `TEACHER_N = 5000`.

---

## 5) Datasets used

- Hate Speech Detection Curated  
  https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset/data
- Hate Speech and Offensive Language Dataset  
  https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
- Jigsaw Toxic Comment Classification Challenge  
  https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
- Memotion Dataset 7K (multimodal)  
  https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k
- Suspicious Communication on Social Platforms  
  https://www.kaggle.com/datasets/syedabbasraza/suspicious-communication-on-social-platforms

---

## 6) Troubleshooting (quick)

- Checkpoint not found: verify processed_data/best_novel.pt or set CKPT_PATH.
- Torch GPU mismatch: install the CPU Torch wheel instead of a CUDA wheel.
- OCR errors: keep OCR engine as “easyocr” (default).