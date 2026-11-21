# Interactive Medical Image Segmentation

An educational research prototype that predicts lung fields in chest X-rays and lets a user refine the mask with positive and negative point clicks. The repository includes a trained checkpoint, Streamlit interface, evaluation artifacts, presentation, and report. **You do not need to train the model to run the demo.**

This is not a medical device and is not validated for clinical use.

## Run the demo on macOS

Tested project configuration: Python 3.11 and PyTorch 2.8.0. macOS with Apple Silicon can use PyTorch's MPS backend when available; Intel Macs use CPU. The included model checkpoint means no dataset download or training is needed just to try the app.

1. Install Python 3.11 or newer from [python.org](https://www.python.org/downloads/macos/) or with Homebrew.
2. In Terminal, run:

```bash
git clone https://github.com/godnix7/image-and-analysis-project.git
cd image-and-analysis-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

3. Open the local address printed by Streamlit (normally `http://localhost:8501`). Choose **Upload my own image** and select a PNG, JPG/JPEG, BMP, or TIFF chest X-ray. Select **Add missed lung area** or **Remove wrongly included area**, then click that point on the image. The model updates immediately. **Undo last click** removes one correction; **Clear all clicks** resets the prediction.

For a comparison score, expand **Compare with a reference mask** and upload a matching binary reference mask. Without a reference mask, the app shows the prediction but does not report Dice/IoU.

### No-training requirement

The demo loads `checkpoints/best.pt`, the checkpoint selected from the completed training run. Training is only needed if you want to reproduce or change the experiment. The included checkpoint is about 17 MB; allow a little time for the first model load on a Mac.

## What model is used?

The project uses a custom, compact **2D U-Net** built with PyTorch. It is trained from scratch; it is not a pretrained foundation model and does not call a cloud AI service.

- Four encoder stages use 24, 48, 96, and 192 feature channels, followed by a 384-channel bottleneck and a skip-connected decoder.
- Each convolution block contains two 3 x 3 convolutions with batch normalization and ReLU.
- The four input channels are the grayscale X-ray, a positive-click map, a negative-click map, and the previous probability mask. The output is one lung-mask logit channel.
- Images are resized to 256 x 256. Clicks are encoded as Gaussian maps with a 12-pixel standard deviation.
- Training uses binary cross-entropy plus soft Dice loss, AdamW, and ReduceLROnPlateau. Simulated training clicks are sampled from the largest false-negative or false-positive error region.
- Device selection is automatic: CUDA, Apple Silicon MPS, then CPU.

At inference, the model predicts a mask from the image and empty click/prior channels. Each user click is encoded, the model predicts again, and that prediction becomes the prior mask for the next click.

## Reproduce training (optional)

The training experiment used the NLM Montgomery County CXR collection: 138 radiographs and left/right lung masks fused into binary masks. With seed 42, the image-pair split was 96 training, 21 validation, and 21 test images. Because the collection has 135 patients, this was not a patient-disjoint split.

The NLM dataset README asks users not to share the dataset outside their research group or organization. **Raw images and masks are not included in this repository.** Confirm access with the data provider before obtaining or sharing the data. If authorized, place it locally with:

```bash
python data/download_montgomery.py
python data/prepare_montgomery_masks.py
```

Then train and evaluate:

```bash
python train.py --data data/MontgomerySet/CXR_png --epochs 20 --batch-size 4 --size 256
python evaluate.py --data data/MontgomerySet/CXR_png --checkpoint checkpoints/best.pt
```

The downloader uses `requests` to retrieve the NLM files. Training can take a while on a laptop, especially on CPU; reduce `--batch-size` or `--size` if memory is limited. The Mac demo itself does not require these commands.

## Results and artifacts

On the held-out 21-image split, mean Dice/IoU were:

| Simulated clicks | Dice | IoU |
|---:|---:|---:|
| 0 | 0.662 | 0.504 |
| 1 | 0.938 | 0.884 |
| 3 | 0.928 | 0.866 |
| 5 | 0.909 | 0.834 |
| 10 | 0.886 | 0.798 |

These are exploratory results on a small, single-source test set. The offline interaction evaluation chooses clicks using test-mask errors (oracle clicks), not real users; the later click scores are not monotonic. There is no external validation, confidence interval, or clinical evaluation.

- `Interactive_Medical_Image_Segmentation_Presentation.pptx` — editable project presentation.
- `Interactive_Medical_Image_Segmentation_Report.pdf` — project report.
- `checkpoints/best.pt` — trained checkpoint used by the demo.
- `results/evaluation/metrics.json` — aggregate and per-image evaluation metrics.
- `results/graphs/` — training and interaction curves.

The NLM data provider's README asks that publications cite the Montgomery dataset papers and attribute the National Library of Medicine. See the [official dataset README](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/NLM-MontgomeryCXRSet-ReadMe.pdf).

## Useful commands

```bash
python predict.py --image path/to/chest_xray.png --checkpoint checkpoints/best.pt --output prediction.png
python -m unittest discover -s tests -v
python build_deliverables.py  # rebuild the PDF report
```

The PowerPoint and PDF are already included; Node.js and the Codex artifact runtime are only needed to regenerate the editable presentation with `build_presentation.mjs`.

## Troubleshooting

- **`checkpoints/best.pt` not found:** run the command from the cloned project folder and confirm that the checkpoint file was downloaded by Git.
- **PyTorch install fails:** update Python and pip, then rerun `python -m pip install -r requirements.txt` inside the active `.venv`.
- **App feels slow:** CPU inference is supported. Apple Silicon uses MPS automatically if the installed PyTorch build exposes it.
- **No score for an uploaded X-ray:** upload its matching binary reference mask in the **Compare with a reference mask** expander.

