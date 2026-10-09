# Image Reconstruction GAN

This code trains a 3D residual U-Net with a 3D PatchGAN to reconstruct SUB2-SUB5 from pre-contrast T1WI, T2WI, DWI, and acquired SUB1.

The YAML contains adjustable public training defaults. No patient data, actual study splits, trained weights, or private experiment settings are included.

## 1. Set up Anaconda on Windows 11

Open **Anaconda Prompt** and run the following commands. Change the project path if you saved it elsewhere.

```bat
cd /d E:\Image_Reconstruction_GAN
conda create -n image_gan python=3.10 -y
conda activate image_gan
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

The last command should print `True` for NVIDIA GPU training. The PyTorch installation command follows the [official Windows instructions](https://pytorch.org/get-started/previous-versions/#v231).

## 2. Prepare your images

Use anonymized NIfTI images that have already been aligned and cropped to **21 slices x 128 x 128 pixels**. All images and masks for a patient must share the same size and physical coordinates. Registration and cropping are not included here.

Put one patient's files in one folder. Use the following names, replacing `example001` with your own anonymous patient ID:

```text
data/cases/example001/
    example001-T1_PRE_0.nii.gz
    example001-T2_T2_0.nii.gz
    example001-DWI_DWI_0.nii.gz
    example001-SUB_SUB_1.nii.gz
    example001-SUB_SUB_2.nii.gz
    example001-SUB_SUB_3.nii.gz
    example001-SUB_SUB_4.nii.gz
    example001-SUB_SUB_5.nii.gz
    roi_example001-T1_POST_2.nii.gz
    roi_example001-T1_POST_3.nii.gz
    roi_example001-T1_POST_4.nii.gz
    roi_example001-T1_POST_5.nii.gz
```

The four `roi_` files are binary lesion masks for phases 2-5. They are used for training losses, not as network input channels. The code handles intensity normalization and random flipping.

Open **configs/reconstruction_gan.yaml** and change `data.lesion_root` to your data folder. In YAML, use forward slashes in Windows paths, for example `E:/MyImages/cases`.

## 3. Divide patients into training, validation, and test sets

```bat
python scripts/make_splits.py
```

The default is **80% / 10% / 10%**, with no patient shared between sets. Use at least 10 complete cases for this example split. The script creates `outputs/patient_split.json` from your own data. Training reuses this file; no real study split is supplied. To choose different proportions, edit `split` in the YAML before running the script.

## 4. Train

```bat
python scripts/train.py
```

The defaults are **GAN enabled, 220 epochs, batch size 8, AdamW, and learning rate 0.000025**. Training starts from random weights. All settings are in the single YAML; there are no inherited configuration files. The settings are public defaults, not an archive of the parameters used to produce published results.

The first run saves weights in `outputs/exp/checkpoints/`. Use `best.pt` for inference; it is selected using validation reconstruction loss. Later runs use `exp2`, `exp3`, and so on. Training progress is saved in `metrics/train_history.csv` inside each run folder.

If GPU memory is insufficient, lower the batch size:

```bat
python scripts/train.py --batch-size 2
```

To continue an interrupted run:

```bat
python scripts/train.py --resume outputs/exp/checkpoints/last.pt --epochs 220
```

Use checkpoints that you created or obtained from a trusted source.

## 5. Reconstruct a new case

The new case needs only **T1 pre-contrast, T2WI, DWI, and SUB1**, named as above. ROI masks and SUB2-SUB5 are not required.

```bat
python scripts/inference.py --case-root data/cases/example001 --checkpoint outputs/exp/checkpoints/best.pt --output-dir predictions/example001
```

Replace the case and checkpoint paths with your own. If you changed model or normalization settings during training, add `--config outputs/exp/run_config.yaml` to use that run's settings.

The output folder contains **pred_SUB2, pred_SUB3, pred_SUB4, and pred_SUB5** NIfTI images in the original intensity scale and image coordinates. These are subtraction images. Enhanced-phase images can be obtained separately as `T1 pre-contrast + predicted SUB`. Downstream image processing, TIC analysis, and reporting are not included.
