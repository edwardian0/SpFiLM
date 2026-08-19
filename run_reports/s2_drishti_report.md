# Step 2 Report: DRISHTI-GS In-Domain Baseline

## 1. Objective

This experiment extends **Step 2 of the Spatial FiLM project** to the DRISHTI-GS domain. A plain 2D U-Net was trained and evaluated in-domain before introducing Global FiLM, Spatial FiLM, or cross-domain testing.

The aim was to confirm that the existing segmentation pipeline can reliably segment the **optic disc and optic cup** on DRISHTI-GS. The project brief specifically requires the plain backbone to be validated independently on the retinal datasets before domain-shift experiments begin.

## 2. Dataset and split

DRISHTI-GS contains **101 colour fundus images**, comprising 70 glaucomatous and 31 normal eyes. The original dataset defines a **50-image training set and 51-image test set**. Images were acquired at Aravind Eye Hospital, with optic disc and cup annotations obtained from four glaucoma experts.

The present experiment preserves the original 51-image test set and divides the original 50 training images into training and validation subsets:

| Split      |  Images | Glaucoma | Normal |
| ---------- | ------: | -------: | -----: |
| Training   |      40 |       26 |     14 |
| Validation |      10 |        6 |      4 |
| Test       |      51 |       38 |     13 |
| **Total**  | **101** |   **70** | **31** |

This therefore gives a **40/10/51 train-validation-test split**, with the 50-image development partition divided 80/20 for training and validation.

Images and masks were processed as full fundus images using **aspect-preserving letterboxing to 512 × 512**, with no optic-disc ROI cropping. This retains global spatial information that will later be relevant when testing Spatial FiLM.

## 3. Model and training

The same plain 2D U-Net used for the REFUGE baseline was used here, containing **1,944,066 trainable parameters**. No conditioning was applied and deep supervision remained deferred to ensure that it can later be introduced consistently across all experimental arms.

Key training settings were:

* maximum epochs: **300**
* batch size: **8**
* initial learning rate: **1 × 10⁻³**
* weight decay: **1 × 10⁻⁵**
* early-stopping patience: **20 epochs**
* random seed: **42**
* horizontal flipping: **p = 0.5**
* rotation: **±10°**
* brightness/contrast augmentation: **0.1**

Training used the combined BCE and soft Dice objective. The best checkpoint was selected using the lowest validation loss.

Training stopped after **epoch 146**, with the best model selected at **epoch 126**:

* validation loss: **0.1467**
* validation Dice, disc: **0.9507**
* validation Dice, cup: **0.8249**

Total training time was approximately **411 seconds** on an NVIDIA A100 GPU.

## 4. Test results

Evaluation was performed on all 51 held-out DRISHTI-GS test images using a probability threshold of 0.5.

| Structure      |                Dice |                 IoU |         HD95 (px) |            Accuracy |
| -------------- | ------------------: | ------------------: | ----------------: | ------------------: |
| **Optic disc** | **0.9510 ± 0.0624** | **0.9117 ± 0.0868** |  **8.82 ± 27.86** | **0.9972 ± 0.0051** |
| **Optic cup**  | **0.8207 ± 0.1375** | **0.7155 ± 0.1695** | **14.23 ± 28.59** | **0.9955 ± 0.0036** |

Metrics were calculated on the 512 × 512 letterboxed grid, so HD95 is reported in resized-grid pixels rather than native-image pixels or millimetres. No images were excluded from HD95 calculation.

As with REFUGE, optic disc segmentation was stronger and more consistent than optic cup segmentation. The larger standard deviation for the cup indicates substantially greater variation between test images.

This is consistent with the original DRISHTI-GS paper, which describes optic cup segmentation as the more difficult task because cup boundaries are less clearly defined and exhibit greater inter-observer variability.

## 5. Sanity check and conclusion

The original DRISHTI-GS paper reported test-set F-scores of approximately **0.96 for the optic disc** and **0.79 for the optic cup**. The current Dice results of **0.9510** and **0.8207**, respectively, are therefore within a sensible range for the dataset.

This comparison is only approximate because the original paper used its own consensus-ground-truth and evaluation procedure, while the present pipeline evaluates resized binary masks and uses HD95 as the boundary metric.

Overall, the DRISHTI-GS Step 2 baseline is successful:

**Disc Dice: 0.9510**
**Cup Dice: 0.8207**

Together with the REFUGE baseline, this confirms that the plain U-Net pipeline can learn sensible in-domain segmentations across more than one retinal imaging source. The next stage is to complete the remaining in-domain baseline and then use these results as reference points for the **cross-domain performance drop in Step 3**.

### Reproducibility note

The original DRISHTI-GS publication states that ground-truth masks were publicly released only for the 50 training images, while the 51 test labels were originally retained for server-side evaluation. The local dataset used in this run provides targets for all 51 test images, so the provenance of these test masks should be documented in the final methodology.
