k23123868@erc-hpc-login2:~/edward/logs$ cat ~/edward/spfilm/artifacts/runs/refuge_s2_36631447/summary_table.txt

# Step 2 Report: REFUGE In-Domain Baseline

## 1. Objective

This experiment forms part of **Step 2 of the Spatial FiLM project brief**, whose purpose is to validate the segmentation pipeline on a single imaging domain before introducing cross-domain evaluation or conditioning.

A plain 2D U-Net was therefore trained and tested using only the **REFUGE Zeiss domain**, with no Global FiLM or Spatial FiLM conditioning.

The aim was to establish a reliable in-domain reference performance for optic disc and optic cup segmentation.

## 2. Dataset and experimental split

The original REFUGE challenge contains **1,200 colour fundus images**, divided into three sets of 400 images. The official training set was acquired using a **Zeiss Visucam 500**, while the two challenge test sets were acquired using a **Canon CR-2**, creating a natural acquisition-domain difference.

For this baseline, only the 400 Zeiss images were used. They were divided into:

| Split      |  Images |
| ---------- | ------: |
| Training   |     256 |
| Validation |      64 |
| Test       |      80 |
| **Total**  | **400** |

This is an **internal experimental split**, rather than the original REFUGE challenge partition. Keeping all three subsets within the Zeiss source ensures that this experiment measures in-domain segmentation performance only.

Images and masks were resized using aspect-preserving letterboxing onto a **512 × 512 full-image grid**. Predictions were evaluated directly on this grid rather than resampled to their original resolution. Consequently, HD95 is reported in 512-grid pixels rather than native-image pixels or millimetres.

## 3. Model and training

A plain 2D U-Net containing **1,944,066 trainable parameters** was trained using a combined **binary cross-entropy and soft Dice loss**.

The run had a maximum training horizon of **300 epochs**, but early stopping terminated training after epoch 83. The checkpoint with the lowest validation loss occurred at **epoch 63** and was selected for final testing.

At the selected checkpoint:

* validation loss: **0.0954**
* validation Dice, optic disc: **0.9564**
* validation Dice, optic cup: **0.8705**

Training began with a learning rate of (1\times10^{-3}), which was progressively reduced as validation performance plateaued. Total training time was approximately **325 seconds on CUDA**.

Importantly, the nominal 300-epoch configuration did **not** result in 300 epochs of training. Performance had stabilised considerably earlier, and early stopping selected epoch 63.

Deep supervision remains deferred for this experiment and will be introduced to all comparison arms together later to maintain a controlled comparison.

## 4. Test results

Final evaluation was performed on the 80 held-out Zeiss images using a probability threshold of **0.5**.

| Structure      |                Dice |                 IoU |       HD95 (px) |            Accuracy |
| -------------- | ------------------: | ------------------: | --------------: | ------------------: |
| **Optic disc** | **0.9567 ± 0.0190** | **0.9176 ± 0.0334** | **4.09 ± 4.39** | **0.9986 ± 0.0006** |
| **Optic cup**  | **0.8637 ± 0.0628** | **0.7653 ± 0.0943** | **5.13 ± 2.47** | **0.9988 ± 0.0008** |

Values are mean ± standard deviation across the 80 test images. Disc and cup metrics were reported separately, with no combined Dice score. No images were excluded from either HD95 calculation.

The optic disc was segmented consistently, with a Dice score of approximately **0.96**. Optic cup segmentation remained more difficult but achieved a Dice score of approximately **0.86**, which is expected given the less distinct cup boundary in colour fundus photographs.

## 5. Interpretation and conclusion

The experiment confirms that the baseline segmentation pipeline performs reliably within the REFUGE Zeiss domain.

Increasing the maximum training horizon from the earlier configuration to 300 epochs did not require substantially longer optimisation because validation performance plateaued and early stopping selected epoch 63. The resulting test performance remained stable, with strong optic disc segmentation and sensible optic cup segmentation.

The Step 2 requirement can now be considered satisfied:

* the plain backbone trains successfully;
* the segmentation pipeline produces sensible disc and cup predictions;
* the experiment is reproducible using fixed train/validation/test partitions;
* and an in-domain reference performance has been established.

The resulting baseline is:

**Disc Dice: 0.9567**
**Cup Dice: 0.8637**

The next stage is **Step 3**, where the same plain model will be evaluated under domain shift. Training and testing will occur across different retinal acquisition domains, allowing the drop from this in-domain baseline to be quantified before Global FiLM and Spatial FiLM are introduced.


## Training metrics per epoch

  epoch |        lr | train_loss |   val_loss | val_dice_disc | val_dice_cup |     time | best
----------------------------------------------------------------------------------------------
  1/300 | 1.000e-03 |     1.6062 |     1.5374 |        0.8468 |       0.0140 |     6.5s | *
  2/300 | 1.000e-03 |     1.4922 |     1.4431 |        0.8860 |       0.0295 |     4.8s | *
  3/300 | 1.000e-03 |     1.3998 |     1.3532 |        0.8894 |       0.2479 |     4.7s | *
  4/300 | 1.000e-03 |     1.3102 |     1.2629 |        0.8656 |       0.3272 |     4.0s | *
  5/300 | 1.000e-03 |     1.2164 |     1.1730 |        0.8790 |       0.4206 |     3.7s | *
  6/300 | 1.000e-03 |     1.1230 |     1.0760 |        0.9153 |       0.6254 |     3.8s | *
  7/300 | 1.000e-03 |     1.0300 |     0.9888 |        0.8957 |       0.6907 |     3.7s | *
  8/300 | 1.000e-03 |     0.9393 |     0.8940 |        0.9038 |       0.6540 |     3.7s | *
  9/300 | 1.000e-03 |     0.8522 |     0.8094 |        0.9223 |       0.7766 |     3.8s | *
 10/300 | 1.000e-03 |     0.7651 |     0.7218 |        0.9336 |       0.7238 |     3.7s | *
 11/300 | 1.000e-03 |     0.6868 |     0.6510 |        0.9313 |       0.7225 |     3.7s | *
 12/300 | 1.000e-03 |     0.6213 |     0.5883 |        0.9362 |       0.6956 |     3.8s | *
 13/300 | 1.000e-03 |     0.5554 |     0.5256 |        0.9377 |       0.6911 |     3.8s | *
 14/300 | 1.000e-03 |     0.4928 |     0.4594 |        0.9436 |       0.7568 |     3.9s | *
 15/300 | 1.000e-03 |     0.4278 |     0.3927 |        0.9352 |       0.7843 |     3.6s | *
 16/300 | 1.000e-03 |     0.3493 |     0.3075 |        0.9432 |       0.7835 |     3.8s | *
 17/300 | 1.000e-03 |     0.2748 |     0.2461 |        0.9400 |       0.8396 |     3.8s | *
 18/300 | 1.000e-03 |     0.2220 |     0.1947 |        0.9396 |       0.8345 |     3.8s | *
 19/300 | 1.000e-03 |     0.1883 |     0.2001 |        0.9452 |       0.7674 |     3.8s | 
 20/300 | 1.000e-03 |     0.1729 |     0.1777 |        0.9351 |       0.8060 |     3.7s | *
 21/300 | 1.000e-03 |     0.1528 |     0.1545 |        0.9384 |       0.8439 |     3.8s | *
 22/300 | 1.000e-03 |     0.1466 |     0.1368 |        0.9453 |       0.8525 |     3.7s | *
 23/300 | 1.000e-03 |     0.1428 |     0.1412 |        0.9411 |       0.8265 |     3.8s | 
 24/300 | 1.000e-03 |     0.1395 |     0.1378 |        0.9474 |       0.8229 |     3.8s | 
 25/300 | 1.000e-03 |     0.1288 |     0.1213 |        0.9460 |       0.8633 |     3.8s | *
 26/300 | 1.000e-03 |     0.1258 |     0.1300 |        0.9474 |       0.8460 |     3.8s | 
 27/300 | 1.000e-03 |     0.1210 |     0.1135 |        0.9509 |       0.8640 |     3.9s | *
 28/300 | 1.000e-03 |     0.1182 |     0.1160 |        0.9506 |       0.8494 |     3.8s | 
 29/300 | 1.000e-03 |     0.1184 |     0.1155 |        0.9477 |       0.8597 |     3.6s | 
 30/300 | 1.000e-03 |     0.1166 |     0.1116 |        0.9521 |       0.8636 |     3.8s | *
 31/300 | 1.000e-03 |     0.1124 |     0.1073 |        0.9496 |       0.8669 |     3.8s | *
 32/300 | 1.000e-03 |     0.1096 |     0.1158 |        0.9515 |       0.8402 |     3.8s | 
 33/300 | 1.000e-03 |     0.1082 |     0.1078 |        0.9519 |       0.8594 |     3.7s | 
 34/300 | 1.000e-03 |     0.1063 |     0.1050 |        0.9524 |       0.8601 |     3.8s | *
 35/300 | 1.000e-03 |     0.1121 |     0.1413 |        0.9348 |       0.8242 |     3.8s | 
 36/300 | 1.000e-03 |     0.1348 |     0.1404 |        0.9364 |       0.8238 |     3.8s | 
 37/300 | 1.000e-03 |     0.1156 |     0.1088 |        0.9494 |       0.8577 |     3.8s | 
 38/300 | 5.000e-04 |     0.1102 |     0.1077 |        0.9496 |       0.8602 |     3.7s | 
 39/300 | 5.000e-04 |     0.1061 |     0.1058 |        0.9514 |       0.8574 |     3.7s | 
 40/300 | 5.000e-04 |     0.1019 |     0.1062 |        0.9510 |       0.8606 |     3.8s | 
 41/300 | 5.000e-04 |     0.0998 |     0.1035 |        0.9525 |       0.8614 |     3.7s | *
 42/300 | 5.000e-04 |     0.1015 |     0.1028 |        0.9526 |       0.8633 |     3.8s | *
 43/300 | 5.000e-04 |     0.1004 |     0.1018 |        0.9530 |       0.8655 |     3.8s | *
 44/300 | 5.000e-04 |     0.1000 |     0.1031 |        0.9530 |       0.8601 |     3.8s | 
 45/300 | 5.000e-04 |     0.0986 |     0.1019 |        0.9532 |       0.8611 |     3.7s | 
 46/300 | 5.000e-04 |     0.0981 |     0.0990 |        0.9544 |       0.8661 |     3.8s | *
 47/300 | 5.000e-04 |     0.0957 |     0.1029 |        0.9532 |       0.8586 |     3.9s | 
 48/300 | 5.000e-04 |     0.0940 |     0.1014 |        0.9532 |       0.8655 |     3.9s | 
 49/300 | 5.000e-04 |     0.0947 |     0.1026 |        0.9540 |       0.8591 |     3.9s | 
 50/300 | 2.500e-04 |     0.0943 |     0.1012 |        0.9543 |       0.8601 |     3.8s | 
 51/300 | 2.500e-04 |     0.0928 |     0.0974 |        0.9554 |       0.8691 |     3.8s | *
 52/300 | 2.500e-04 |     0.0897 |     0.1003 |        0.9529 |       0.8658 |     3.7s | 
 53/300 | 2.500e-04 |     0.0899 |     0.0974 |        0.9546 |       0.8722 |     3.8s | 
 54/300 | 2.500e-04 |     0.0888 |     0.0963 |        0.9554 |       0.8703 |     3.7s | *
 55/300 | 2.500e-04 |     0.0880 |     0.0963 |        0.9562 |       0.8685 |     3.8s | 
 56/300 | 2.500e-04 |     0.0879 |     0.0974 |        0.9546 |       0.8689 |     3.8s | 
 57/300 | 2.500e-04 |     0.0882 |     0.0978 |        0.9560 |       0.8641 |     3.8s | 
 58/300 | 1.250e-04 |     0.0884 |     0.0981 |        0.9552 |       0.8686 |     3.9s | 
 59/300 | 1.250e-04 |     0.0889 |     0.0986 |        0.9549 |       0.8646 |     3.8s | 
 60/300 | 1.250e-04 |     0.0878 |     0.0984 |        0.9560 |       0.8629 |     3.7s | 
 61/300 | 1.250e-04 |     0.0856 |     0.0973 |        0.9562 |       0.8652 |     3.7s | 
 62/300 | 1.250e-04 |     0.0856 |     0.0961 |        0.9561 |       0.8716 |     3.8s | *
 63/300 | 1.250e-04 |     0.0872 |     0.0954 |        0.9564 |       0.8705 |     3.9s | *
 64/300 | 1.250e-04 |     0.0859 |     0.0965 |        0.9555 |       0.8712 |     3.8s | 
 65/300 | 1.250e-04 |     0.0849 |     0.0966 |        0.9558 |       0.8685 |     3.8s | 
 66/300 | 1.250e-04 |     0.0847 |     0.0967 |        0.9563 |       0.8671 |     3.7s | 
 67/300 | 6.250e-05 |     0.0844 |     0.0983 |        0.9561 |       0.8629 |     3.7s | 
 68/300 | 6.250e-05 |     0.0836 |     0.0962 |        0.9567 |       0.8694 |     3.8s | 
 69/300 | 6.250e-05 |     0.0828 |     0.0979 |        0.9558 |       0.8650 |     3.8s | 
 70/300 | 6.250e-05 |     0.0827 |     0.0963 |        0.9563 |       0.8689 |     3.8s | 
 71/300 | 3.125e-05 |     0.0826 |     0.0971 |        0.9564 |       0.8659 |     3.8s | 
 72/300 | 3.125e-05 |     0.0826 |     0.0968 |        0.9563 |       0.8678 |     3.8s | 
 73/300 | 3.125e-05 |     0.0815 |     0.0967 |        0.9563 |       0.8683 |     3.8s | 
 74/300 | 3.125e-05 |     0.0817 |     0.0966 |        0.9567 |       0.8669 |     3.8s | 
 75/300 | 1.563e-05 |     0.0816 |     0.0964 |        0.9566 |       0.8679 |     3.8s | 
 76/300 | 1.563e-05 |     0.0812 |     0.0963 |        0.9564 |       0.8689 |     3.8s | 
 77/300 | 1.563e-05 |     0.0814 |     0.0963 |        0.9566 |       0.8687 |     3.8s | 
 78/300 | 1.563e-05 |     0.0813 |     0.0963 |        0.9565 |       0.8686 |     3.8s | 
 79/300 | 7.813e-06 |     0.0806 |     0.0965 |        0.9565 |       0.8682 |     3.8s | 
 80/300 | 7.813e-06 |     0.0812 |     0.0964 |        0.9565 |       0.8686 |     3.9s | 
 81/300 | 7.813e-06 |     0.0813 |     0.0964 |        0.9566 |       0.8685 |     3.7s | 
 82/300 | 7.813e-06 |     0.0819 |     0.0965 |        0.9566 |       0.8682 |     3.7s | 
 83/300 | 3.906e-06 |     0.0810 |     0.0966 |        0.9565 |       0.8680 |     3.8s | 
early_stopping best_epoch=63