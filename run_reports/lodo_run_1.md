aggregated 20 Stage 3 run(s) across 1 arm(s): stage3_lodo_plain_unet
warning: 20 run(s) were produced from a dirty working tree, so the recorded
  commit does not fully identify the code that ran:
  stage3_lodo_plain_unet/drishti_gs/seed_42,
  stage3_lodo_plain_unet/drishti_gs/seed_43,
  stage3_lodo_plain_unet/drishti_gs/seed_44,
  stage3_lodo_plain_unet/drishti_gs/seed_45,
  stage3_lodo_plain_unet/drishti_gs/seed_46,
  stage3_lodo_plain_unet/refuge_canon_val/seed_42,
  stage3_lodo_plain_unet/refuge_canon_val/seed_43,
  stage3_lodo_plain_unet/refuge_canon_val/seed_44,
  stage3_lodo_plain_unet/refuge_canon_val/seed_45,
  stage3_lodo_plain_unet/refuge_canon_val/seed_46,
  stage3_lodo_plain_unet/refuge_zeiss/seed_42,
  stage3_lodo_plain_unet/refuge_zeiss/seed_43,
  stage3_lodo_plain_unet/refuge_zeiss/seed_44,
  stage3_lodo_plain_unet/refuge_zeiss/seed_45,
  stage3_lodo_plain_unet/refuge_zeiss/seed_46,
  stage3_lodo_plain_unet/rim_one_dl/seed_42,
  stage3_lodo_plain_unet/rim_one_dl/seed_43,
  stage3_lodo_plain_unet/rim_one_dl/seed_44,
  stage3_lodo_plain_unet/rim_one_dl/seed_45,
  stage3_lodo_plain_unet/rim_one_dl/seed_46

## held-out domain: drishti_gs    arm: stage3_lodo_plain_unet
  51 locked test images; seeds 42, 43, 44, 45, 46 (n=5)
  intervals are 95% CI over the per-seed means
  HD95 unit: letterboxed-grid pixels
  structure metric        mean                  95% CI   sd(seeds)
  disc      Dice        0.7589        [0.6207, 0.8972]       0.111
  disc      IoU         0.6882        [0.5352, 0.8412]       0.123
  disc      HD95         80.19         [30.11, 130.27]        40.3
  cup       Dice        0.6350        [0.6052, 0.6649]       0.024
  cup       IoU         0.4992        [0.4667, 0.5317]      0.0262
  cup       HD95         48.93          [25.55, 72.32]        18.8
  HD95 (disc): finite 51/51 per seed, the same images every seed
  HD95 (cup): finite 51/51 per seed, the same images every seed

## held-out domain: refuge_canon_val    arm: stage3_lodo_plain_unet
  80 locked test images; seeds 42, 43, 44, 45, 46 (n=5)
  intervals are 95% CI over the per-seed means
  HD95 unit: letterboxed-grid pixels
  structure metric        mean                  95% CI   sd(seeds)
  disc      Dice        0.8684        [0.7854, 0.9515]      0.0669
  disc      IoU         0.7938        [0.7005, 0.8871]      0.0752
  disc      HD95         35.50           [2.05, 68.94]        26.9
  cup       Dice        0.7184        [0.6206, 0.8162]      0.0788
  cup       IoU         0.5894        [0.4869, 0.6919]      0.0825
  cup       HD95         19.85           [4.86, 34.84]        12.1
  HD95 (disc): finite 80/80 per seed, the same images every seed
  HD95 (cup): finite 80/80 per seed, the same images every seed

## held-out domain: refuge_zeiss    arm: stage3_lodo_plain_unet
  80 locked test images; seeds 42, 43, 44, 45, 46 (n=5)
  intervals are 95% CI over the per-seed means
  HD95 unit: letterboxed-grid pixels
  structure metric        mean                  95% CI   sd(seeds)
  disc      Dice        0.8834        [0.8614, 0.9055]      0.0178
  disc      IoU         0.8109        [0.7811, 0.8408]       0.024
  disc      HD95         25.19           [8.59, 41.79]        13.4
  cup       Dice        0.7139        [0.6661, 0.7616]      0.0385
  cup       IoU         0.5761        [0.5180, 0.6342]      0.0468
  cup       HD95         17.46          [10.86, 24.05]        5.31
  HD95 (disc): finite 80/80 per seed, the same images every seed
  HD95 (cup): finite 80/80 per seed, the same images every seed

## held-out domain: rim_one_dl    arm: stage3_lodo_plain_unet
  97 locked test images; seeds 42, 43, 44, 45, 46 (n=5)
  intervals are 95% CI over the per-seed means
  HD95 unit: native pixels
  structure metric        mean                  95% CI   sd(seeds)
  disc      Dice        0.0537        [0.0359, 0.0715]      0.0143
  disc      IoU         0.0279        [0.0185, 0.0373]     0.00756
  disc      HD95        184.07        [159.30, 208.84]        19.9
  cup       Dice        0.0247       [-0.0090, 0.0584]      0.0272
  cup       IoU         0.0135       [-0.0054, 0.0325]      0.0152
  cup       HD95        241.30        [197.01, 285.60]        35.7
  cup       HD95*       239.23        [196.83, 281.64]        34.2
  HD95* is the same metric restricted to the images finite in every seed
  HD95 (disc): finite 97/97 per seed, the same images every seed
  HD95 (cup): finite 94-97/97 per seed; only 93/97 are finite in all 5 seeds,
      so the means above are taken over 5 different image sets and are not
      directly comparable across seeds; the HD95* row restricts every seed to
      the common images

paired between-arm test not run: the paired comparison needs two conditioning
  arms scored on the same locked folds, and only ['stage3_lodo_plain_unet']
  has been run. The substrate is built and ready; pass --paired-arms A B once
  the second arm exists.

wrote artifacts/stage3_lodo_summary.json
wrote artifacts/stage3_lodo_summary.csv
wrote run_reports/s3_lodo_combined_report.md
(spfilm) k23123868@erc-hpc-login2:~/edward/spfilm$