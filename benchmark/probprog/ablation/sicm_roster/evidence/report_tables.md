### Headline table (batch batch4, last-trial statistic)

| Pattern | Model | Size | SICM (s) | No SICM (s) | Speedup |
|---|---|---|---|---|---|
| CholScale | Leroux CAR, Scotland lip cancer | n=56 | 1.85 | 10.10 | 5.5x |
| CholScale | Leroux CAR, NYC traffic sub-map | n=250 | 153.06 | 600.30 | 3.9x |
| OuterProdScale | Hierarchical MVN | K=60, J=2 | 3.81 | 17.53 | 4.6x |
| EigenLift | Marginalized LMM, MN radon | n=250 | 0.15 | 6.46 | 43.1x |

### Both statistics, every batch

| Model | Batch | SICM trials (s) | SICM last | SICM median | No SICM trials (s) | No SICM last | No SICM median | Speedup last | Speedup median |
|---|---|---|---|---|---|---|---|---|---|
| Leroux CAR, Scotland lip cancer | batch1 | 2.067; 1.915 | 1.915 | 1.991 | 10.442; 10.061 | 10.061 | 10.251 | 5.3x | 5.1x |
| Leroux CAR, Scotland lip cancer | batch2 | 2.102; 1.906 | 1.906 | 2.004 | 10.551; 10.063 | 10.063 | 10.307 | 5.3x | 5.1x |
| Leroux CAR, Scotland lip cancer | batch3 | 2.101; 1.769 | 1.769 | 1.935 | 10.546; 10.500 | 10.500 | 10.523 | 5.9x | 5.4x |
| Leroux CAR, Scotland lip cancer | batch4 | 2.071; 1.846 | 1.846 | 1.959 | 10.506; 10.099 | 10.099 | 10.303 | 5.5x | 5.3x |
| Leroux CAR, NYC traffic sub-map | batch4 | 151.390; 153.062 | 153.062 | 152.226 | 631.992; 600.298 | 600.298 | 616.145 | 3.9x | 4.0x |
| Hierarchical MVN | batch1 | 4.324; 4.080 | 4.080 | 4.202 | 18.010; 16.903 | 16.903 | 17.457 | 4.1x | 4.2x |
| Hierarchical MVN | batch2 | 4.219; 3.954 | 3.954 | 4.086 | 18.325; 17.305 | 17.305 | 17.815 | 4.4x | 4.4x |
| Hierarchical MVN | batch4 | 4.255; 3.808 | 3.808 | 4.031 | 17.828; 17.529 | 17.529 | 17.679 | 4.6x | 4.4x |
| Marginalized LMM, MN radon | batch1 | 0.619; 0.152 | 0.152 | 0.385 | 6.583; 5.179 | 5.179 | 5.881 | 34.2x | 15.3x |
| Marginalized LMM, MN radon | batch2 | 0.551; 0.148 | 0.148 | 0.350 | 7.512; 8.048 | 8.048 | 7.780 | 54.2x | 22.2x |
| Marginalized LMM, MN radon | batch4 | 0.596; 0.150 | 0.150 | 0.373 | 8.150; 6.456 | 6.456 | 7.303 | 43.1x | 19.6x |
| Phylogenetic Poisson regression | batch1 | 19.206; 20.354 | 20.354 | 19.780 | 459.085; 445.756 | 445.756 | 452.420 | 21.9x | 22.9x |
| Phylogenetic Poisson regression | batch2 | 19.204; 20.914 | 20.914 | 20.059 | 466.542; 444.516 | 444.516 | 455.529 | 21.3x | 22.7x |

### Mixing / movement gate (post-warmup, both arms, every batch)

| Model | Batch | Arm | n_eff min/med/max | Rhat min/med/max | min unique draws per chain (of N) | PLL[-1] per chain |
|---|---|---|---|---|---|---|
| Leroux CAR, Scotland lip cancer | batch1 | SICM | 345.1 / 566.2 / 607.4 | 0.9977 / 1.0006 / 1.0055 | 300 (alpha, of 300) | -125.89, -125.66 |
| Leroux CAR, Scotland lip cancer | batch1 | No SICM | 372.3 / 459.4 / 541.5 | 1.0019 / 1.0048 / 1.0085 | 300 (alpha, of 300) | -124.67, -126.25 |
| Leroux CAR, Scotland lip cancer | batch2 | SICM | 345.1 / 566.2 / 607.4 | 0.9977 / 1.0006 / 1.0055 | 300 (alpha, of 300) | -125.89, -125.66 |
| Leroux CAR, Scotland lip cancer | batch2 | No SICM | 372.3 / 459.4 / 541.5 | 1.0019 / 1.0048 / 1.0085 | 300 (alpha, of 300) | -124.67, -126.25 |
| Leroux CAR, Scotland lip cancer | batch3 | SICM | 345.1 / 566.2 / 607.4 | 0.9977 / 1.0006 / 1.0055 | 300 (alpha, of 300) | -125.89, -125.66 |
| Leroux CAR, Scotland lip cancer | batch3 | No SICM | 372.3 / 459.4 / 541.5 | 1.0019 / 1.0048 / 1.0085 | 300 (alpha, of 300) | -124.67, -126.25 |
| Leroux CAR, Scotland lip cancer | batch4 | SICM | 345.1 / 566.2 / 607.4 | 0.9977 / 1.0006 / 1.0055 | 300 (alpha, of 300) | -125.89, -125.66 |
| Leroux CAR, Scotland lip cancer | batch4 | No SICM | 372.3 / 459.4 / 541.5 | 1.0019 / 1.0048 / 1.0085 | 300 (alpha, of 300) | -124.67, -126.25 |
| Leroux CAR, NYC traffic sub-map | batch4 | SICM | 34.4 / 39.1 / 273.3 | 0.9998 / 1.0139 / 1.0373 | 300 (alpha, of 300) | -468.30, -458.81 |
| Leroux CAR, NYC traffic sub-map | batch4 | No SICM | 3.4 / 3.9 / 155.1 | 1.0190 / 1.4977 / 1.6529 | 300 (alpha, of 300) | -469.02, -464.52 |
| Hierarchical MVN | batch1 | SICM | 4.4 / 28.9 / 84.9 | 0.9986 / 1.0612 / 1.4499 | 150 (sigma_obs, of 150) | -165.06, -168.84 |
| Hierarchical MVN | batch1 | No SICM | 3.9 / 25.7 / 122.6 | 0.9972 / 1.0704 / 1.5108 | 150 (sigma_obs, of 150) | -158.49, -172.46 |
| Hierarchical MVN | batch2 | SICM | 4.4 / 28.9 / 84.9 | 0.9986 / 1.0612 / 1.4499 | 150 (sigma_obs, of 150) | -165.06, -168.84 |
| Hierarchical MVN | batch2 | No SICM | 3.9 / 25.7 / 122.6 | 0.9972 / 1.0704 / 1.5108 | 150 (sigma_obs, of 150) | -158.49, -172.46 |
| Hierarchical MVN | batch4 | SICM | 4.4 / 28.9 / 84.9 | 0.9986 / 1.0612 / 1.4499 | 150 (sigma_obs, of 150) | -165.06, -168.84 |
| Hierarchical MVN | batch4 | No SICM | 3.9 / 25.7 / 122.6 | 0.9972 / 1.0704 / 1.5108 | 150 (sigma_obs, of 150) | -158.49, -172.46 |
| Marginalized LMM, MN radon | batch1 | SICM | 367.2 / 516.8 / 591.1 | 0.9983 / 1.0029 / 1.0044 | 297 (sigma_e, of 300) | -290.70, -290.66 |
| Marginalized LMM, MN radon | batch1 | No SICM | 409.9 / 433.0 / 521.0 | 1.0035 / 1.0058 / 1.0068 | 295 (sigma_e, of 300) | -290.67, -289.94 |
| Marginalized LMM, MN radon | batch2 | SICM | 367.2 / 516.8 / 591.1 | 0.9983 / 1.0029 / 1.0044 | 297 (sigma_e, of 300) | -290.70, -290.66 |
| Marginalized LMM, MN radon | batch2 | No SICM | 409.9 / 433.0 / 521.0 | 1.0035 / 1.0058 / 1.0068 | 295 (sigma_e, of 300) | -290.67, -289.94 |
| Marginalized LMM, MN radon | batch4 | SICM | 367.2 / 516.8 / 591.1 | 0.9983 / 1.0029 / 1.0044 | 297 (sigma_e, of 300) | -290.70, -290.66 |
| Marginalized LMM, MN radon | batch4 | No SICM | 409.9 / 433.0 / 521.0 | 1.0035 / 1.0058 / 1.0068 | 295 (sigma_e, of 300) | -290.67, -289.94 |
| Phylogenetic Poisson regression | batch1 | SICM | 10.7 / 69.0 / 161.9 | 0.9988 / 1.0260 / 1.1587 | 300 (b0, of 300) | -313.43, -312.92 |
| Phylogenetic Poisson regression | batch1 | No SICM | 7.1 / 68.1 / 308.9 | 0.9983 / 1.0254 / 1.2284 | 300 (b0, of 300) | -312.03, -318.57 |
| Phylogenetic Poisson regression | batch2 | SICM | 10.7 / 69.0 / 161.9 | 0.9988 / 1.0260 / 1.1587 | 300 (b0, of 300) | -313.43, -312.92 |
| Phylogenetic Poisson regression | batch2 | No SICM | 7.1 / 68.1 / 308.9 | 0.9983 / 1.0254 / 1.2284 | 300 (b0, of 300) | -312.03, -318.57 |

### Machine record

| Batch | md5 start | md5 end | load1 at start | load1 at end | window |
|---|---|---|---|---|---|
| batch1 | 21a6bd08 | 21a6bd08 | 3.94 | 71.81 | 2026-08-25T01:03:43-05:00 -> 2026-08-25T01:27:03-05:00 |
| batch2 | 21a6bd08 | 21a6bd08 | 5.39 | 72.32 | 2026-08-25T01:30:19-05:00 -> 2026-08-25T01:53:45-05:00 |
| batch3 | 21a6bd08 | ? | 7.12 | ? | 2026-08-25T01:58:52-05:00 -> ? |
| batch4 | 21a6bd08 | 21a6bd08 | 21.81 | 27.16 | 2026-08-25T02:28:26-05:00 -> 2026-08-25T03:01:35-05:00 |

### Per-cell load record

| Batch | Cell | load1 before | load1 after | window | script exit |
|---|---|---|---|---|---|
| batch1 | car_spatial_scotland | 3.87 | 18.65 | 2026-08-25T01:03:51-05:00 -> 2026-08-25T01:05:50-05:00 | COMMAND_EXIT_CODE="0" |
| batch1 | hierarchical_mvn | 18.09 | 32.87 | 2026-08-25T01:07:22-05:00 -> 2026-08-25T01:09:14-05:00 | COMMAND_EXIT_CODE="0" |
| batch1 | lmm_radon | 16.32 | 20.92 | 2026-08-25T01:05:59-05:00 -> 2026-08-25T01:07:14-05:00 | COMMAND_EXIT_CODE="0" |
| batch1 | phylo_pois | 28.21 | 98.89 | 2026-08-25T01:09:23-05:00 -> 2026-08-25T01:26:55-05:00 | COMMAND_EXIT_CODE="0" |
| batch2 | car_spatial_scotland | 5.39 | 19.67 | 2026-08-25T01:30:27-05:00 -> 2026-08-25T01:32:27-05:00 | COMMAND_EXIT_CODE="0" |
| batch2 | hierarchical_mvn | 25.16 | 30.21 | 2026-08-25T01:34:01-05:00 -> 2026-08-25T01:35:53-05:00 | COMMAND_EXIT_CODE="0" |
| batch2 | lmm_radon | 18.34 | 26.91 | 2026-08-25T01:32:35-05:00 -> 2026-08-25T01:33:53-05:00 | COMMAND_EXIT_CODE="0" |
| batch2 | phylo_pois | 26.03 | 92.13 | 2026-08-25T01:36:02-05:00 -> 2026-08-25T01:53:38-05:00 | COMMAND_EXIT_CODE="0" |
| batch3 | car_spatial_scotland | 7.12 | 30.57 | 2026-08-25T01:59:00-05:00 -> 2026-08-25T02:01:10-05:00 | COMMAND_EXIT_CODE="0" |
| batch4 | car_spatial_scotland | 21.81 | 23.63 | 2026-08-25T02:28:34-05:00 -> 2026-08-25T02:30:37-05:00 | COMMAND_EXIT_CODE="0" |
| batch4 | car_spatial_nyc | 21.98 | 91.45 | 2026-08-25T02:30:45-05:00 -> 2026-08-25T02:57:59-05:00 | COMMAND_EXIT_CODE="0" |
| batch4 | hierarchical_mvn | 35.70 | 36.84 | 2026-08-25T02:59:35-05:00 -> 2026-08-25T03:01:27-05:00 | COMMAND_EXIT_CODE="0" |
| batch4 | lmm_radon | 84.36 | 41.46 | 2026-08-25T02:58:07-05:00 -> 2026-08-25T02:59:26-05:00 | COMMAND_EXIT_CODE="0" |
