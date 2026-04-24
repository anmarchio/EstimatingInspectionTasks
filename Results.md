Found similarity files:
* cnn: D:\dev\EstimatingInspectionTasks\results\similarity\20260416-181811\20260115-181800_resnet.csv
* edge: D:\dev\EstimatingInspectionTasks\results\similarity\20260416-181811\20260416-181835_edgeDen.csv
* texture: D:\dev\EstimatingInspectionTasks\results\similarity\20260416-181811\20260416-181819_textComp.csv
* entropy: D:\dev\EstimatingInspectionTasks\results\similarity\20260416-181811\20260416-181811_histEnt.csv
* frequency: D:\dev\EstimatingInspectionTasks\results\similarity\20260416-181811\20260416-181934_fourFreq.csv
* superpixel: D:\dev\EstimatingInspectionTasks\results\similarity\20260416-181811\20260416-181917_noOfSup.csv

# FOR MEAN

## CORRELATION ANALYSIS

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

✅ Pearson correlation: 0.15282738031592208  (p = 8.447183043127579e-09 )

✅ Spearman correlation: 0.22038928333239893  (p = 6.298218730861326e-17 )

---
--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---

✅ Low Similarity (s < 0.5)- Pearson correlation: 0.08934884754447274  (p = 0.0018807866759571797 )

✅ Low Similarity (s < 0.5) - Spearman correlation: 0.155115161688789  (p = 6.010988723701384e-08 )

✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation: -0.08606655322435822  (p = 0.34995359216877436 )

✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation: -0.042607282476442146  (p = 0.6440354309179425 )

✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation: 0.2918442907874027  (p = 0.0095256528340452 )

✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation: 0.27451850751709683  (p = 0.015004794137957857 )

---
--- Analysis for Bin Size: 0.2 ---

Bin (-0.001, 0.2]: Pearson correlation = -0.004 (p = 0.92790), Spearman correlation = 0.034 (p = 0.48838)

Bin (0.2, 0.4]: Pearson correlation = -0.011 (p = 0.78250), Spearman correlation = 0.048 (p = 0.20826)
Bin (0.4, 0.6]: Pearson correlation = 0.018 (p = 0.80630), Spearman correlation = 0.024 (p = 0.73499)

Bin (0.6, 0.8]: Pearson correlation = 0.193 (p = 0.18832), Spearman correlation = 0.191 (p = 0.19303)

Bin (0.8, 1.0]: Pearson correlation = 0.292 (p = 0.01937), Spearman correlation = 0.315 (p = 0.01117)

---
--- Analysis for Bin Size: 0.1 ---

Bin (-0.001, 0.1]: Pearson correlation = -0.108 (p = 0.58580), Spearman correlation = -0.091 (p = 0.64665)

Bin (0.1, 0.2]: Pearson correlation = -0.077 (p = 0.12664), Spearman correlation = -0.070 (p = 0.16629)

Bin (0.2, 0.3]: Pearson correlation = 0.151 (p = 0.00106), Spearman correlation = 0.168 (p = 0.00027)

Bin (0.3, 0.4]: Pearson correlation = -0.026 (p = 0.71141), Spearman correlation = -0.015 (p = 0.83341)

Bin (0.4, 0.5]: Pearson correlation = -0.029 (p = 0.75965), Spearman correlation = 0.035 (p = 0.71322)

Bin (0.5, 0.6]: Pearson correlation = -0.083 (p = 0.45155), Spearman correlation = -0.114 (p = 0.30179)

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:420: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for bin_range, group in correlation_df.groupby('bin'):

Bin (0.6, 0.7]: Pearson correlation = 0.072 (p = 0.67673), Spearman correlation = 0.064 (p = 0.70893)

Bin (0.7, 0.8]: Pearson correlation = 0.112 (p = 0.72810), Spearman correlation = -0.057 (p = 0.86145)

Bin (0.8, 0.9]: Pearson correlation = -0.171 (p = 0.35053), Spearman correlation = -0.008 (p = 0.96657)

Bin (0.9, 1.0]: Pearson correlation = 0.405 (p = 0.02141), Spearman correlation = 0.366 (p = 0.03950)

### MULTIPLE METRICS LINEAR REGRESSION

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Pipelines total: 38

Pipelines by transfer_rate > 0.5: 32

---
Top pipelines by mean_cross_score:

                                mean_cross_score  transfer_rate  combined_score
source                                                                         

MVTec_AD_Wood_Scratch                   0.108823       0.918919        0.099999

MVTec_AD_Bottle_Broken_Sm               0.104860       0.837838        0.087856

MVTec_AD_Hazelnut_Crack                 0.084178       0.972973        0.081903

MVTec_AD_Metal_Nut                      0.079351       0.729730        0.057905

severstal-steel                         0.077334       0.783784        0.060613

MVTec_AD_Tile_Crack                     0.073861       0.540541        0.039925

MVTec_AD_Pill_Crack                     0.069799       0.837838        0.058481

AirCarbon3_80.jpg_bright                0.066951       0.891892        0.059713

MVTec_AD_Zipper_Rough                   0.063903       0.675676        0.043178

MAIPreform2_Spule0_0816_Upside          0.061231       0.675676        0.041372


---
Top pipelines by combined_score (mean * transfer_rate):

                           mean_cross_score  transfer_rate  combined_score

source                                                                    

MVTec_AD_Wood_Scratch              0.108823       0.918919        0.099999

MVTec_AD_Bottle_Broken_Sm          0.104860       0.837838        0.087856

MVTec_AD_Hazelnut_Crack            0.084178       0.972973        0.081903

severstal-steel                    0.077334       0.783784        0.060613

AirCarbon3_80.jpg_bright           0.066951       0.891892        0.059713

MVTec_AD_Pill_Crack                0.069799       0.837838        0.058481

MVTec_AD_Metal_Nut                 0.079351       0.729730        0.057905

AirCarbon3_80.jpg_dark_3           0.055090       0.891892        0.049134

AirCarbon3_80.jpg_dark_5           0.054834       0.837838        0.045942

Pultrusion_Window                  0.046199       0.972973        0.044950

### CORRELATION ANALYSIS

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Overall:

✅ Pearson correlation: 0.06823923059577064  (p = 0.010483741878230661 )

✅ Spearman correlation: 0.10460217197555477  (p = 8.511266866131907e-05 )


---
--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---

[SKIP] Not enough data for Pearson (low similarity): n=0

✅ Low Similarity (s < 0.5)- Pearson correlation: nan  (p = nan )

✅ Low Similarity (s < 0.5) - Spearman correlation: nan  (p = nan )

✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation: 0.23572927371669375  (p = 0.10675491112387495 )

✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation: 0.13262311826036058  (p = 0.36886687382777805 )

✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation: 0.06239955449162207  (p = 0.021470161918457818 )

✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation: 0.10303213946842915  (p = 0.00014268188094801232 )


---
--- Analysis for Bin Size: 0.2 ---

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:420: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for bin_range, group in correlation_df.groupby('bin'):

Bin (0.4, 0.6]: Pearson correlation = -0.381 (p = 0.45658), Spearman correlation = -0.359 (p = 0.48520)

Bin (0.6, 0.8]: Pearson correlation = 0.065 (p = 0.47551), Spearman correlation = -0.007 (p = 0.93672)

Bin (0.8, 1.0]: Pearson correlation = 0.050 (p = 0.07209), Spearman correlation = 0.096 (p = 0.00058)


---
--- Analysis for Bin Size: 0.1 ---

Bin (0.5, 0.6]: Pearson correlation = -0.381 (p = 0.45658), Spearman correlation = -0.359 (p = 0.48520)

Bin (0.6, 0.7]: Pearson correlation = 0.168 (p = 0.28899), Spearman correlation = 0.072 (p = 0.64941)

Bin (0.7, 0.8]: Pearson correlation = 0.088 (p = 0.43103), Spearman correlation = 0.038 (p = 0.73246)

Bin (0.8, 0.9]: Pearson correlation = -0.096 (p = 0.20546), Spearman correlation = -0.105 (p = 0.16724)

Bin (0.9, 1.0]: Pearson correlation = 0.036 (p = 0.23284), Spearman correlation = 0.092 (p = 0.00236)

### MULTIPLE METRICS LINEAR REGRESSION

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Pipelines total: 38

Pipelines by transfer_rate > 0.5: 32

---
Top pipelines by mean_cross_score:

                                mean_cross_score  transfer_rate  combined_score
source                                                                         

MVTec_AD_Wood_Scratch                   0.108823       0.918919        0.099999

MVTec_AD_Bottle_Broken_Sm               0.104860       0.837838        0.087856

MVTec_AD_Hazelnut_Crack                 0.084178       0.972973        0.081903

MVTec_AD_Metal_Nut                      0.079351       0.729730        0.057905

severstal-steel                         0.077334       0.783784        0.060613

MVTec_AD_Tile_Crack                     0.073861       0.540541        0.039925

MVTec_AD_Pill_Crack                     0.069799       0.837838        0.058481

AirCarbon3_80.jpg_bright                0.066951       0.891892        0.059713

MVTec_AD_Zipper_Rough                   0.063903       0.675676        0.043178

MAIPreform2_Spule0_0816_Upside          0.061231       0.675676        0.041372


---
Top pipelines by combined_score (mean * transfer_rate):
                           mean_cross_score  transfer_rate  combined_score

source                                                                    

MVTec_AD_Wood_Scratch              0.108823       0.918919        0.099999

MVTec_AD_Bottle_Broken_Sm          0.104860       0.837838        0.087856

MVTec_AD_Hazelnut_Crack            0.084178       0.972973        0.081903

severstal-steel                    0.077334       0.783784        0.060613

AirCarbon3_80.jpg_bright           0.066951       0.891892        0.059713

MVTec_AD_Pill_Crack                0.069799       0.837838        0.058481

MVTec_AD_Metal_Nut                 0.079351       0.729730        0.057905

AirCarbon3_80.jpg_dark_3           0.055090       0.891892        0.049134

AirCarbon3_80.jpg_dark_5           0.054834       0.837838        0.045942

Pultrusion_Window                  0.046199       0.972973        0.044950

### CORRELATION ANALYSIS

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Overall:

✅ Pearson correlation: 0.08021004573100962  (p = 0.00261432922281734 )

✅ Spearman correlation: 0.09500367853289078  (p = 0.0003607832953031716 )


---
--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---

[SKIP] Not enough data for Pearson (low similarity): n=0

✅ Low Similarity (s < 0.5)- Pearson correlation: nan  (p = nan )

✅ Low Similarity (s < 0.5) - Spearman correlation: nan  (p = nan )

✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation: 0.21536913711440753  (p = 0.14153372770129163 )

✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation: 0.12844399236793108  (p = 0.38427815349142735 )

✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation: 0.07365836711743817  (p = 0.006616015752420631 )

✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation: 0.08426194158878479  (p = 0.001884792133869728 )


---
--- Analysis for Bin Size: 0.2 ---

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:420: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for bin_range, group in correlation_df.groupby('bin'):

Bin (0.4, 0.6]: Pearson correlation = -0.091 (p = 0.68562), Spearman correlation = -0.046 (p = 0.84041)

Bin (0.6, 0.8]: Pearson correlation = -0.114 (p = 0.47176), Spearman correlation = -0.078 (p = 0.62441)

Bin (0.8, 1.0]: Pearson correlation = 0.059 (p = 0.02959), Spearman correlation = 0.075 (p = 0.00588)


---
--- Analysis for Bin Size: 0.1 ---

Bin (0.5, 0.6]: Pearson correlation = -0.091 (p = 0.68562), Spearman correlation = -0.046 (p = 0.84041)

Bin (0.6, 0.7]: Pearson correlation = 0.154 (p = 0.45172), Spearman correlation = 0.211 (p = 0.30125)

Bin (0.7, 0.8]: Pearson correlation = 0.324 (p = 0.22088), Spearman correlation = 0.228 (p = 0.39576)

Bin (0.8, 0.9]: Pearson correlation = -0.170 (p = 0.01867), Spearman correlation = -0.087 (p = 0.23218)

Bin (0.9, 1.0]: Pearson correlation = 0.021 (p = 0.48548), Spearman correlation = 0.035 (p = 0.23895)

### MULTIPLE METRICS LINEAR REGRESSION

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Pipelines total: 38

Pipelines by transfer_rate > 0.5: 32


---
Top pipelines by mean_cross_score:
                                mean_cross_score  transfer_rate  combined_score

source                                                                         

MVTec_AD_Wood_Scratch                   0.108823       0.918919        0.099999

MVTec_AD_Bottle_Broken_Sm               0.104860       0.837838        0.087856

MVTec_AD_Hazelnut_Crack                 0.084178       0.972973        0.081903

MVTec_AD_Metal_Nut                      0.079351       0.729730        0.057905

severstal-steel                         0.077334       0.783784        0.060613

MVTec_AD_Tile_Crack                     0.073861       0.540541        0.039925

MVTec_AD_Pill_Crack                     0.069799       0.837838        0.058481

AirCarbon3_80.jpg_bright                0.066951       0.891892        0.059713

MVTec_AD_Zipper_Rough                   0.063903       0.675676        0.043178

MAIPreform2_Spule0_0816_Upside          0.061231       0.675676        0.041372


---
Top pipelines by combined_score (mean * transfer_rate):

                           mean_cross_score  transfer_rate  combined_score
source                                                                    

MVTec_AD_Wood_Scratch              0.108823       0.918919        0.099999

MVTec_AD_Bottle_Broken_Sm          0.104860       0.837838        0.087856

MVTec_AD_Hazelnut_Crack            0.084178       0.972973        0.081903

severstal-steel                    0.077334       0.783784        0.060613

AirCarbon3_80.jpg_bright           0.066951       0.891892        0.059713

MVTec_AD_Pill_Crack                0.069799       0.837838        0.058481

MVTec_AD_Metal_Nut                 0.079351       0.729730        0.057905

AirCarbon3_80.jpg_dark_3           0.055090       0.891892        0.049134

AirCarbon3_80.jpg_dark_5           0.054834       0.837838        0.045942

Pultrusion_Window                  0.046199       0.972973        0.044950

### CORRELATION ANALYSIS

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Overall:

✅ Pearson correlation: 0.11190361790239738  (p = 2.6051065199340947e-05 )

✅ Spearman correlation: 0.15102788207552206  (p = 1.265373246855991e-08 )


---
--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---

[SKIP] Not enough data for Pearson (low similarity): n=0

✅ Low Similarity (s < 0.5)- Pearson correlation: nan  (p = nan )

✅ Low Similarity (s < 0.5) - Spearman correlation: nan  (p = nan )

✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation: -0.017473843924975772  (p = 0.929674942492129 )

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:420: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for bin_range, group in correlation_df.groupby('bin'):

✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation: 0.11193426169068095  (p = 0.5706576338761618 )

✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation: 0.11532677463741994  (p = 1.7744213081542128e-05 )

✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation: 0.15269587705609505  (p = 1.2223430275452305e-08 )


---
--- Analysis for Bin Size: 0.2 ---

Bin (0.6, 0.8]: Pearson correlation = -0.048 (p = 0.47420), Spearman correlation = 0.022 (p = 0.74113)

Bin (0.8, 1.0]: Pearson correlation = 0.117 (p = 0.00006), Spearman correlation = 0.146 (p = 0.00000)


---
--- Analysis for Bin Size: 0.1 ---

Bin (0.6, 0.7]: Pearson correlation = -0.017 (p = 0.92967), Spearman correlation = 0.112 (p = 0.57066)

Bin (0.7, 0.8]: Pearson correlation = -0.038 (p = 0.59112), Spearman correlation = 0.036 (p = 0.61347)

Bin (0.8, 0.9]: Pearson correlation = 0.143 (p = 0.00584), Spearman correlation = 0.171 (p = 0.00097)

Bin (0.9, 1.0]: Pearson correlation = 0.114 (p = 0.00122), Spearman correlation = 0.090 (p = 0.01006)

### MULTIPLE METRICS LINEAR REGRESSION

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Pipelines total: 38

Pipelines by transfer_rate > 0.5: 32

---
Top pipelines by mean_cross_score:

                                mean_cross_score  transfer_rate  combined_score

source                                                                         

MVTec_AD_Wood_Scratch                   0.108823       0.918919        0.099999

MVTec_AD_Bottle_Broken_Sm               0.104860       0.837838        0.087856

MVTec_AD_Hazelnut_Crack                 0.084178       0.972973        0.081903

MVTec_AD_Metal_Nut                      0.079351       0.729730        0.057905

severstal-steel                         0.077334       0.783784        0.060613

MVTec_AD_Tile_Crack                     0.073861       0.540541        0.039925

MVTec_AD_Pill_Crack                     0.069799       0.837838        0.058481

AirCarbon3_80.jpg_bright                0.066951       0.891892        0.059713

MVTec_AD_Zipper_Rough                   0.063903       0.675676        0.043178

MAIPreform2_Spule0_0816_Upside          0.061231       0.675676        0.041372

---
Top pipelines by combined_score (mean * transfer_rate):
                           mean_cross_score  transfer_rate  combined_score

source                                                                    

MVTec_AD_Wood_Scratch              0.108823       0.918919        0.099999

MVTec_AD_Bottle_Broken_Sm          0.104860       0.837838        0.087856

MVTec_AD_Hazelnut_Crack            0.084178       0.972973        0.081903

severstal-steel                    0.077334       0.783784        0.060613

AirCarbon3_80.jpg_bright           0.066951       0.891892        0.059713

MVTec_AD_Pill_Crack                0.069799       0.837838        0.058481

MVTec_AD_Metal_Nut                 0.079351       0.729730        0.057905

AirCarbon3_80.jpg_dark_3           0.055090       0.891892        0.049134

AirCarbon3_80.jpg_dark_5           0.054834       0.837838        0.045942

Pultrusion_Window                  0.046199       0.972973        0.044950

### CORRELATION ANALYSIS

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Overall:

✅ Pearson correlation: 0.03717033308846598  (p = 0.16361817764147576 )

✅ Spearman correlation: 0.048021822286472574  (p = 0.0718455932861254 )


---
--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---

[SKIP] Not enough data for Pearson (low similarity): n=0

✅ Low Similarity (s < 0.5)- Pearson correlation: nan  (p = nan )

✅ Low Similarity (s < 0.5) - Spearman correlation: nan  (p = nan )

✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation: nan  (p = nan )

✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation: nan  (p = nan )

✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation: 0.03570257976268976  (p = 0.18121938754008024 )

✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation: 0.0469876780983796  (p = 0.0784024262707158 )

---
--- Analysis for Bin Size: 0.2 ---

Bin (0.6, 0.8]: Pearson correlation = -0.016 (p = 0.91799), Spearman correlation = -0.038 (p = 0.80439)

Bin (0.8, 1.0]: Pearson correlation = 0.030 (p = 0.26252), Spearman correlation = 0.043 (p = 0.10936)


---
--- Analysis for Bin Size: 0.1 ---

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:355: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return pearsonr(df["x"], df["y"])

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:398: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  medium_spearman_corr, medium_spearman_p = spearmanr(medium_similarity["similarity"],

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:420: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for bin_range, group in correlation_df.groupby('bin'):

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:422: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  bin_pearson_corr, bin_pearson_p = pearsonr(group["similarity"], group["cross_score"])

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:423: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  bin_spearman_corr, bin_spearman_p = spearmanr(group["similarity"], group["cross_score"])

Bin (0.6, 0.7]: Pearson correlation = nan (p = nan), Spearman correlation = nan (p = nan)

Bin (0.7, 0.8]: Pearson correlation = -0.053 (p = 0.73104), Spearman correlation = -0.054 (p = 0.72819)

Bin (0.8, 0.9]: Pearson correlation = 0.038 (p = 0.67876), Spearman correlation = -0.028 (p = 0.75841)

Bin (0.9, 1.0]: Pearson correlation = 0.006 (p = 0.83663), Spearman correlation = 0.033 (p = 0.24849)

### MULTIPLE METRICS LINEAR REGRESSION

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Pipelines total: 38

Pipelines by transfer_rate > 0.5: 32

Top pipelines by mean_cross_score:

---
                                mean_cross_score  transfer_rate  combined_score

source                                                                         

MVTec_AD_Wood_Scratch                   0.108823       0.918919        0.099999

MVTec_AD_Bottle_Broken_Sm               0.104860       0.837838        0.087856

MVTec_AD_Hazelnut_Crack                 0.084178       0.972973        0.081903

MVTec_AD_Metal_Nut                      0.079351       0.729730        0.057905

severstal-steel                         0.077334       0.783784        0.060613

MVTec_AD_Tile_Crack                     0.073861       0.540541        0.039925

MVTec_AD_Pill_Crack                     0.069799       0.837838        0.058481

AirCarbon3_80.jpg_bright                0.066951       0.891892        0.059713

MVTec_AD_Zipper_Rough                   0.063903       0.675676        0.043178

MAIPreform2_Spule0_0816_Upside          0.061231       0.675676        0.041372


---
Top pipelines by combined_score (mean * transfer_rate):

                           mean_cross_score  transfer_rate  combined_score

source                                                                    

MVTec_AD_Wood_Scratch              0.108823       0.918919        0.099999

MVTec_AD_Bottle_Broken_Sm          0.104860       0.837838        0.087856

MVTec_AD_Hazelnut_Crack            0.084178       0.972973        0.081903

severstal-steel                    0.077334       0.783784        0.060613

AirCarbon3_80.jpg_bright           0.066951       0.891892        0.059713

MVTec_AD_Pill_Crack                0.069799       0.837838        0.058481

MVTec_AD_Metal_Nut                 0.079351       0.729730        0.057905

AirCarbon3_80.jpg_dark_3           0.055090       0.891892        0.049134

AirCarbon3_80.jpg_dark_5           0.054834       0.837838        0.045942

Pultrusion_Window                  0.046199       0.972973        0.044950 

### CORRELATION ANALYSIS

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Overall:

✅ Pearson correlation: 0.077177615831682  (p = 0.003783752031261387 )

✅ Spearman correlation: 0.07893875100382414  (p = 0.00305716984658079 )


---
--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---

[SKIP] Not enough data for Pearson (low similarity): n=0

✅ Low Similarity (s < 0.5)- Pearson correlation: nan  (p = nan )

✅ Low Similarity (s < 0.5) - Spearman correlation: nan  (p = nan )

✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation: 0.07956181455594756  (p = 0.2938787915312846 )

✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation: 0.01646629634775879  (p = 0.8282798660044169 )

✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation: 0.03567823836979987  (p = 0.21114978928396655 )

✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation: 0.06041983453249709  (p = 0.03410847780813963 )


---
--- Analysis for Bin Size: 0.2 ---

Bin (0.4, 0.6]: Pearson correlation = 0.127 (p = 0.41303), Spearman correlation = 0.023 (p = 0.88437)

Bin (0.6, 0.8]: Pearson correlation = 0.088 (p = 0.15811), Spearman correlation = 0.013 (p = 0.83127)


D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:420: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for bin_range, group in correlation_df.groupby('bin'):

Bin (0.8, 1.0]: Pearson correlation = 0.040 (p = 0.18602), Spearman correlation = 0.050 (p = 0.09862)

---
--- Analysis for Bin Size: 0.1 ---

Bin (0.5, 0.6]: Pearson correlation = 0.127 (p = 0.41303), Spearman correlation = 0.023 (p = 0.88437)

Bin (0.6, 0.7]: Pearson correlation = -0.011 (p = 0.90365), Spearman correlation = -0.021 (p = 0.80910)

Bin (0.7, 0.8]: Pearson correlation = -0.027 (p = 0.76068), Spearman correlation = 0.061 (p = 0.49444)

Bin (0.8, 0.9]: Pearson correlation = 0.089 (p = 0.45868), Spearman correlation = 0.104 (p = 0.38406)

Bin (0.9, 1.0]: Pearson correlation = 0.004 (p = 0.88901), Spearman correlation = 0.025 (p = 0.41550)

### MULTIPLE METRICS LINEAR REGRESSION

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Pipelines total: 38

Pipelines by transfer_rate > 0.5: 32

Top pipelines by mean_cross_score:

                                mean_cross_score  transfer_rate  combined_score

source                                                                         

MVTec_AD_Wood_Scratch                   0.108823       0.918919        0.099999

MVTec_AD_Bottle_Broken_Sm               0.104860       0.837838        0.087856

MVTec_AD_Hazelnut_Crack                 0.084178       0.972973        0.081903

MVTec_AD_Metal_Nut                      0.079351       0.729730        0.057905

severstal-steel                         0.077334       0.783784        0.060613

MVTec_AD_Tile_Crack                     0.073861       0.540541        0.039925

MVTec_AD_Pill_Crack                     0.069799       0.837838        0.058481

AirCarbon3_80.jpg_bright                0.066951       0.891892        0.059713

MVTec_AD_Zipper_Rough                   0.063903       0.675676        0.043178

MAIPreform2_Spule0_0816_Upside          0.061231       0.675676        0.041372


---
Top pipelines by combined_score (mean * transfer_rate):
                           mean_cross_score  transfer_rate  combined_score

source                                                                    

MVTec_AD_Wood_Scratch              0.108823       0.918919        0.099999

MVTec_AD_Bottle_Broken_Sm          0.104860       0.837838        0.087856

MVTec_AD_Hazelnut_Crack            0.084178       0.972973        0.081903

severstal-steel                    0.077334       0.783784        0.060613

AirCarbon3_80.jpg_bright           0.066951       0.891892        0.059713

MVTec_AD_Pill_Crack                0.069799       0.837838        0.058481

MVTec_AD_Metal_Nut                 0.079351       0.729730        0.057905

AirCarbon3_80.jpg_dark_3           0.055090       0.891892        0.049134

AirCarbon3_80.jpg_dark_5           0.054834       0.837838        0.045942

Pultrusion_Window                  0.046199       0.972973        0.044950

### PIPELINE REUSE & MULTI-METRIC ANALYSIS

                       source  mean_cross_score  ...  count_targets  combined_score

32      MVTec_AD_Wood_Scratch          0.108823  ...             37        0.066902

24    MVTec_AD_Hazelnut_Crack          0.084178  ...             37        0.053941

19  MVTec_AD_Bottle_Broken_Sm          0.104860  ...             37        0.045581

1    AirCarbon3_80.jpg_bright          0.066951  ...             37        0.040677

37            severstal-steel          0.077334  ...             37        0.033825

[5 rows x 13 columns]

       metric  n_obs  r_squared  ...       p_value          aic          bic

0         cnn   1406   0.023356  ...  8.447183e-09 -2010.452246 -1999.955238

3     entropy   1406   0.012522  ...  2.605107e-05 -1994.941533 -1984.444525

2     texture   1406   0.006434  ...  2.614329e-03 -1986.298782 -1975.801773

5  superpixel   1406   0.005956  ...  3.783752e-03 -1985.623561 -1975.126553

1        edge   1406   0.004657  ...  1.048374e-02 -1983.786304 -1973.289296

4   frequency   1406   0.001382  ...  1.636182e-01 -1979.167764 -1968.670756

[6 rows x 8 columns]

                            OLS Regression Results                            
---
Dep. Variable:            cross_score   R-squared:                       0.040

Model:                            OLS   Adj. R-squared:                  0.035

Method:                 Least Squares   F-statistic:                     8.378

Date:                Thu, 23 Apr 2026   Prob (F-statistic):           4.64e-10

Time:                        17:31:36   Log-Likelihood:                 1019.5

No. Observations:                1406   AIC:                            -2023.

Df Residuals:                    1398   BIC:                            -1981.

Df Model:                           7                                         

Covariance Type:            nonrobust                                         

                     coef    std err          t      P>|t|      [0.025      0.975]

const              0.0428      0.003     13.655      0.000       0.037       0.049

cnn                0.0154      0.003      4.635      0.000       0.009       0.022

edge               0.0043      0.004      0.998      0.319      -0.004       0.013

texture            0.0021      0.004      0.513      0.608      -0.006       0.010

entropy            0.0132      0.004      3.562      0.000       0.006       0.021

frequency         -0.0062      0.004     -1.413      0.158      -0.015       0.002

superpixel         0.0077      0.004      1.891      0.059      -0.000       0.016

original_score    -0.0048      0.003     -1.452      0.147      -0.011       0.002


---
Omnibus:                      306.629   Durbin-Watson:                   1.755

Prob(Omnibus):                  0.000   Jarque-Bera (JB):            10642.823

Skew:                           0.161   Prob(JB):                         0.00

Kurtosis:                      16.475   Cond. No.                         2.95


Notes:

[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                           Logit Regression Results                           


---
Dep. Variable:          good_transfer   No. Observations:                 1406

Model:                          Logit   Df Residuals:                     1398

Method:                           MLE   Df Model:                            7

Date:                Thu, 23 Apr 2026   Pseudo R-squ.:                 0.07429

Time:                        17:31:38   Log-Likelihood:                -808.83

converged:                       True   LL-Null:                       -873.74

Covariance Type:            nonrobust   LLR p-value:                 6.855e-25


---

                     coef    std err          z      P>|z|      [0.025      0.975]

const             -0.8783      0.063    -14.005      0.000      -1.001      -0.755

cnn                0.2895      0.063      4.611      0.000       0.166       0.412

edge               0.2182      0.098      2.224      0.026       0.026       0.410

texture            0.1522      0.095      1.607      0.108      -0.033       0.338

entropy            0.3851      0.077      5.013      0.000       0.235       0.536

frequency         -0.1392      0.095     -1.472      0.141      -0.324       0.046

superpixel         0.2835      0.087      3.256      0.001       0.113       0.454

original_score    -0.1560      0.064     -2.424      0.015      -0.282      -0.030




## FOR BEST MCC

### CORRELATION ANALYSIS ---

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Overall:

✅ Pearson correlation: 0.12375453331489811  (p = 3.2530982029878423e-06 )

✅ Spearman correlation: 0.15404967578565698  (p = 6.402088894866122e-09 )


---
--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---

✅ Low Similarity (s < 0.5)- Pearson correlation: 0.09055568587208834  (p = 0.0016293732812479758 )

✅ Low Similarity (s < 0.5) - Spearman correlation: 0.10667339505827814  (p = 0.00020373779238714158 )

✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation: 0.0625155677328606  (p = 0.4975690405434236 )

✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation: 0.04544665284659821  (p = 0.6220924023833645 )

✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation: 0.15255525999015118  (p = 0.18240108521271564 )

✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation: 0.19628705818595735  (p = 0.0850024428951917 )


---
--- Analysis for Bin Size: 0.2 ---

Bin (-0.001, 0.2]: Pearson correlation = 0.155 (p = 0.00148), Spearman correlation = 0.154 (p = 0.00159)

Bin (0.2, 0.4]: Pearson correlation = 0.016 (p = 0.68145), Spearman correlation = 0.038 (p = 0.31875)

Bin (0.4, 0.6]: Pearson correlation = -0.026 (p = 0.71727), Spearman correlation = -0.003 (p = 0.96237)

Bin (0.6, 0.8]: Pearson correlation = 0.093 (p = 0.52810), Spearman correlation = 0.201 (p = 0.16988)

Bin (0.8, 1.0]: Pearson correlation = 0.488 (p = 0.00004), Spearman correlation = 0.438 (p = 0.00029)


---
--- Analysis for Bin Size: 0.1 ---

Bin (-0.001, 0.1]: Pearson correlation = 0.499 (p = 0.00684), Spearman correlation = 0.057 (p = 0.77503)

Bin (0.1, 0.2]: Pearson correlation = 0.062 (p = 0.21948), Spearman correlation = 0.060 (p = 0.24064)

Bin (0.2, 0.3]: Pearson correlation = 0.094 (p = 0.04243), Spearman correlation = 0.105 (p = 0.02326)

Bin (0.3, 0.4]: Pearson correlation = 0.043 (p = 0.53822), Spearman correlation = -0.009 (p = 0.90022)

Bin (0.4, 0.5]: Pearson correlation = 0.013 (p = 0.89384), Spearman correlation = 0.058 (p = 0.54035)

Bin (0.5, 0.6]: Pearson correlation = -0.046 (p = 0.67771), Spearman correlation = -0.073 (p = 0.51097)

Bin (0.6, 0.7]: Pearson correlation = -0.205 (p = 0.23131), Spearman correlation = -0.066 (p = 0.70226)

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:420: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for bin_range, group in correlation_df.groupby('bin'):

Bin (0.7, 0.8]: Pearson correlation = -0.093 (p = 0.77281), Spearman correlation = -0.297 (p = 0.34880)

Bin (0.8, 0.9]: Pearson correlation = 0.075 (p = 0.68426), Spearman correlation = 0.228 (p = 0.20992)

Bin (0.9, 1.0]: Pearson correlation = 0.373 (p = 0.03569), Spearman correlation = 0.340 (p = 0.05686)

### MULTIPLE METRICS LINEAR REGRESSION ---

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Pipelines total: 38

Pipelines by transfer_rate > 0.5: 33


---
Top pipelines by mean_cross_score:

                             mean_cross_score  transfer_rate  combined_score

source                                                                      

AirCarbon3_80.jpg_dark_2             0.133486       0.864865        0.115448

MVTec_AD_Wood_Scratch                0.125064       0.864865        0.108163

MVTec_AD_Zipper_Rough                0.120984       0.891892        0.107905

MVTec_AD_Capsule                     0.113963       0.972973        0.110883

MVTec_AD_Toothbrush_Sm               0.113285       0.972973        0.110223

severstal-steel                      0.109127       0.864865        0.094380

AirCarbon3_80.jpg_dark_3             0.105951       0.891892        0.094497

CF_ReferenceSet_Small_Light          0.101112       0.891892        0.090181

MVTec_AD_Tile_Crack                  0.098125       0.729730        0.071605

MVTec_AD_Screw_Scratch               0.094959       0.891892        0.084693


---
Top pipelines by combined_score (mean * transfer_rate):

                             mean_cross_score  transfer_rate  combined_score

source                                                                      

AirCarbon3_80.jpg_dark_2             0.133486       0.864865        0.115448

MVTec_AD_Capsule                     0.113963       0.972973        0.110883

MVTec_AD_Toothbrush_Sm               0.113285       0.972973        0.110223

MVTec_AD_Wood_Scratch                0.125064       0.864865        0.108163

MVTec_AD_Zipper_Rough                0.120984       0.891892        0.107905

AirCarbon3_80.jpg_dark_3             0.105951       0.891892        0.094497

severstal-steel                      0.109127       0.864865        0.094380

CF_ReferenceSet_Small_Light          0.101112       0.891892        0.090181

MVTec_AD_Screw_Scratch               0.094959       0.891892        0.084693

MVTec_AD_Metal_Nut                   0.087780       0.945946        0.083035

### CORRELATION ANALYSIS ---

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Overall:

✅ Pearson correlation: 0.11600498018368753  (p = 1.296703114201736e-05 )

✅ Spearman correlation: 0.16659535990566782  (p = 3.274625281712733e-10 )


---
--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---

[SKIP] Not enough data for Pearson (low similarity): n=0

✅ Low Similarity (s < 0.5)- Pearson correlation: nan  (p = nan )

✅ Low Similarity (s < 0.5) - Spearman correlation: nan  (p = nan )

✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation: 0.061027612176370996  (p = 0.6803020775924037 )

✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation: 0.18502831848078954  (p = 0.20802047764605136 )

✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation: 0.12217685453807231  (p = 6.327971582396604e-06 )

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:420: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for bin_range, group in correlation_df.groupby('bin'):

✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation: 0.16580032324871422  (p = 7.902170624713401e-10 )


---
--- Analysis for Bin Size: 0.2 ---

Bin (0.4, 0.6]: Pearson correlation = 0.585 (p = 0.22224), Spearman correlation = 0.956 (p = 0.00284)

Bin (0.6, 0.8]: Pearson correlation = -0.108 (p = 0.23043), Spearman correlation = -0.100 (p = 0.26828)

Bin (0.8, 1.0]: Pearson correlation = 0.108 (p = 0.00011), Spearman correlation = 0.150 (p = 0.00000)


---
--- Analysis for Bin Size: 0.1 ---

Bin (0.5, 0.6]: Pearson correlation = 0.585 (p = 0.22224), Spearman correlation = 0.956 (p = 0.00284)

Bin (0.6, 0.7]: Pearson correlation = -0.139 (p = 0.37906), Spearman correlation = 0.115 (p = 0.46818)

Bin (0.7, 0.8]: Pearson correlation = -0.096 (p = 0.39178), Spearman correlation = -0.078 (p = 0.48461)

Bin (0.8, 0.9]: Pearson correlation = 0.017 (p = 0.82662), Spearman correlation = 0.002 (p = 0.97853)

Bin (0.9, 1.0]: Pearson correlation = 0.082 (p = 0.00629), Spearman correlation = 0.114 (p = 0.00014)

### MULTIPLE METRICS LINEAR REGRESSION ---

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Pipelines total: 38

Pipelines by transfer_rate > 0.5: 33

Top pipelines by mean_cross_score:

                             mean_cross_score  transfer_rate  combined_score

source                                                                      

AirCarbon3_80.jpg_dark_2             0.133486       0.864865        0.115448

MVTec_AD_Wood_Scratch                0.125064       0.864865        0.108163

MVTec_AD_Zipper_Rough                0.120984       0.891892        0.107905

MVTec_AD_Capsule                     0.113963       0.972973        0.110883

MVTec_AD_Toothbrush_Sm               0.113285       0.972973        0.110223

severstal-steel                      0.109127       0.864865        0.094380

AirCarbon3_80.jpg_dark_3             0.105951       0.891892        0.094497

CF_ReferenceSet_Small_Light          0.101112       0.891892        0.090181

MVTec_AD_Tile_Crack                  0.098125       0.729730        0.071605

MVTec_AD_Screw_Scratch               0.094959       0.891892        0.084693


---
Top pipelines by combined_score (mean * transfer_rate):

                             mean_cross_score  transfer_rate  combined_score

source                                                                      

AirCarbon3_80.jpg_dark_2             0.133486       0.864865        0.115448

MVTec_AD_Capsule                     0.113963       0.972973        0.110883

MVTec_AD_Toothbrush_Sm               0.113285       0.972973        0.110223

MVTec_AD_Wood_Scratch                0.125064       0.864865        0.108163

MVTec_AD_Zipper_Rough                0.120984       0.891892        0.107905

AirCarbon3_80.jpg_dark_3             0.105951       0.891892        0.094497

severstal-steel                      0.109127       0.864865        0.094380

CF_ReferenceSet_Small_Light          0.101112       0.891892        0.090181

MVTec_AD_Screw_Scratch               0.094959       0.891892        0.084693

MVTec_AD_Metal_Nut                   0.087780       0.945946        0.083035

### CORRELATION ANALYSIS ---

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'


Overall:

✅ Pearson correlation: 0.13721373256357927  (p = 2.405306832511116e-07 )

✅ Spearman correlation: 0.12530670162934826  (p = 2.4410607188984748e-06 )


---
--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---

[SKIP] Not enough data for Pearson (low similarity): n=0

✅ Low Similarity (s < 0.5)- Pearson correlation: nan  (p = nan )

✅ Low Similarity (s < 0.5) - Spearman correlation: nan  (p = nan )

✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation: 0.19360551699377593  (p = 0.18733764754224477 )

✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation: 0.16758140253899706  (p = 0.2549088305509159 )

✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation: 0.07097047227851072  (p = 0.008890727657157342 )

✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation: 0.08490826700713483  (p = 0.0017377703990467367 )


---
--- Analysis for Bin Size: 0.2 ---

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:420: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for bin_range, group in correlation_df.groupby('bin'):

Bin (0.4, 0.6]: Pearson correlation = -0.058 (p = 0.79755), Spearman correlation = -0.188 (p = 0.40146)

Bin (0.6, 0.8]: Pearson correlation = 0.251 (p = 0.10896), Spearman correlation = 0.238 (p = 0.12850)

Bin (0.8, 1.0]: Pearson correlation = 0.051 (p = 0.06257), Spearman correlation = 0.074 (p = 0.00695)

---
--- Analysis for Bin Size: 0.1 ---

Bin (0.5, 0.6]: Pearson correlation = -0.058 (p = 0.79755), Spearman correlation = -0.188 (p = 0.40146)

Bin (0.6, 0.7]: Pearson correlation = 0.498 (p = 0.00964), Spearman correlation = 0.469 (p = 0.01573)

Bin (0.7, 0.8]: Pearson correlation = 0.304 (p = 0.25155), Spearman correlation = 0.314 (p = 0.23687)

Bin (0.8, 0.9]: Pearson correlation = -0.078 (p = 0.28393), Spearman correlation = -0.038 (p = 0.60439)

Bin (0.9, 1.0]: Pearson correlation = 0.015 (p = 0.61292), Spearman correlation = 0.052 (p = 0.07775)

### MULTIPLE METRICS LINEAR REGRESSION ---

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Pipelines total: 38

Pipelines by transfer_rate > 0.5: 33

---
Top pipelines by mean_cross_score:

                             mean_cross_score  transfer_rate  combined_score

source                                                                      

AirCarbon3_80.jpg_dark_2             0.133486       0.864865        0.115448

MVTec_AD_Wood_Scratch                0.125064       0.864865        0.108163

MVTec_AD_Zipper_Rough                0.120984       0.891892        0.107905

MVTec_AD_Capsule                     0.113963       0.972973        0.110883

MVTec_AD_Toothbrush_Sm               0.113285       0.972973        0.110223

severstal-steel                      0.109127       0.864865        0.094380

AirCarbon3_80.jpg_dark_3             0.105951       0.891892        0.094497

CF_ReferenceSet_Small_Light          0.101112       0.891892        0.090181

MVTec_AD_Tile_Crack                  0.098125       0.729730        0.071605

MVTec_AD_Screw_Scratch               0.094959       0.891892        0.084693

---
Top pipelines by combined_score (mean * transfer_rate):

                             mean_cross_score  transfer_rate  combined_score

source                                                                      

AirCarbon3_80.jpg_dark_2             0.133486       0.864865        0.115448

MVTec_AD_Capsule                     0.113963       0.972973        0.110883

MVTec_AD_Toothbrush_Sm               0.113285       0.972973        0.110223

MVTec_AD_Wood_Scratch                0.125064       0.864865        0.108163

MVTec_AD_Zipper_Rough                0.120984       0.891892        0.107905

AirCarbon3_80.jpg_dark_3             0.105951       0.891892        0.094497

severstal-steel                      0.109127       0.864865        0.094380

CF_ReferenceSet_Small_Light          0.101112       0.891892        0.090181

MVTec_AD_Screw_Scratch               0.094959       0.891892        0.084693

MVTec_AD_Metal_Nut                   0.087780       0.945946        0.083035

### CORRELATION ANALYSIS

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:420: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for bin_range, group in correlation_df.groupby('bin'):

Overall:

✅ Pearson correlation: 0.10671033333486346  (p = 6.0931241428009064e-05 )

✅ Spearman correlation: 0.1313063997657262  (p = 7.789125799660832e-07 )

---
--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---

[SKIP] Not enough data for Pearson (low similarity): n=0

✅ Low Similarity (s < 0.5)- Pearson correlation: nan  (p = nan )

✅ Low Similarity (s < 0.5) - Spearman correlation: nan  (p = nan )

✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation: 0.27747548781831605  (p = 0.1528404510987612 )

✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation: 0.13417656475514833  (p = 0.49605274220212503 )

✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation: 0.08821498707421302  (p = 0.001045133959117886 )

✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation: 0.11089259659167079  (p = 3.699547003560556e-05 )


---
--- Analysis for Bin Size: 0.2 ---

Bin (0.6, 0.8]: Pearson correlation = 0.110 (p = 0.09677), Spearman correlation = 0.172 (p = 0.00924)

Bin (0.8, 1.0]: Pearson correlation = 0.065 (p = 0.02608), Spearman correlation = 0.096 (p = 0.00096)

---
--- Analysis for Bin Size: 0.1 ---

Bin (0.6, 0.7]: Pearson correlation = 0.277 (p = 0.15284), Spearman correlation = 0.134 (p = 0.49605)

Bin (0.7, 0.8]: Pearson correlation = 0.032 (p = 0.65634), Spearman correlation = 0.062 (p = 0.38647)

Bin (0.8, 0.9]: Pearson correlation = 0.046 (p = 0.37485), Spearman correlation = 0.094 (p = 0.07288)

Bin (0.9, 1.0]: Pearson correlation = 0.030 (p = 0.39826), Spearman correlation = 0.025 (p = 0.47107)

## MULTIPLE METRICS LINEAR REGRESSION

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Pipelines total: 38
Pipelines by transfer_rate > 0.5: 33

---
Top pipelines by mean_cross_score:

                             mean_cross_score  transfer_rate  combined_score

source                                                                      

AirCarbon3_80.jpg_dark_2             0.133486       0.864865        0.115448

MVTec_AD_Wood_Scratch                0.125064       0.864865        0.108163

MVTec_AD_Zipper_Rough                0.120984       0.891892        0.107905

MVTec_AD_Capsule                     0.113963       0.972973        0.110883

MVTec_AD_Toothbrush_Sm               0.113285       0.972973        0.110223

severstal-steel                      0.109127       0.864865        0.094380

AirCarbon3_80.jpg_dark_3             0.105951       0.891892        0.094497

CF_ReferenceSet_Small_Light          0.101112       0.891892        0.090181

MVTec_AD_Tile_Crack                  0.098125       0.729730        0.071605

MVTec_AD_Screw_Scratch               0.094959       0.891892        0.084693

---
Top pipelines by combined_score (mean * transfer_rate):

                             mean_cross_score  transfer_rate  combined_score

source                                                                      

AirCarbon3_80.jpg_dark_2             0.133486       0.864865        0.115448

MVTec_AD_Capsule                     0.113963       0.972973        0.110883

MVTec_AD_Toothbrush_Sm               0.113285       0.972973        0.110223

MVTec_AD_Wood_Scratch                0.125064       0.864865        0.108163

MVTec_AD_Zipper_Rough                0.120984       0.891892        0.107905

AirCarbon3_80.jpg_dark_3             0.105951       0.891892        0.094497

severstal-steel                      0.109127       0.864865        0.094380

CF_ReferenceSet_Small_Light          0.101112       0.891892        0.090181

MVTec_AD_Screw_Scratch               0.094959       0.891892        0.084693

MVTec_AD_Metal_Nut                   0.087780       0.945946        0.083035

## CORRELATION ANALYSIS

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Overall:
✅ Pearson correlation: 0.1018511731155772  (p = 0.000130435440717564 )

✅ Spearman correlation: 0.14074311324799582  (p = 1.163745112312868e-07 )

---
--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---

[SKIP] Not enough data for Pearson (low similarity): n=0

✅ Low Similarity (s < 0.5)- Pearson correlation: nan  (p = nan )

✅ Low Similarity (s < 0.5) - Spearman correlation: nan  (p = nan )

✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation: nan  (p = nan )

✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation: nan  (p = nan )

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:355: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return pearsonr(df["x"], df["y"])

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:398: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  medium_spearman_corr, medium_spearman_p = spearmanr(medium_similarity["similarity"],

✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation: 0.10073583948638434  (p = 0.00015630665066903733 )

✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation: 0.13958338603587353  (p = 1.510655156101587e-07 )

---
--- Analysis for Bin Size: 0.2 ---

Bin (0.6, 0.8]: Pearson correlation = -0.100 (p = 0.50664), Spearman correlation = -0.155 (p = 0.30526)

Bin (0.8, 1.0]: Pearson correlation = 0.103 (p = 0.00014), Spearman correlation = 0.131 (p = 0.00000)

---
--- Analysis for Bin Size: 0.1 ---

Bin (0.6, 0.7]: Pearson correlation = nan (p = nan), Spearman correlation = nan (p = nan)

Bin (0.7, 0.8]: Pearson correlation = -0.119 (p = 0.44340), Spearman correlation = -0.174 (p = 0.25854)

Bin (0.8, 0.9]: Pearson correlation = -0.036 (p = 0.69354), Spearman correlation = -0.087 (p = 0.34280)

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:420: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for bin_range, group in correlation_df.groupby('bin'):

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:422: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  bin_pearson_corr, bin_pearson_p = pearsonr(group["similarity"], group["cross_score"])

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:423: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  bin_spearman_corr, bin_spearman_p = spearmanr(group["similarity"], group["cross_score"])

Bin (0.9, 1.0]: Pearson correlation = 0.058 (p = 0.03962), Spearman correlation = 0.093 (p = 0.00098)

## MULTIPLE METRICS LINEAR REGRESSION

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Pipelines total: 38
Pipelines by transfer_rate > 0.5: 33

---
Top pipelines by mean_cross_score:

                             mean_cross_score  transfer_rate  combined_score

source                                                                      

AirCarbon3_80.jpg_dark_2             0.133486       0.864865        0.115448

MVTec_AD_Wood_Scratch                0.125064       0.864865        0.108163

MVTec_AD_Zipper_Rough                0.120984       0.891892        0.107905

MVTec_AD_Capsule                     0.113963       0.972973        0.110883

MVTec_AD_Toothbrush_Sm               0.113285       0.972973        0.110223

severstal-steel                      0.109127       0.864865        0.094380

AirCarbon3_80.jpg_dark_3             0.105951       0.891892        0.094497

CF_ReferenceSet_Small_Light          0.101112       0.891892        0.090181

MVTec_AD_Tile_Crack                  0.098125       0.729730        0.071605

MVTec_AD_Screw_Scratch               0.094959       0.891892        0.084693

---
Top pipelines by combined_score (mean * transfer_rate):

                             mean_cross_score  transfer_rate  combined_score

source                                                                      

AirCarbon3_80.jpg_dark_2             0.133486       0.864865        0.115448

MVTec_AD_Capsule                     0.113963       0.972973        0.110883

MVTec_AD_Toothbrush_Sm               0.113285       0.972973        0.110223

MVTec_AD_Wood_Scratch                0.125064       0.864865        0.108163

MVTec_AD_Zipper_Rough                0.120984       0.891892        0.107905

AirCarbon3_80.jpg_dark_3             0.105951       0.891892        0.094497

severstal-steel                      0.109127       0.864865        0.094380

CF_ReferenceSet_Small_Light          0.101112       0.891892        0.090181

MVTec_AD_Screw_Scratch               0.094959       0.891892        0.084693

MVTec_AD_Metal_Nut                   0.087780       0.945946        0.083035

## CORRELATION ANALYSIS

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

D:\dev\EstimatingInspectionTasks\src\statistical_analysis.py:420: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  for bin_range, group in correlation_df.groupby('bin'):

Overall:

✅ Pearson correlation: 0.14250841609659712  (p = 8.03913836193237e-08 )

✅ Spearman correlation: 0.08955089994356782  (p = 0.0007748401939893116 )


---
--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---

[SKIP] Not enough data for Pearson (low similarity): n=0

✅ Low Similarity (s < 0.5)- Pearson correlation: nan  (p = nan )

✅ Low Similarity (s < 0.5) - Spearman correlation: nan  (p = nan )

✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation: 0.16443514456758146  (p = 0.029198134178015853 )

✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation: 0.15605053852765122  (p = 0.03862307767723171 )

✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation: 0.06553553439955746  (p = 0.021530252802969312 )

✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation: 0.027686625602380884  (p = 0.3319437097997956 )

---
--- Analysis for Bin Size: 0.2 ---

Bin (0.4, 0.6]: Pearson correlation = 0.286 (p = 0.05940), Spearman correlation = 0.264 (p = 0.08292)

Bin (0.6, 0.8]: Pearson correlation = 0.125 (p = 0.04483), Spearman correlation = 0.077 (p = 0.21800)

Bin (0.8, 1.0]: Pearson correlation = 0.009 (p = 0.76922), Spearman correlation = -0.005 (p = 0.87820)

---
--- Analysis for Bin Size: 0.1 ---

Bin (0.5, 0.6]: Pearson correlation = 0.286 (p = 0.05940), Spearman correlation = 0.264 (p = 0.08292)

Bin (0.6, 0.7]: Pearson correlation = 0.037 (p = 0.67703), Spearman correlation = -0.001 (p = 0.98811)

Bin (0.7, 0.8]: Pearson correlation = 0.127 (p = 0.15432), Spearman correlation = 0.129 (p = 0.14746)

Bin (0.8, 0.9]: Pearson correlation = 0.136 (p = 0.25560), Spearman correlation = 0.042 (p = 0.72801)


Bin (0.9, 1.0]: Pearson correlation = -0.031 (p = 0.32356), Spearman correlation = -0.014 (p = 0.64422)

## MULTIPLE METRICS LINEAR REGRESSION

[UNKNOWN INDEX LABEL] '80.jpg_bright'

[UNKNOWN COLUMN LABEL] '80.jpg_bright'

Pipelines total: 38

### Pipelines by transfer_rate > 0.5: 33

Top pipelines by mean_cross_score:

                             mean_cross_score  transfer_rate  combined_score
source                                                                      

AirCarbon3_80.jpg_dark_2             0.133486       0.864865        0.115448

MVTec_AD_Wood_Scratch                0.125064       0.864865        0.108163

MVTec_AD_Zipper_Rough                0.120984       0.891892        0.107905

MVTec_AD_Capsule                     0.113963       0.972973        0.110883

MVTec_AD_Toothbrush_Sm               0.113285       0.972973        0.110223

severstal-steel                      0.109127       0.864865        0.094380

AirCarbon3_80.jpg_dark_3             0.105951       0.891892        0.094497

CF_ReferenceSet_Small_Light          0.101112       0.891892        0.090181

MVTec_AD_Tile_Crack                  0.098125       0.729730        0.071605

MVTec_AD_Screw_Scratch               0.094959       0.891892        0.084693

---
#### mean * transfer_rate

Top pipelines by combined_score (mean * transfer_rate):

                             mean_cross_score  transfer_rate  combined_score

source                                                                      

AirCarbon3_80.jpg_dark_2             0.133486       0.864865        0.115448

MVTec_AD_Capsule                     0.113963       0.972973        0.110883

MVTec_AD_Toothbrush_Sm               0.113285       0.972973        0.110223

MVTec_AD_Wood_Scratch                0.125064       0.864865        0.108163

MVTec_AD_Zipper_Rough                0.120984       0.891892        0.107905

AirCarbon3_80.jpg_dark_3             0.105951       0.891892        0.094497

severstal-steel                      0.109127       0.864865        0.094380

CF_ReferenceSet_Small_Light          0.101112       0.891892        0.090181

MVTec_AD_Screw_Scratch               0.094959       0.891892        0.084693

MVTec_AD_Metal_Nut                   0.087780       0.945946        0.083035

## PIPELINE REUSE & MULTI-METRIC ANALYSIS

                      source  mean_cross_score  ...  count_targets  combined_score

3   AirCarbon3_80.jpg_dark_2          0.133486  ...             37        0.075685

30    MVTec_AD_Toothbrush_Sm          0.113285  ...             37        0.074763

32     MVTec_AD_Wood_Scratch          0.125064  ...             37        0.071737

21          MVTec_AD_Capsule          0.113963  ...             37        0.071241

33     MVTec_AD_Zipper_Rough          0.120984  ...             37        0.068540

[5 rows x 13 columns]

       metric  n_obs  r_squared  ...       p_value          aic          bic

5  superpixel   1406   0.020309  ...  8.039138e-08 -1735.164764 -1724.667756

2     texture   1406   0.018828  ...  2.405307e-07 -1733.040860 -1722.543852

0         cnn   1406   0.015315  ...  3.253098e-06 -1728.016616 -1717.519608

1        edge   1406   0.013457  ...  1.296703e-05 -1725.366096 -1714.869087

3     entropy   1406   0.011387  ...  6.093124e-05 -1722.418980 -1711.921972

4   frequency   1406   0.010374  ...  1.304354e-04 -1720.978419 -1710.481410

[6 rows x 8 columns]
                            OLS Regression Results                            
---
Dep. Variable:            cross_score   R-squared:                       0.050
Model:                            OLS   Adj. R-squared:                  0.045
Method:                 Least Squares   F-statistic:                     10.40
Date:                Thu, 23 Apr 2026   Prob (F-statistic):           8.82e-13
Time:                        17:32:57   Log-Likelihood:                 890.85
No. Observations:                1406   AIC:                            -1766.
Df Residuals:                    1398   BIC:                            -1724.
Df Model:                           7                                         
Covariance Type:            nonrobust                                         

---
                     coef    std err          t      P>|t|      [0.025      0.975]
---
const              0.0557      0.003     16.206      0.000       0.049       0.062
cnn                0.0128      0.004      3.528      0.000       0.006       0.020
edge               0.0026      0.005      0.550      0.583      -0.007       0.012
texture            0.0037      0.004      0.828      0.408      -0.005       0.012
entropy            0.0100      0.004      2.452      0.014       0.002       0.018
frequency          0.0044      0.005      0.914      0.361      -0.005       0.014
superpixel         0.0159      0.004      3.536      0.000       0.007       0.025
original_score    -0.0087      0.004     -2.379      0.017      -0.016      -0.002
---
Omnibus:                      367.554   Durbin-Watson:                   1.690
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2145.551
Skew:                           1.084   Prob(JB):                         0.00
Kurtosis:                       8.650   Cond. No.                         2.94
---

Notes:

[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                           Logit Regression Results                           
---
Dep. Variable:          good_transfer   

No. Observations:                 1406

Model:                          Logit   

Df Residuals:                     1398

Method:                           MLE   

Df Model:                            7

Date:                Thu, 23 Apr 2026   

Pseudo R-squ.:                 0.05150

Time:                        17:32:58   

Log-Likelihood:                -877.14

converged:                       True   

LL-Null:                       -924.76

Covariance Type:            nonrobust   LLR p-value:                 1.029e-17
---

                     coef    std err          z      P>|z|      [0.025      0.975]
---
const             -0.5940      0.058    -10.164      0.000      -0.709      -0.479

cnn                0.1777      0.060      2.954      0.003       0.060       0.296

edge               0.1553      0.088      1.772      0.076      -0.016       0.327

texture            0.1689      0.089      1.887      0.059      -0.006       0.344

entropy            0.2186      0.069      3.152      0.002       0.083       0.355

frequency          0.0195      0.087      0.224      0.823      -0.151       0.190

superpixel         0.2554      0.081      3.167      0.002       0.097       0.414

`original_score`    -0.1199      0.061     -1.968      0.049      -0.239      -0.001

---

### Correlation Analysis for `20260416-181811`

#### Distribution for:  D:\dev\EstimatingInspectionTasks\results\similarity\20260416-181811\20260115-181800_resnet.csv

| Similarity | Range             | Count | Percentage |
|------------|-------------------|-------|------------|
| None       | > -1.0 and <= 0.3 | 442   | 62.87%     |
| Low        | > 0.3 and <= 0.5  | 162   | 23.04%     |
| Medium     | > 0.5 and <= 0.7  | 60    | 8.53%      |
| High       | > 0.7 and <= 1.0  | 38    | 5.41%      |
| Other      | (unclassified)    | 1     | 0.14%      |
shape after drop: (38, 38)
common count: 38
shape after align: (38, 38)
#### Distribution for:  D:\dev\EstimatingInspectionTasks\results\similarity\20260416-181811\20260416-181811_histEnt.csv
| Similarity | Range             | Count | Percentage |
|------------|-------------------|-------|------------|
| None       | > -1.0 and <= 0.3 | 0     | 0.00%      |
| Low        | > 0.3 and <= 0.5  | 0     | 0.00%      |
| Medium     | > 0.5 and <= 0.7  | 14    | 1.99%      |
| High       | > 0.7 and <= 1.0  | 688   | 97.87%     |
| Other      | (unclassified)    | 1     | 0.14%      |
shape after drop: (38, 38)
common count: 38
shape after align: (38, 38)

#### Distribution for:  D:\dev\EstimatingInspectionTasks\results\similarity\20260416-181811\20260416-181819_textComp.csv
| Similarity | Range             | Count | Percentage |
|------------|-------------------|-------|------------|
| None       | > -1.0 and <= 0.3 | 0     | 0.00%      |
| Low        | > 0.3 and <= 0.5  | 0     | 0.00%      |
| Medium     | > 0.5 and <= 0.7  | 24    | 3.41%      |
| High       | > 0.7 and <= 1.0  | 679   | 96.59%     |
shape after drop: (38, 38)
common count: 38
shape after align: (38, 38)

#### Distribution for:  D:\dev\EstimatingInspectionTasks\results\similarity\20260416-181811\20260416-181835_edgeDen.csv
| Similarity | Range             | Count | Percentage |
|------------|-------------------|-------|------------|
| None       | > -1.0 and <= 0.3 | 0     | 0.00%      |
| Low        | > 0.3 and <= 0.5  | 0     | 0.00%      |
| Medium     | > 0.5 and <= 0.7  | 24    | 3.41%      |
| High       | > 0.7 and <= 1.0  | 678   | 96.44%     |
| Other      | (unclassified)    | 1     | 0.14%      |
shape after drop: (38, 38)
common count: 38
shape after align: (38, 38)

#### Distribution for:  D:\dev\EstimatingInspectionTasks\results\similarity\20260416-181811\20260416-181917_noOfSup.csv
| Similarity | Range             | Count | Percentage |
|------------|-------------------|-------|------------|
| None       | > -1.0 and <= 0.3 | 0     | 0.00%      |
| Low        | > 0.3 and <= 0.5  | 0     | 0.00%      |
| Medium     | > 0.5 and <= 0.7  | 88    | 12.52%     |
| High       | > 0.7 and <= 1.0  | 615   | 87.48%     |
shape after drop: (38, 38)
common count: 38
shape after align: (38, 38)

#### Distribution for:  D:\dev\EstimatingInspectionTasks\results\similarity\20260416-181811\20260416-181934_fourFreq.csv
| Similarity | Range             | Count | Percentage |
|------------|-------------------|-------|------------|
| None       | > -1.0 and <= 0.3 | 0     | 0.00%      |
| Low        | > 0.3 and <= 0.5  | 0     | 0.00%      |
| Medium     | > 0.5 and <= 0.7  | 1     | 0.14%      |
| High       | > 0.7 and <= 1.0  | 702   | 99.86%     |
shape after drop: (38, 38)
common count: 38
shape after align: (38, 38)

# Statistical Figures

## Similarity Heatmaps

<img src="results/similarity/20260416-181811/20260115-181800_resnet_heatmap.png" alt="Similarity Heatmap Resnet" width="300">

<img src="results/similarity/20260416-181811/20260416-181811_histEnt_heatmap.png" alt="Similarity Heatmap histEnt" width="300">

<img src="results/similarity/20260416-181811/20260416-181819_textComp_heatmap.png" alt="Similarity Heatmap textComp" width="300">

<img src="results/similarity/20260416-181811/20260416-181835_edgeDen_heatmap.png" alt="Similarity Heatmap edgeDen" width="300">

<img src="results/similarity/20260416-181811/20260416-181917_noOfSup_heatmap.png" alt="Similarity Heatmap noOfSup" width="300">

<img src="results/similarity/20260416-181811/20260416-181934_fourFreq_heatmap.png" alt="Similarity Heatmap fourFreq" width="300">

## Similarity Scatterplots


<img src="results/similarity/20260416-181811/20260115-181800_resnet_best_scatter.png" alt="Resnet Scatterplot" width="300">

<img src="results/similarity/20260416-181811/20260416-181811_histEnt_best_scatter.png" alt="histEnt Scatterplot" width="300">

<img src="results/similarity/20260416-181811/20260416-181819_textComp_best_scatter.png" alt="textComp Scatterplot" width="300">

<img src="results/similarity/20260416-181811/20260416-181835_edgeDen_best_scatter.png" alt="edgeDen Scatterplot" width="300">

<img src="results/similarity/20260416-181811/20260416-181917_noOfSup_best_scatter.png" alt="noOfSup Scatterplot" width="300">

<img src="results/similarity/20260416-181811/20260416-181934_fourFreq_best_scatter.png" alt="fourFreq Scatterplot" width="300">
