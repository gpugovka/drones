# Kinematics-Based UAV Telemetry Fault Detection Using Tree Ensembles

A multi-class machine learning pipeline for detecting and classifying Unmanned Aerial Vehicle (UAV) hardware and control faults from multi-sensor telemetry logs using physics-informed feature engineering and decision tree ensembles.

## Project Overview

Unmanned Aerial Vehicles rely on synchronized sensor fusion (IMU, magnetometer, barometer, and GPS) to maintain stable flight. When a hardware or control subsystem fails, raw single-axis telemetry readings often exhibit sequential time-block artifacts in simulation logs, risking temporal memorization rather than true physical fault learning.

To address this, this project:
1. **Prevents Temporal Leakage:** Removes log timestamps and raw single-axis readings that encourage time-window memorization.
2. **Engineers Rotation-Invariant Kinematics Features:** Combines 3-axis sensor vectors into physical magnitudes and computes modular-wrapped attitude tracking errors.
3. **Evaluates Non-Linear Decision Boundaries:** Benchmarks a cross-validated linear baseline (`LogisticRegressionCV`) against non-linear tree ensembles (`RandomForestClassifier` and `XGBClassifier`).

---

## Dataset & Fault Classes

This project uses `Fusion_Data.csv` from the **TLMUAV Anomaly Detection Datasets** (`12,253` telemetry records across `18` raw sensor features), generated from a software-in-the-loop UAV simulation environment.

* **Kaggle Dataset:** [TLMUAV Anomaly Detection Datasets](https://www.kaggle.com/datasets/luyucwnu/tlmuav-anomaly-detection-datasets)

### Target Classes (`labels`)
| Label | Flight / Fault State |
| :---: | :--- |
| `0` | Normal Flight |
| `1` | GPS Fault |
| `2` | Accelerometer Fault |
| `3` | Engine Fault |
| `4` | Remote-Control (RC) System Fault |

---

## Physics-Based Feature Engineering

Rather than training on raw single-axis sensor readings, seven rotation-invariant physical features and control error indicators are extracted:

* **3D Acceleration Magnitude:** Total specific force across all three accelerometer axes:
  $$a_{\text{total}} = \sqrt{\text{abAccX}^2 + \text{abAccY}^2 + \text{abAccZ}^2}$$
* **3D Angular Velocity Magnitude:** Total rotational rate across all three gyroscope axes:
  $$\omega_{\text{total}} = \sqrt{\text{abGyrX}^2 + \text{abGyrY}^2 + \text{abGyrZ}^2}$$
* **Total Magnetic Field Strength:** Overall ambient magnetic magnitude to detect compass/navigation anomalies:
  $$B_{\text{total}} = \sqrt{\text{MagX}^2 + \text{MagY}^2 + \text{MagZ}^2}$$
* **Attitude Tracking Errors:** Difference between desired (`Des*`) and actual measured angles, with modular wrapping on Yaw ($[0^\circ, 360^\circ] \to [-180^\circ, 180^\circ]$) to prevent false $360^\circ$ boundary spikes:

  $$\text{Error}_{\text{Roll}} = \text{DesRoll} - \text{Roll}$$

  $$\text{Error}_{\text{Pitch}} = \text{DesRPitch} - \text{Pitch}$$
  
  $$\text{Error}_{\text{Yaw}} = ((\text{DesYaw} - \text{Yaw} + 180) \pmod{360}) - 180$$
* **Combined Roll-Pitch Error (`ErrRP`):** Flight controller attitude tracking error indicator.

---

## Model Performance

Evaluated on an 80/20 stratified/shuffled train-test split (`2,451` unseen test samples) with `StandardScaler` fit strictly on the training set:

| Model | Test Accuracy | Macro F1-Score | Test Errors (out of 2,451) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** (`LogisticRegressionCV`) | 71.73% | 0.70 | 693 |
| **Random Forest** (`RandomForestClassifier`) | 98.94% | 0.98 | 26 |
| **XGBoost** (`XGBClassifier`) | **99.35%** | **0.99** | **16** |

**Key Takeaway:** Tree ensembles dramatically outperform the linear baseline (99.35% vs. 71.73%), confirming that UAV hardware and control anomalies manifest across non-linear kinematic boundaries.

---

## Repository Structure

```text
├── Drone_HW_Anomaly_Detection.ipynb   # End-to-end EDA, feature engineering, training & evaluation notebook
├── attitude_angles.jpg                # Reference diagram of UAV body-frame axes (excluded from MIT license)
├── CITATION.cff                       # GitHub citation metadata
├── LICENSE                            # MIT License + Third-Party Media & Dataset Copyright Notice
└── README.md                          # Project documentation
```

---

## Quickstart

1. **Run in Google Colab or Kaggle:**
   Open `Drone_HW_Anomaly_Detection.ipynb` in Colab or Kaggle. The notebook automatically downloads `Fusion_Data.csv` via `kagglehub`.
2. **Run Locally:**
   ```bash
   git clone https://github.com/gpugovka/drones.git
   cd drones
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost kagglehub
   jupyter notebook Drone_HW_Anomaly_Detection.ipynb
   ```

---

## Evaluation Metrics

For each class $c \in \{0, 1, 2, 3, 4\}$ with true positives ($TP_c$), false positives ($FP_c$), false negatives ($FN_c$), and total test samples $N = 2{,}451$:

* **Overall Accuracy:**
  $$\text{Accuracy} = \frac{\sum_{c=0}^{4} TP_c}{N}$$

* **Per-Class Precision ($P_c$) & Recall ($R_c$):**
  $$P_c = \frac{TP_c}{TP_c + FP_c}, \quad R_c = \frac{TP_c}{TP_c + FN_c}$$

* **Per-Class & Macro F1-Score:**
  
  $$F_{1,c} = \frac{2 P_c R_c}{P_c + R_c}, \quad \text{Macro } F_1 = \frac{1}{5}\sum_{c=0}^{4} F_{1,c}$$

* **Per-Class & Macro PR-AUC (Average Precision):**
  
  $$\text{AP}_c = \sum_n (R_{c,n} - R_{c,n-1})*P_{c,n}$$
  
  $$\quad \text{Macro PR-AUC} = \frac{1}{5}\sum_{c=0}^{4} \text{AP}_c$$
  
  where $P_{c,n}$ and $R_{c,n}$ are the precision and recall of class $c$ at probability threshold $n$.

---

## References

1. **Dataset Reference:** [https://doi.org/10.3390/app13074301](https://doi.org/10.3390/app13074301)
2. **Fusion Data Reference:** [https://doi.org/10.1016/j.engappai.2024.107961](https://doi.org/10.1016/j.engappai.2024.107961)
3. **Kinematics & Attitude Reference:** Beard, R. W., & McLain, T. W. (2012). *Small Unmanned Aircraft: Theory and Practice*. Princeton University Press.

---

## License & Citing This Work

The original code, feature engineering, and analysis in this repository are licensed under the **MIT License** (see [LICENSE](LICENSE) for full terms).

### Third-Party Media & Dataset Notice
* `attitude_angles.jpg` (Figure 3.1: *"Definition of axes of motion"*) is reproduced from *Small Unmanned Aircraft: Theory and Practice* by Randal W. Beard and Timothy W. McLain (© 2012 Princeton University Press) strictly for educational purposes under Fair Use and is excluded from the repository's MIT License.
* The TLMUAV dataset (`Fusion_Data.csv`) is licensed by its original authors under the MIT License.

### Citation
If you use or reference this notebook, code, or methodology in your work, please retain the copyright notice and cite it using GitHub's **"Cite this repository"** button or as follows:

* **Author:** Erin B. (2026)
* **Title:** Kinematics-Based UAV Telemetry Fault Detection Using Tree Ensembles
* **Repository:** [https://github.com/gpugovka/drones](https://github.com/gpugovka/drones)
