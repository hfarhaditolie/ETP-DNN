<div align="center">

# Swin Transformer and Generative Adversarial Networks for Accurate Battery Electrode Thickness Prediction in Manufacturing Using Ultrasound Sensing
[**Hamidreza Farhadi Tolie**](https://scholar.google.com/citations?user=nzCbjWIAAAAJ&hl=en&authuser=1)<sup>a, b</sup> · [**Erdogan Guk**](https://scholar.google.com/citations?user=29k7kPAAAAAJ&hl=en&oi=ao)<sup>a, b</sup> · [**James Marco**](https://scholar.google.com/citations?user=icR08CQAAAAJ&hl=en&oi=ao)<sup>a, b</sup> · [**Mona Faraji Niri**](https://scholar.google.com/citations?user=1PK7IocAAAAJ&hl=en&oi=ao)<sup>a, b</sup>

<sup>a</sup> Warwick Manufacturing Group, University of Warwick, coventry, UK

<sup>b</sup> The Faraday Institution, Quad One, Harwell Science and Innovation Campus, Didcot, UK

<hr>

<a href='https://www.sciencedirect.com/science/article/pii/S0378775325017550'><img src='https://img.shields.io/badge/%20ETP DNN%20-%20Paper?label=Manuscript&labelColor=(255%2C0%2C0)&color=red' alt='Project Page'></a>

<br>

</div>

This repository provides the PyTorch implementation of the electrode thickness prediction model and the synthetic data generation framework developed by the authors. The associated manuscript is available [here]().


## Abstract

> In battery manufacturing, precise electrode thickness measurement is essential for ensuring quality. Existing methods such as mechanical calipers and optical sensors, are limited by their ability to provide continuous, accurate in-line monitoring and suffer from surface sensitivity and implementation/maintenance costs. This study proposes a novel approach that integrates ultrasound-based sensing with a newly designed deep neural network to overcome these limitations. To further enhance prediction accuracy, a conditional generative adversarial network (cGAN) is introduced to generate synthetic data reflective of electrode-specific ultrasound signal characteristics, improving model generalisation. The proposed framework offers a real-time, continuous, non-contact and high-precision solution for electrode thickness monitoring, significantly reducing the need for recalibration. The experimental data collected from a battery production pilot-line confirms the capability. As one of the first works to combine ultrasound sensing with deep generative modelling for this application, it demonstrates the potential for cost-effective, scalable implementation in support of low-carbon battery manufacturing. The source code is publicly available at https://github.com/hfarhaditolie/ETP-DNN/.
---

![Image Description](https://ars.els-cdn.com/content/image/1-s2.0-S0378775325017550-gr2_lrg.jpg)

![Image Description](https://ars.els-cdn.com/content/image/1-s2.0-S0378775325017550-gr3_lrg.jpg)

## Table of Contents

- [Usage](#usage)
- [Citation](#citation)
- [Feedback](#feedback)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Usage
To run the machine learning (ML) and deep learning (DL) models, simply execute the corresponding main scripts. Each dataset has its own associated ML and DL scripts, which handle both training and testing when run.

For example, to run the models for the cathode dataset:

```bash
python3 mainMLCathode.py # Run the ML model
python3 mainDLCathode.py # Run the DL model
```

### Fine-Tuning DL Models Using Synthetic Data

This section provides a workflow to generate synthetic data using CGAN and fine-tune DL models for both **cathode** and **anode** datasets.

#### 📁 Directory Structure
<pre lang="markdown"> ## 📁 Directory Structure ``` ├── data_augmentation_using_cgan.py ├── Fine Tuning/ │ ├── trainCathode.py │ ├── FineTuneCathode.py │ ├── trainAnode.py │ └── FineTuneAnode.py ``` </pre>

🔧 Step-by-Step Instructions

#### 1. Generate Synthetic Data
Use the CGAN-based script to create synthetic data for both cathode and anode datasets:

```bash
python3 data_augmentation_using_cgan.py
```
#### 2. Train Model on Synthetic Data
```bash
python3 Fine Tuning/trainCathode.py # Cathode
python3 Fine Tuning/trainAnode.py # Anode
```
#### 3. Fine-Tune Model on Real Data
```bash
python3 Fine Tuning/FineTuneCathode.py # Cathode
python3 Fine Tuning/FineTuneAnode.py # Anode
```
#### 📌 Notes

- ✅ Ensure synthetic data is generated **before** training the model.
- 🔁 Fine-tuning should always follow training on synthetic data.
- 📂 Adjust file and folder paths if you modify the project structure.
- 🧠 The synthetic data is generated using a Conditional GAN (CGAN).
- 🧪 Both training and fine-tuning scripts include training and testing stages.
- 📊 Make sure the real datasets (anode/cathode) are available before fine-tuning.
  
## Citation
```bash
@article{FARHADITOLIE2025237919,
title = {Swin Transformer and generative adversarial networks for accurate battery electrode thickness prediction in manufacturing using ultrasound sensing},
journal = {Journal of Power Sources},
volume = {655},
pages = {237919},
year = {2025},
issn = {0378-7753},
doi = {https://doi.org/10.1016/j.jpowsour.2025.237919},
url = {https://www.sciencedirect.com/science/article/pii/S0378775325017550},
author = {Hamidreza {Farhadi Tolie} and Erdogan Guk and James Marco and Mona {Faraji Niri}},
keywords = {Battery manufacturing, Ultrasonic sensing, Machine learning, Deep neural networks, Process optimisation, Quality control},
abstract = {In battery manufacturing, precise electrode thickness measurement is essential for ensuring quality. Existing methods such as mechanical calipers and optical sensors, are limited by their ability to provide continuous, accurate in-line monitoring and suffer from surface sensitivity and implementation/maintenance costs. This study proposes a novel approach that integrates ultrasound-based sensing with a newly designed deep neural network to overcome these limitations. To further enhance prediction accuracy, a conditional generative adversarial network (cGAN) is introduced to generate synthetic data reflective of electrode-specific ultrasound signal characteristics, improving model generalisation. The proposed framework offers a real-time, continuous, non-contact and high-precision solution for electrode thickness monitoring, significantly reducing the need for recalibration. The experimental data collected from a battery production pilot-line confirms the capability. As one of the first works to combine ultrasound sensing with deep generative modelling for this application, it demonstrates the potential for cost-effective, scalable implementation in support of low-carbon battery manufacturing. The source code is publicly available at https://github.com/hfarhaditolie/ETP-DNN/.}
}
```

## Feedback
If you have any enquires or feedback, please do not hesitate to contact us via @(hamidreza.farhadi-tolie@warwick.ac.uk, h.farhaditolie@gmail.com)

## Acknowledgement
We gratefully acknowledge the developers of cGAN for generously sharing their source code, available [here](https://github.com/jscriptcoder/Data-Augmentation-using-cGAN). Their work significantly facilitated our implementation of a PyTorch-based version, enabling the generation of synthetic ultrasound signals to enhance thickness prediction performance.

## License
This project is licensed under the [MIT License](LICENSE).
