<div align="center">

# Swin Transformer and Generative Adversarial Networks for Accurate Battery Electrode Thickness Prediction in Manufacturing Using Ultrasound Sensing
[**Hamidreza Farhadi Tolie**](https://scholar.google.com/citations?user=nzCbjWIAAAAJ&hl=en&authuser=1)<sup>a, b</sup> · [**Erdogan Guk**](https://scholar.google.com/citations?user=29k7kPAAAAAJ&hl=en&oi=ao)<sup>a, b</sup> · [**James Marco**](https://scholar.google.com/citations?user=icR08CQAAAAJ&hl=en&oi=ao)<sup>a, b</sup> · [**Mona Faraji Niri**](https://scholar.google.com/citations?user=1PK7IocAAAAJ&hl=en&oi=ao)<sup>a, b</sup>

<sup>a</sup> Warwick Manufacturing Group, University of Warwick, coventry, UK

<sup>b</sup> The Faraday Institution, Quad One, Harwell Science and Innovation Campus, Didcot, UK

<hr>

<a href=''><img src='https://img.shields.io/badge/%20ETP-DNN%20-%20Paper?label=Manuscript&labelColor=(255%2C0%2C0)&color=red' alt='Project Page'></a>

<br>

</div>

This repository provides the PyTorch implementation of the electrode thickness prediction model and the synthetic data generation framework developed by the authors. The associated manuscript is available [here]().


## Abstract

> In battery manufacturing, precise electrode thickness measurement is essential for ensuring quality. Existing methods such as mechanical calipers and optical sensors, are limited by their ability to provide continuous, accurate in-line monitoring and suffer from surface sensitivity and implementation/maintenance costs. This study proposes a novel approach that integrates ultrasound-based sensing with a newly designed deep neural network to overcome these limitations. To further enhance prediction accuracy, a conditional generative adversarial network (cGAN) is introduced to generate synthetic data reflective of electrode-specific ultrasound signal characteristics, improving model generalisation. The proposed framework offers a real-time, continuous, non-contact and high-precision solution for electrode thickness monitoring, significantly reducing the need for recalibration. The experimental data collected from a battery production pilot-line confirms the capability. As one of the first works to combine ultrasound sensing with deep generative modelling for this application, it demonstrates the potential for cost-effective, scalable implementation in support of low-carbon battery manufacturing. The source code is publicly available at https://github.com/hfarhaditolie/ETP-DNN/.
---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Feedback](#feedback)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Feedback
If you have any enquires or feedback, please do not hesitate to contact us via @(hamidreza.farhadi-tolie@warwick.ac.uk, h.farhaditolie@gmail.com)

## Acknowledgement
We gratefully acknowledge the developers of cGAN for generously sharing their source code, available [here](https://github.com/jscriptcoder/Data-Augmentation-using-cGAN). Their work significantly facilitated our implementation of a PyTorch-based version, enabling the generation of synthetic ultrasound signals to enhance thickness prediction performance.

## License
This project is licensed under the [MIT License](LICENSE).
