# Plant Disease Detection from Leaf Images

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-red.svg)]()

> **⚠️ Note:** This repository contains code for manuscripts currently under review. Code and documentation are subject to change. Final versions will be linked to published papers upon acceptance.

## Overview

This repository provides Python scripts for plant disease detection using deep learning on leaf images. The code supports data preprocessing, model training, evaluation, and visualization of results, as implemented in our submitted papers.

## Dataset

This work uses the **New Plant Diseases Dataset** available on Kaggle:

> **New Plant Diseases Dataset**  
> *Image dataset containing different healthy and unhealthy crop leaves*  
> [https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)[reference:0]

**Dataset Summary:**
- ~87,000 RGB images of healthy and diseased crop leaves[reference:1]
- 38 distinct classes (14 crop species × various disease/healthy conditions)[reference:2]
- 80/20 train/validation split preserving directory structure[reference:3]
- Additional test directory with 33 images for prediction[reference:4]
- Dataset size: 1.43 GB[reference:5]

The dataset is recreated using offline augmentation from the original [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)[reference:6].

## Repository Structure
