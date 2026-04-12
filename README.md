# Multimodal Predictive Model for Clinical Outcomes in Emergency Departments 

This project generates and evaluates multimodal machine learning models that leverage structured data, clinical notes, and ECG to predict emergency department (ED) outcomes.


## Overview

This project explores multimodal prediction of ED outcomes using:
- Structured clinical data
- Clinical note data
- ECG data

Five prediction tasks are evaluated at different clinical time points:
1. Hospitalization at ED triage
2. Critical outcome at ED triage
3. 72-hour ED revisit at ED dispostion (not included in paper due to poor performance)
4. Critical outcome at ED disposition
5. Hospitalization at ED disposition

Each modality is modeled independently and combined using late fusion multimodal techniques.


## Data

This project uses:
- MIMIC-IV-ED (structured data)
- MIMIC-IV-Note (clinical notes data)
- MIMIC-IV-ECG (ECG data)


## Repository Structure

### Dataset Construction
- `extract_master_dataset.ipynb` Generates original master dataset
- `data_general_processing.ipynb` Processes master dataset for structured data prediction model
- `add_ecg_to_masterdataset_*.py` ECG data integration into master dataset*
- `add_notes_to_masterdataset_*.py` Clinical note data integration into master dataset*
- `clean_notes_*.py` Clinical note preprocessing*
- `MV-COMBINE-DATASETS_*.ipynb` Merges all modalities into final datasets*

* done separately for ED disposition and ED triage prediction timepoints

### Indvidual Modality Models

- `Task*-STRUCTUREDDATA-FINAL.ipynb` Structured data models
- `Task*-CLINICALNOTES-FINAL.ipynb` Clinical note data models
- `Task*-ECGDATA-FINAL.ipynb` ECG data models

### Multimodal Models
- `MULTIMODAL_MODEL_task1.ipynb` Task 1 Multimodal Model
- `MULTIMODAL_MODEL_task2.ipynb` Task 2 Multimodal Model
- `MULTIMODAL_MODEL_task3.ipynb` Task 3 Multimodal Model
- `MULTIMODAL_MODEL_task4.ipynb` Task 4 Multimodal Model
- `MULTIMODAL_MODEL_task5.ipynb` Task 5 Multimodal Model


## Instructions

### 1. Preprocess Data
Run:
- `extract_master_dataset.ipynb`
- `data_general_processing.ipynb`
- `add_ecg_to_masterdataset_*.py`
- `add_notes_to_masterdataset_*.py`
- `clean_notes_*.py`
- `MV-COMBINE-DATASETS_*.ipynb`

### 2. Train Individual Modality Models
Run:
- Structured: `Task*-STRUCTUREDDATA-FINAL.ipynb`
- Notes: `Task*-CLINICALNOTES-FINAL.ipynb`
- ECG: `Task*-ECGDATA-FINAL.ipynb`

### 3. Train Multimodal Models
Run:
- `MULTIMODAL_MODEL_task*.ipynb`
