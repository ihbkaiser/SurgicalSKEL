# SurgicalSKEL
## Data Preparation

### Step 1: Download Dataset
Download the Endovis datasets from the following [link](https://drive.google.com/drive/folders/1-tnXXNpxJut1TI7Fn957a62wn5bfGAAc?usp=drive_link).

**We are focusing currently on the Endovis18 dataset. We are actively working to extend support to the Endovis17 dataset in the near future :)**

### Step 2: Data Directory Structure
After downloading, organize the dataset as follows:

```
endovis_data/
├── 17/
│   ├── train/
│   │   ├── annotations/
│   │   ├── binary_annotations/
│   │   └── images/
│   └── val/
│       ├── annotations/
│       ├── binary_annotations/
│       └── images/
└── 18/
    ├── train/
    │   ├── binary_annotations/
    │   └── images/
    └── val/
        ├── annotations/
        ├── binary_annotations/
        ├── images/
        └── sam_features_h/
```
### Step 3: Precompute Embeddings
Run the following script to generate required embeddings:

```bash
python surgicalSKEL/precompute_embeddings.py
```
Make sure the embeddings are organized as follows:
```
endovis_data/
precomputed_class_embeddings/
└── 18/
    ├── train/
    └── val/

precomputed_sam_embeddings/
└── 18/
    ├── train/
    └── val/

segment_anything/
surgicalSKEL/
...
```

## Training

```bash
pip install -r requirements.txt
```
```bash
python surgicalSKEL/train.py
```
