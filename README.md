# GEF

## Introduction
 The full name of GEF is "GNN-based Evaluation Framework for FPGA Routing Architecture". It is a GNN-based Evaluation Framework to predict the routability and the area-delay product (ADP) of various FPGA routing architectures.  The Rou-P integrates Self-Attention Pooling (SAGPool), while the ADP-P benefits from intermediate supervision through auxiliary node-level labels.

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/Wang-Yuanqi-source/GEF.git
```

### Requirements
The requirements of this repo are listed in the file ``requirements.txt``. If you have any trouble in building the environment, we provide a conda environment called "dgl".

```bash
source dgl/bin/activate
```

### Usage
Due to the large size of our dataset, it is hosted on Google Drive. Before using the project, please download the dataset using the link provided in ``dataset/dataset.txt``, and extract the contents into the ``dataset``/ directory.


