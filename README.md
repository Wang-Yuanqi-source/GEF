# GEF

## Introduction
 The full name of GEF is "GNN-based Evaluation Framework for FPGA Routing Architecture". It is a GNN-based Evaluation Framework to predict the routability and the area-delay product (ADP) of various FPGA routing architectures. The Rou-P integrates Self-Attention Pooling (SAGPool), while the ADP-P benefits from intermediate supervision through auxiliary node-level labels.

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
Due to the large size of our dataset, it is hosted on Google Drive. Before using the project, please download the dataset using the link provided in ``dataset/dataset.txt``, and extract the contents into the ``dataset/`` directory.

The Rou-P model (for routability prediction) is lacated at ``models/Rou-P/`` director. You need to train the model first by:
```bash
cd models/Rou-P/train
chmod +x run_new_SAG.sh
./run_new_SAG.sh
```
Remember to change the ``dataset_dir`` in ``run_new_SAG.sh`` to you dataset location before you run the command above.


