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

#### Rou-P
The Rou-P model (for routability prediction) is lacated at ``models/Rou-P/`` director. You need to train the model first by:
```bash
cd models/Rou-P/train
chmod +x run_new_SAG.sh
./run_new_SAG.sh
```
Remember to change the ``dataset_dir`` in ``run_new_SAG.sh`` to you dataset location before you run the command above.

We also provide a pre-trained optimal model, which achieves a prediction accuracy of 94.56% for routability. It is located at ``models/Rou-P/best_model/best_routability_model.pth``

#### ADP-P
The ADP-P model (for ADP prediction) is lacated at ``models/ADP-P/`` director. You need to train the model first by:
```bash
cd models/ADP-P/train/{benchmark}
chmod +x run_new_gnn.sh
./run_new_gnn.sh
```
Remember to change the ``dataset_dir`` in ``run_new_gnn.sh`` to you dataset location before you run the command above.

If you wish to perform training on all benchmarks in a single run, you can use ``models/ADP-P/train/run_all_models.sh``.
```bash
cd models/ADP-P/train/
chmod +x run_all_models.sh
./run_all_models.sh
```

We also provide a pre-trained optimal model, which achieves a prediction accuracy of 94.56% for routability. It is located at ``models/Rou-P/best_model/best_routability_model.pth``


