# Structure-inclusive similarity based directed GNN: A method that can control information flow to predict drug-target binding affinity

> Exploring the association between drugs and targets is essential for drug discovery and repurposing. Comparing with the traditional methods that regard the exploration as a binary classification task, predicting the drug-target binding affinity can provide more specific information. Many studies work based on the assumption that similar drugs may interact with the same target. These methods constructed a symmetric graph according to the undirected drug similarity or target similarity. Although these similarities can measure the difference between two molecules, it is unable to analyze the inclusion relationship of their substructure. For example, if drug A contains all the substructures of drug B, then in the message-passing mechanism of the graph neural network, drug A should acquire all the properties of drug B, while drug B should only obtain some of the properties of A. To this end, we proposed a structure-inclusive similarity (SIS) which measures the similarity of two drugs by considering the inclusion relationship of their substructures. Based on SIS, we constructed a drug graph and a target graph, respectively, and predicted the binding affinities between drugs and targets by a graph convolutional network-based model. Experimental results show that considering the inclusion relationship of the substructure of two molecules can effectively improve the accuracy of the prediction model. The performance of our SIS-based prediction method outperforms several state-of-the-art methods for drug-target binding affinity prediction. The case studies demonstrate that our model is a practical tool to predict the binding affinity between drugs and targets.

## Installation

Follow the `PyTroch` and `PyG` installation instructions to install the dependencies。

[PyTroch installation instructions](https://pytorch.org/get-started/locally/)

[PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

For other dependencies and their versions please refer to the [requirements.txt](./requirements.txt) file.

``` bash
pip install -r requirements.txt
```


## Data

All `data` is located in the data directory, which is derived from the original KIBA and Davis data.

## Usage

``` python
python main.py 
    [--device DEVICE] 
    [-d [kiba, davis]] 
    [--sim-type SIM_TYPE] 
    [-dt D_THRESHOLD] 
    [-pt P_THRESHOLD] 
    [-s SETTING] 
    [-e EPOCHS] 
    [-b BATCH_SIZE] 
    [-lr LEARNING_RATE] 
    [-l1 LAMBDA_1]
```

The parameters are described below and can also be viewed by executing `python main.py -h`.

``` python
options:
  -h, --help            show this help message and exit
  --device DEVICE       Name of the processor used for computing
  -d [kiba, davis], --dataset [kiba, davis]
                        Name of the selected data set
  --sim-type SIM_TYPE   Similarity Strategy
  -dt D_THRESHOLD, --d_threshold D_THRESHOLD
                        Thresholds for drug relationship graphs
  -pt P_THRESHOLD, --p_threshold P_THRESHOLD
                        Thresholds for protein relationship graphs
  -s SETTING, --setting SETTING
                        Experimental setting
  -e EPOCHS, --epochs EPOCHS
                        Number of training iterations required
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Size of each training batch
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        The step size at each iteration
  -l1 LAMBDA_1, --lambda_1 LAMBDA_1
                        AutoEncoder loss function weights
```

### Quick Example


Epoch: 1000 train loss: 2.561278 train mse: 0.041191 test mse: 0.190082 ci: 0.907858 rm2: 0.752413
Namespace(device='cpu', epochs=1000, dataset='davis', batch_size=512, learning_rate=0.001, lambda_1=1e-05, lambda_2=1, sim_type='sis', weight_decay=0.0, unit=0.1)

``` python
python main.py
    --device=cpu \
    -d davis \
    --sim-type=sis \ 
    -e 1000 \
    -b 512 \
    -lr 0.001 \
    -l1 1e-5
```