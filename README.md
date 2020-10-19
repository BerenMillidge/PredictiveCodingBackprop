# PredictiveCodingBackprop
Code for the paper "Predictive Coding Approximates Backprop along Arbitrary Computation Graphs". This repo contains code to reproduce all figures and experiments in the paper. If you find this code useful please cite the paper https://arxiv.org/pdf/2006.04182.pdf

## Installation and Usage
Simply `git clone` the repository to your home computer. The `numerical_results.py` file will recreate the numerical results figures in section 5.1. The `cnn.py` file contains the predictive coding and backprop CNNs used in section 5.2. The `lstm.py` and `rnn_names.py` files contain predictive coding and backprop LSTMs and RNNs used in section 5.3.

## Requirements 

The code is written in [Pyython 3.x] and uses the following packages:
* [NumPY]
* [PyTorch] version 1.3.1
* [TensorFlow] version 1.x (only for downloading shakespeare dataset)
* [matplotlib] for plotting figures

## Citation

If you enjoyed the paper or found the code useful, please cite as: 

```
@article{millidge2020predictive,
  title={Predictive Coding Approximates Backprop along Arbitrary Computation Graphs},
  author={Millidge, Beren and Tschantz, Alexander and Buckley, Christopher L},
  journal={arXiv preprint arXiv:2006.04182},
  year={2020}
}
```
