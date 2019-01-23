# VAE in Fisher-Shannon Plane and Fisher-AutoEncoder
This is a PyTorch implementation of FAE. For mathmatical formulation and detailed description, please refer to paper:
* Huangjie Zheng, Jiangchao Yao, Ya Zhang, Ivor W. Tsang and Jia Wang, Understanding VAEs in Fisher-Shannon Plane, [paper](https://arxiv.org/abs/1807.03723), 2018

## Requirements
The code is compatible with:
* `pytorch 0.4.0`

## Data
The experiments can be run on the following datasets:
* static MNIST: links to the datasets can found at [link](https://github.com/yburda/iwae/tree/master/datasets/BinaryMNIST);
* binary MNIST: the dataset is loaded from PyTorch;
* OMNIGLOT: the dataset could be downloaded from [link](https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat);
* Caltech 101 Silhouettes: the dataset could be downloaded from [link](https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat).
* Frey Faces: the dataset could be downloaded from [link](https://github.com/y0ast/Variational-Autoencoder/blob/master/freyfaces.pkl).
* CIFAR 10: the dataset is loaded from PyTorch.

## Run the experiment
1. Set-up your experiment in `experiment.py`.
2. Run experiment:
```bash
python experiment.py  <your option>
```
## Models
The FAE is implemented in a VAE model with 5 intermediate layers. You can run it by setting `model_name` to `ConvVAE`.

Thanks to the implementation of J. M. Tomczak, 
You can also run a vanilla VAE, a one-layered VAE or a two-layered HVAE with the standard prior or the VampPrior by setting `model_name` argument to either: (i) `vae` or `hvae_2level` for MLP, (ii) `convvae_2level` for convnets, (iii) `pixelhvae_2level` for (ii) with a PixelCNN-based decoder, and specifying `prior` argument to either `standard` or `vampprior`. Modifying the Fisher information computation in the function `calculate_loss` is sufficient.

## Citation

Please cite our paper if you use this code in your research:

```
@article{zheng2018understanding,
  title={Understanding VAEs in Fisher-Shannon Plane},
  author={Zheng, Huangjie and Yao, Jiangchao and Zhang, Ya and Tsang, Ivor W},
  journal={arXiv preprint arXiv:1807.03723},
  year={2018}
}
```

## Acknowledgments
The implementation of VAE is heavily borrowed from the code by Jakub M. Tomczak (VampPrior VAE).
