# PCAT: Functionality and Data Stealing from Split Learning by Pseudo-Client Attack

## Abstract

We explore the attack on SL in a more general and challenging situation where the client model is a unknown to the server and gets more complex and deeper. Different from the conventional model inversion, we investigate the inherent privacy leakage through the server model in SL and reveal that clients' functionality and private data can be easily stolen by the server model, and a series of intermediate server models during SL can even cause more leakage. Based on the insights, we propose a new attack on SL: Pseudo-Client ATtack (PCAT). To the best of our knowledge, this is the first attack for a semi-honest server to steal clients' functionality, reconstruct private inputs and infer private labels without any knowledge about the clients' model. The only requirement for the server is a tiny dataset (about 0.1% - 5% of the private training set) for the same learning task. What's more, the attack is transparent to clients, so a server can obtain clients' privacy without taking any risk of being detected by the client. We implement PCAT on various benchmark datasets and models. Extensive experiments testify that our attack significantly outperforms the state-of-the-art attack in various conditions, including more complex models and learning tasks, even in non-i.i.d. conditions. Moreover, our functionality stealing attack is resilient to the existing defensive mechanism.

__Our paper has been accepted at USENIX Security 2023!__

[Paper](https://www.usenix.org/conference/usenixsecurity23/presentation/gao)

- base.py: The basic PCAT attack with label alignment, supporting MNIST and CIFAR10 datasets.

- vanilla.py: It is corresponding to vanilla-PCAT.

Other files:

- config.py: Default settings for training.

- model.py: Containing LeNet5 an VGG models.

- dataio.py: Containing some dataio functions: splitting private and public idx list, customize Dataset class, different dataloader.

- util.py: Containing test function.

## Requirements

This code has been written in python 3.9, PyTorch 2.0.1 and torchvision 0.15.2, however other versions may work.

## Get start

```bash
python main.py
```

If you need to modify the dataset size and model splitting method, please modify the parameters in config.py.

## Notes

Maybe you will have questions about the ./data/mnist/sampleIdx or ./data/cifar10/sampleIdx folder, so here is more explanation. The meaning of each .dat file is as follows: 

Start with ./data/mnist/sampleIdx/sequence/train_glob0.dat as an example. The file contains a list. Each number in this list represents the location (idx) of the sample labeled "0" in the mnist original dataset training set, arranged in ascending order. The corresponding file in the shuffle folder represents the result of shuffling this list. If you want to shuffle the list again, you can load the files in the sequence folder, scramble them and save them again.

## Contact

If you have any questions about the paper and code, please feel free to contact me: gxb1320276347@mail.ustc.edu.cn
