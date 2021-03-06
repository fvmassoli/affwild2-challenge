# A Multi-resolution Approach to Expression Recognition in the Wild

This repository contains the code relative to the paper "[A Multi-resolution Approach to Expression Recognition in the Wild](https://arxiv.org/abs/2103.05723)"
by Fabio Valerio Massoli (ISTI - CNR), Donato Cafarelli (Unipi),  Giuseppe Amato (ISTI - CNR), and Fabrizio Falchi (ISTI - CNR).

**Please note:** 
We are researchers, not a software company, and have no personnel devoted to documenting and maintaing this research code. 
Therefore this code is offered "AS IS". Exact reproduction of the numbers in the paper depends on exact reproduction of many factors, 
including the version of all software dependencies and the choice of underlying hardware (GPU model, etc). Therefore you should expect
to need to re-tune your hyperparameters slightly for your new setup.


## How to run the code

Minimal usage:

```
python main_affwild2.py -o adam -bp <base model checkpoint > -op <output folder path> -df <dataset path> -tr
```

The *base model checkpoint* can be downloaded from [here](https://github.com/ox-vgg/vgg_face2). It is the SE-ResNet-50 with a final 
features vector 2048-dim.


## Reference
For all the details about the training procedure and the experimental results, 
please have a look at the [paper](https://arxiv.org/abs/2103.05723).

To cite our work, please use the following form

```
@article{massoli2021multi,
  title={A Multi-resolution Approach to Expression Recognition in the Wild},
  author={Massoli, Fabio Valerio and Cafarelli, Donato and Amato, Giuseppe and Falchi, Fabrizio},
  journal={arXiv preprint arXiv:2103.05723},
  year={2021}
}
```

## Contacts
If you have any question about our work, please contact [Dr. Fabio Valerio Massoli](mailto:fabio.massoli@isti.cnr.it). 

Have fun! :-D
