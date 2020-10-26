# NSCaching
The Code for our paper ["NSCaching: Simple and Efficient Negative Sampling for Knowledge Graph Embedding"](https://arxiv.org/abs/1812.06410) and the short paper has been published in ICDE2019 long version has been publised in VLDBJ.

The extension of NSCaching: "Efficient, Simple and Automated Negative Sampling for Knowledge Graph Embedding" has been accepte by the VLDB Journal.

Readers are welcomed to fork this repository to reproduce the experiments and follow our work. Please kindly cite our paper

    @inproceedings{zhang2019nscaching,
      title={NSCaching: Simple and Efficient Negative Sampling for Knowledge Graph Embedding},
      author={Zhang, Yongqi and Yao, Quanming and Shao, Yingxia and Chen, Lei},
      booktitle={2019 IEEE 35th International Conference on Data Engineering (ICDE)},
      pages={614--625},
      year={2019},
      organization={IEEE}
    }

And the automated version

    @article{zhang2020efficient,
        title{Efficient, Simple and Automated Negative Sampling for Knowledge Graph Embedding},
        author={Zhang, Yongqi and Yao, Quanming and Chen, Lei},
        booktitle={The VLDB journal},
        year={2020},
        publisher={Springer}
    }

## Instructions
For the sake of ease, a quick instruction is given for readers to reproduce the whole process on fb15k dataset.
Note that the programs are tested on Linux(Ubuntu release 16.04), Python 3.7 from Anaconda 4.5.11.

Install PyTorch (>0.4.0)
    
    conda install pytorch -c pytorch
    
Get this repo

    git clone https://github.com/yzhangee/NSCaching
    cd NSCaching
Get dataset from THUNLP-OpenKE
  
    git clone https://github.com/thunlp/OpenKE
    mv OpenKE/benchmarks ../KG_Data

### NSCaching+scratch on FB15K

    python train.py

### NSCaching (auto) on FB15K

    python auto_search.py   
or
    bash run_fb15k.sh

