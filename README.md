# PU-Mask: 3D Point Cloud Upsampling via an Implicit Virtual Mask

### Introduction 

This repository is for our IEEE T-CSVT 2024 paper '[PU-Mask: 3D Point Cloud Upsampling via an Implicit Virtual Mask](https://github.com/liuhaoyun/PU-Mask)'.  

### Installation
This repository is based on Tensorflow and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators. 

For installing tensorflow, please follow the official instructions in [here](https://www.tensorflow.org/install/install_linux). The code is tested under TF1.11 (higher version should also work) and Python 3.6 on Ubuntu 16.04.

For compiling TF operators, please check `tf_xxx_compile.sh` under each op subfolder in `code/tf_ops` folder. Note that you need to update `nvcc`, `python` and `tensoflow include library` if necessary. 

### Note
When running the code, if you have `undefined symbol: _ZTIN10tensorflow8OpKernelE` error, you need to compile the TF operators. If you have already added the `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` but still have ` cannot find -ltensorflow_framework` error. Please use 'locate tensorflow_framework
' to locate the tensorflow_framework library and make sure this path is in `$TF_LIB`.

### Usage
   
1. Compile the TF operators
   Follow the above information to compile the TF operators. 
   
2. Train the model:
    First, you need to download the training patches in HDF5 format from [GoogleDrive](https://drive.google.com/open?id=13ZFDffOod_neuF3sOM0YiqNbIJEeSKdZ) and put it in folder `data/train`.
    Then run:
    
   ```shell
   python pu_mask.py --phase train
   ```

4. Evaluate the model:
    First, you need to download the pretrained model from `/log` (automatically generated during training), extract it and put it in your created folder `model`,
    Then run:
    
   ```shell
   python pu_mask.py --phase test
   ```
  We also provided the trained [model](https://drive.google.com/file/d/1qBVF8m_v79rwrN8RhuAcx8yiSB-l201q/view?usp=drive_link), and you will see the input and output results in the folder `data/test/output`.
   
5. The training and testing mesh files can be downloaded from [GoogleDrive](https://drive.google.com/open?id=1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC).

### Evaluation code
We provide the code to calculate the uniform metric in the evaluation code folder. In order to use it, you need to install the CGAL library. Please refer [this link](https://www.cgal.org/download/linux.html) and  [PU-Net](https://github.com/yulequan/PU-Net) to install this library.
Then:
   ```shell
   cd evaluation_code
   cmake .
   make
   ./evaluation Icosahedron.off Icosahedron.xyz
   ```
The second argument is the mesh, and the third one is the predicted points.

## Citation

If PU-Mask is useful for your research, please consider citing:

    @inproceedings{liu2024pumask,
         title={PU-Mask: 3D Point Cloud Upsampling via an Implicit Virtual Mask},
         author={Liu, Hao and Yuan, hui and Hamzaoui, Raouf and Liu, Qi and  Li, Shuai},
         booktitle = {IEEE Transactions on Circuits and Systems for Video Technology ({T-CSVT})},
         year = {2024}
     }


### Questions

Please contact 'liuhaoxb@gmail.com'

### Thanks

Thank [PU-GAN](https://github.com/liruihui/PU-GAN) for the source code support.
