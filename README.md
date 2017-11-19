# Implement VDSR with TensorFlow

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/kweisamx/VDSR/blob/master/LICENSE)

---
![](https://i.imgur.com/8jdho2m.png)

# Introduction

We present a highly accurate single-image super-resolution (SR) method, learn residuals only and use extremely high learning rates enabled by adjustable gradient clipping

---
## Environment

* Ubuntu 16.04
* Python 2.7.12

---

## Dependency

pip

* Tensorflow
* Opencv
* h5py


If you meet the problem with opencv when run the program
```
libSM.so.6: cannot open shared object file: No such file or directory
```

please install dependency package

```
sudo apt-get install libsm6
sudo apt-get install libxrender1
```

---
## How to train
```
python main.py
```

if you want to see the flag 
```
python main.py -h
```

trainning with 10 layer (default is 20)
```
python main.py --layer 10
```

---
## How to test

If you don't input a Test image, it will be default image
```
python main.py --is_train False
```
then result will put in the result directory


if you want to see the result with custom layer ,ex: 10 layer

```
python main.py --is_train False --layer 10
```


If you want to Test your own iamge

use `test_img` flag

```
python main.py --is_train False --test_img Train/t20.bmp
```

then result image also put in the result directory

---

## Result 

* Origin

    ![Imgur](https://i.imgur.com/hhXBTfC.png)
    
    ![Imgur](https://i.imgur.com/Aizh7Z3.png)
    
* Bicbuic 

    ![Imgur](https://i.imgur.com/7UAzDf6.png)
    
    ![](https://i.imgur.com/VozgDoO.png)
    
* Result

    ![](https://i.imgur.com/cWzYQfG.png)
    

    ![Imgur](https://i.imgur.com/gLHjOMP.png)
    
---
## Reference
[kweisamx/SRCNN](https://github.com/kweisamx/SRCNN)
    
