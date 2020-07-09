# Deep Video Deblurring Using Sharpness Features from Exemplars
##### by [Xinguang Xiang](https://imag-njust.net/xinguangxiang/), [Hao Wei](https://github.com/CVHW), [Jinshan Pan](https://jspan.github.io/)


##### Dependencies and Installation
* Python 3 (Recommend to use Anaconda)
* PyTorch0.4.1
* Linux (Tested on Ubuntu 18.04)
* numpy
* tqdm
* imageio
* matplotlib

##### Dataset Preparation
We use the GOPRO_Su dataset to train our models. You can download it from [here](https://github.com/shuochsu/DeepVideoDeblurring) and put the dataset
into **'train_dataset/'**. The dataset should be organized in the following form:

<table><tr><td bgcolor=#FF83FA>
|--dataset name<br/>
&emsp;|--train<br/>
&emsp;&emsp;|--video 01<br/>
&emsp;&emsp;&emsp;|--input<br/>
&emsp;&emsp;&emsp;&emsp;|--frame 01<br/>
&emsp;&emsp;&emsp;&emsp;|--frame 02<br/>
&emsp;&emsp;&emsp;&emsp;|...<br/>
&emsp;&emsp;&emsp;|--GT<br/>
&emsp;&emsp;&emsp;&emsp;|--frame 01<br/>
&emsp;&emsp;&emsp;&emsp;|--frame 02<br/>
&emsp;&emsp;&emsp;&emsp;|...<br/>
&emsp;&emsp;|--video 02<br/>
&emsp;&emsp;...<br/>
&emsp;|--val<br/>
&emsp;|--test<br/>
</td></tr></table>


##### Training 
* Download the FlowNet pretrained model from [Baidu Drive (password:2gca)](https://pan.baidu.com/s/1CXtNHGKF6F27OfIt-o5Pqw) and put it into **'pretrained_model/'**.<br/>
* Prepare the dataset same as above form.<br/>
* Start to train the model. Hyper parameters such as batch size, learning rate, epoch number can be tuned through command line:
<table><tr><td>python main.py --batch_size 256 --lr 1e-4 --epochs 500 --save_models</td></tr></table>


##### Testing
* Download our pretrained model from [Baidu Drive (password:2gca)](https://pan.baidu.com/s/1CXtNHGKF6F27OfIt-o5Pqw) .
* The testing command is shown as follows:
<table><tr><td>python main.py --pre_train model_path --test_only</td></tr></table>


##### Testing your own data
* Put your testing data into 'inference/' which should be organized the same as our given examplars.
* The testing command is shown as follows:
<table><tr><td>python inference.py</td></tr></table>

##### Citation
    @article{xiang2020DVD_SFE,
    title={Deep Video Deblurring Using Sharpness Features from Exemplars},
    author={Xiang, xinguang and Wei, Hao and Pan, jinshan},
    journal={IEEE Transactions on Image Processing},
    year={2020},
    }
