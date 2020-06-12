# GAN
## 主要框架
1. 万物皆可GAN？（陈涵玥）
	——基于Pix2pix和CycleGAN的多种任务实现
2. GAN在具体任务中的针对性改进（谭淑敏）
	——StyleGAN2 生成人像
3. GAN思想的延伸（耿云腾）
	——CAAE

## Pix2pix&CycleGAN
### 环境部署
* Python 3
* PyTorch
* Tensorflow 2.0以下

### 代码功能说明
参考代码：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix



#### Enhance
enhance.py用于数据集的增强，修改数据集文件夹的地址，和加强后的保存地址，运行enhance.py

将原图片进行左右镜像、旋转、增加亮度来增大数据量



#### Pix2pix train/test
利用pix2pix完成灰度图像色彩化的实验

* train

在训练色彩化pix2pix模型时输入：
> python3 train.py --dataroot ./datasets/colorization --name color_pix2pix --model colorization --display_id -1 --gpu_ids -1 --no_html

	训练的模型保存在./checkpoints/color_pix2pix中

在训练中断后重新开始时加上--continue_train 可以接着中断前的latest_net_G.pth和latest_net_D.pth的参数继续训练


* test

在测试风景照图片时输入：
>python3 test.py --dataroot ./datasets/mytest/color/landscape --name color2 --model colorization 

	其中mytest/color/landscape可以换成datasets中其他测试图片的文件夹地址,例如在测试人物图片时输入:
	> python3 test.py --dataroot ./datasets/mytest/color/people --name color2 --model colorization 


* 测试模型的选择

在实验中采用了两种训练集：
1. 风景照较多的训练集训练结果在./checkpoints/color_pix2pix中；
2. 风景照较多的训练集训练结果在./checkpoints/color2中。
可以替换模型进行测试，如下：
>python3 test.py --dataroot ./datasets/mytest/color/landscape --name color_pix2pix --model colorization 
	
如果选定某一个epoch的模型进行测试，则在后面加上--epoch n（其中n%5==0）

所有结果保存在results文件夹中



#### CycleGAN train/test

* train

在梵高画风肖像实验中输入：
>python3 train.py --dataroot ./datasets/style_vangogh --name vangogh_portrait --model cycle_gan --display_id -1 --gpu_ids -1 --no_html --lr 0.0002

	style_vangogh中为自己经过数据集增强建立的肖像画数据集

在进行其他实验时，从参考代码中的数据集下载链接下载训练集和测试集: http://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

* test

下载参考代码中的预训练模型:http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/
直接利用预训练模型对各种cycleGAN的多种应用进行实验:
>python3 test.py --dataroot datasets/mytest/style --name style_monet --model test --no_dropout

	./mytest/style中为自己准备的测试图像，也可以更换为datasets中其他测试集的地址

	--name之后 style_monet可以替换为checkpoints中的其他cycleGAN预训练模型

### 实验效果总结



### 参考文献


## StyleGAN

## CAAE
### 环境部署
* Python 3
* Tensorflow 1.7.0
* Scipy 1.0.0
### 数据集
UTKFace，约21万张带有标记的人脸照片。在data文件夹下解压UTKFace.tar.gz即可。
### 代码功能
* train
> python main.py

该训练过程中epoch为50，使用独立显卡可以较快地完成。受限于硬件设施，本项目使用笔记本电脑的CPU进行训练，50个epoch共耗时75小时33分钟。
