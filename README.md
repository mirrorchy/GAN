# GAN的应用与延伸
## 主要框架
1. 万物皆可GAN？（陈涵玥）
	——基于Pix2pix和CycleGAN的多种任务实现
2. GAN在具体任务中的针对性改进（谭淑敏）
	——StyleGAN2 生成人像
3. GAN思想的延伸（耿云腾）
	——CAAE

## Pix2pix&CycleGAN
### 1 环境部署
* Python 3
* PyTorch 0.4+
* scipy
* Tensorflow 2.0以下

### 2 代码功能说明
参考代码：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

（此处由于上传文件大小限制未上传checkpoints中训练好的模型，模型链接：https://pan.baidu.com/s/1SeSAMDzaysmsHlFG1FW8aQ  密码:yn5f）

说明：百度网盘中存有自己训练的color_pix2pix模型和color2模型，均只保留了最后一次训练的生成网络，可以用于test,但不能用于continue_train

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

	style_vangogh中为自己经过数据集增强建立的肖像画数据集（此repo中style_vangogh数据集为扩充前数据）

在进行其他实验时，从参考代码中的数据集下载链接下载训练集和测试集: http://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

* test

下载参考代码中的预训练模型:http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/
直接利用预训练模型对各种cycleGAN的多种应用进行实验:
>python3 test.py --dataroot datasets/mytest/style --name style_monet --model test --no_dropout

	./mytest/style中为自己准备的测试图像，也可以更换为datasets中其他测试集的地址

	--name之后 style_monet可以替换为checkpoints中的其他cycleGAN预训练模型

### 3 实验效果总结
#### Pix2pix
1. 收集一些人像和风景图片建立数据集，初始时大量为风景照，人像占比约1/10。训练模型保留在checkpoints/color_pix2pix中。
2. 训练到85epoch观察到风景图片虽然部分颜色不能完全还原但是整体效果良好，且继续训练变化不大，但是人物灰度图像色彩化效果差。
	风景照色彩化效果良好；在一些原始色彩不是很常见的风景照中（如天空粉色的霞光、整体偏黄的场景）虽然色彩化图像与ground truth不同，但是符合普遍认知，参见示例图

	而人物图像还原效果经常出现整体偏蓝，或者脸部出现其他色斑的情况，且皮肤颜色失真。观察数据集发现人物图像较少，所以尝试修改数据集使得模型更适用于人物肖像的色彩化

以上结果可以在results/color_pix2pix中看到，保留了最后一个模型的测试结果（风景照为1xx开头，肖像为imgxx开头）

3. 新建立大部分为人像少部分为风景的数据集（由于人像数据收集少，自主编写enhance.py进行数据增强）
4. 由于已经有部分颜色在之前主要是风景照的训练中已经有较好的对应（如植被的绿色、天空的蓝色等），因此尝试利用代码中提供的continue_train模式，在之前的模型参数的基础上进行新数据集的训练。训练模型保留在checkpoints/color2中。训练了55epoch
	在新模型中可以看到人物皮肤的颜色更加真实，但是效果不太稳定。有较大波动，且同一模型对不同照片色彩化效果偏差较大。

	总结各种人物照片还原结果，可以大致概括出：
	1. 对于原始色彩较为单一的场景还原效果好；
	2. 依然会出现有色斑的情况；
	3. 人脸在画面中占比过大或过小更容易出现较大偏差。

以上结果可以在results/color2中看到，保留了最后一个测试结果

#### CycleGAN
在进行自己的梵高画风肖像的实验中，由于CPU运行较慢，因此模型效果不佳。
这一部分主要采用参考代码中的预训练模型。
* 图像风格迁移
分别对Monet, Cezanne, Van Gogh, Ukiyoe四种画风进行了测试，得到的结果保存在在results文件夹中。
	测试集中不同类型风景照效果均较好。
	对自己的测试数据，选择电脑合成图像，拍摄的偏灰白图像，人物肖像三张图片进行测试。
	整体测试效果表现出：
	1. 对不同画风色彩的偏好，勾勒轮廓时笔画的粗细的模拟比较好。
	2. 对于画家的笔触没有准确模拟。比如在梵高画风中常用较短笔触进行颜色覆盖，此类特征难以表现。
	3. 在人像上无论是风格化还是画还原为照片效果都不好，推测原因是数据集大多为风景照。

* 地图的转换
对测试集中街道图像还原为卫星图像进行了测试，效果良好。又选用未名湖街道地图和清朝畅春园地图进行测试。
	测试过程中
	1. 房屋、水域、绿地的对应关系清晰，效果好。
	2. 不同街道地图的颜色对应关系不同，因此自己的测试数据可能效果不理想。
	3. 街道和树木的遮盖关系难以表现。

* 物体之间的转换
测试了从橙子到苹果转换的预训练模型，由于训练集的选择不当，此部分出现很大偏差。且背景的判断和分离较为困难。


#### 参考文献
Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. ICCV 2017

Image-to-Image Translation with Conditional Adversarial Networks.
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros. In CVPR 2017.

## StyleGAN

## FaceAging-CAAE
### 环境部署
* Python 3
* Tensorflow 1.7.0
* Scipy 1.0.0
### 数据集
UTKFace，约21万张带有标记的人脸照片。在data文件夹下解压UTKFace.tar.gz即可。
### 代码功能

该部分代码在文件夹FaceAging中。

#### train
> python main.py

该训练过程中epoch为50，使用独立显卡可以较快地完成。受限于硬件设施，本项目使用笔记本电脑的CPU进行训练，50个epoch共耗时75小时33分钟。

训练过程中会创建一个新的文件夹./save，其中包含summary，samples，test和checkpoint。
* ./save/samples保存每个epoch的重构图像。
* ./save/test保存每个epoch测试的结果。
* ./save/chechpoint保存模型。
* ./save/summary保存batch wise losses，可视化的方法：
>cd save/summary
>tensorboard --logdir

为了可视化的目的我们对它进行了低通滤波。原始的记录保存在summary文件夹中。
#### test
>python main.py --is_train False --testdir your_image_dir --savedir save

正常运行的情况下，屏幕会输出以下代码：

	Building graph ...
	
	Testing Mode

	Loading pre-trained model ...
	SUCCESS ^_^

	Done! Results are saved as save/test/test_as_xxx.png

#### files
* FaceAging.py 用来构建和初始化CAAE模型，并且完成训练和测试。
* ops.py 包含FaceAging.py调用的各种函数，实现卷积、去卷积、全连接、ReLU、加载和保存图像等功能。
* main.py为调用FaceAging.py的主程序。
### 实验结果
* 1~10 epoch时，生成的假脸真实度较低，人脸年龄的区分度也较差。
* 11~30 epoch时，生成的人脸真实度尚可，可以比较清晰地看出每个人在不同年龄段的相貌。
* 31~50 epoch时，生成真实度较高的、年龄区分明显的图像组合。
* 该训练模型对脸部特征清晰、周围无干扰环境的图像效果较好，生成的图像从婴幼儿（0至5岁）到老年人（70至80岁）都比较清晰，而对相对模糊的图像
效果就较差。在本次训练过程中，大部分白人的效果较好，而大部分黑人的轮廓特征不太明显，这就和图片的清晰度等因素有关，比如训练集中的黑人图像
面部特征不突出。
### 参考文献
Zhifei Zhang, Yang Song, and Hairong Qi. "Age Progression/Regression by Conditional Adversarial Autoencoder." IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

