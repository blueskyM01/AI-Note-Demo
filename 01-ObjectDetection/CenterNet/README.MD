#  [CenterNet](https://arxiv.org/abs/1904.07850)
![](./Figure/overview.png)
[B站讲解视频1](https://www.bilibili.com/video/BV1r44y1a75j/?spm_id_from=333.337.search-card.all.click&vd_source=42f1e486842e9add1356fbbd0f4159f3)

[B站讲解视频2，本项目中采用的代码基于此视频](https://www.bilibili.com/video/BV1mK411u77S/?spm_id_from=333.337.search-card.all.click&vd_source=42f1e486842e9add1356fbbd0f4159f3)


## 1. Network
 - Backbone: [ResNet](), [Hourglass](), [Deep Layer Aggregation]()
 - [Decoder](): UpSample + BN + Activation作为一个block，以此堆叠三次作为一个Decoder。其中CenterNet的UpSample为反卷积，激活函数为ReLU。需要注意的是，三个反卷积的核大小都为4x4，卷积核的数目分别为256，128，64。那么经过Decoder之后，feature map的宽高维度则变为原来1/4（比较重要，后面会反复用到），通道维度为64。

 - Head:


## 1.2 Loss function
- CIOU（Complete Intersection over Union）
## 1.3 Annotation

## 1.4 Code
### 1.4.1 Train (x86)
- Environment setup
    - Docker pull
        - ` $ sudo docker pull nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04`
    - 启动镜像
        - `$ sudo docker run --name torch13-1 -itd  -v /home/ntueee/yangjianbing:/root/code -p 2002:22   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --gpus all --shm-size="12g" --restart=always nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04`
    - install ssh (Note that enter container first!)
        - `$ apt-get update`
        - `$ apt-get install vim`
        - `$ apt-get install openssh-server`
        - 设置root密码，后续登录会用到: `$ passwd`
        - 修改配置文件: `$ vim /etc/ssh/sshd_config`
            ``` 
            #PermitRootLogin prohibit-password
            PermitRootLogin yes
            UsePAM yes 修改为 no
            注释这一行PermitRootLogin prohibit-password
            添加一行PermitRootLogin yes
            UsePAM yes 修改为 no #禁用PAM
            ```
        - 重启ssh服务: `$ service ssh restart`
        - 添加开机启动
            - 新建`power_launch.sh`文件，放到根目录：`/root`下，`power_launch.sh`添加如下内容
                ```
                #!/bin/sh -e
                service ssh start &
                ```
            - 获取读写权限：`chmod 777 /root/power_launch.sh`
            - 编辑`~/.bashrc`: `vim ~/.bashrc`，在下面添加
                ```
                if [ -f /root/power_launch.sh ]; then
                        ./root/power_launch.sh
                fi
                ```
    - Install pytorch 1.13
        - 创建python软链接：`$ ln -s /usr/bin/python3 /usr/bin/python`
        - install pip package: `apt-get install pip`
        - 下载[cu116/torch-1.13.0%2Bcu116-cp38-cp38-linux_x86_64.whl](https://download.pytorch.org/whl/cu116/torch-1.13.0%2Bcu116-cp38-cp38-linux_x86_64.whl)，安装：`pip install torch-1.13.0+cu116-cp38-cp38-linux_x86_64.whl`
        - 下载[cu116/torchvision-0.14.0%2Bcu116-cp38-cp38-linux_x86_64.whl](https://download.pytorch.org/whl/cu116/torchvision-0.14.0%2Bcu116-cp38-cp38-linux_x86_64.whl)，安装：`pip install torchvision-0.14.0+cu116-cp38-cp38-linux_x86_64.whl`
        - 测试是否用的GPU：
            ```
            import torch
            flag = torch.cuda.is_available()
            if flag:
                print("CUDA is available")
            else:
                print("CUDA is not available")
            ```
    - Install COCO API
        - pip install Cython
        - pip install pycocotools==2.0.0

### 1.4.2 Forward 
- (x86)

- arm

### 1.4.2 Estimation
