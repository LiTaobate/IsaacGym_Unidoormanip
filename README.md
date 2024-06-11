# 1. 北大机械臂开所有门

[GitHub - sectionZ6/UniDoorManip: This is the official repository of UniDoorManip: Learning Universal Door Manipulation Policy Over Large-scale and Diverse Door Manipulation Environments.](https://github.com/sectionZ6/UniDoorManip?tab=readme-ov-file)

**基于交互的模拟部件：**

[SAPIEN](https://sapien.ucsd.edu/)

其内容：

- 20k FPS 的 RGBD + 分割数据的 GPU 并行视觉数据收集系统
- 示例机器人包括四足机器人、移动机械手和单臂机器人
- 示例任务涵盖桌面、运动和灵巧操作。
- 灵活的统一 GPU 任务构建 API

**3D模型搜索下载：**

[3D Warehouse](https://3dwarehouse.sketchup.com/)

![image](https://github.com/LiTaobate/IsaacGym_Unidoormanip/assets/73519321/682c3268-7b04-4f51-b08e-cf2b71c40e59)

usdz模型可以直接放入isaacsim，但是需要付费下载，

其中collada模型可以导入blender软件，再导出为usd模型文件。

论文里的模型文件下载：https://drive.usercontent.google.com/download?id=1Tkkgyn9slUXmcxYcbTKa1Rj3QeM74SbL&export=download&authuser=0

## 1.1安装过程

### conda 环境配置

环境概述：（三个包对于环境依赖不统一有冲突，逐个配置会有问题，所以要以以下生成的环境为主）

使用的cuda11.7 、cudnn9-cuda-11  、cudn pytorch==1.13.0、python3.8

其中cudnn安装使用：sudo  apt-get install cudnn9-cuda-11

```
conda env create -f environment.yml
```

```
conda activate unidoormanip
```

```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

```jsx
pip install -r requirements.txt
pip install -e .
```

### **1.1.1Pointnet++** 环境配置

官网：

https://github.com/erikwijmans/Pointnet2_PyTorch

**报错1：**

运行`pip install -r requirements.txt` 时候报错，发现实际报错是里边这个指令：

```jsx
pip install ./pointnet2_ops_lib/.  
```

单独运行：

```jsx
pip install ./pointnet2_ops_lib/.                                                                                                                                                                            ─╯
Processing ./pointnet2_ops_lib
  Preparing metadata (setup.py) ... done
Requirement already satisfied: torch>=1.4 in /home/litao/anaconda3/envs/unidoormanip/lib/python3.8/site-packages (from pointnet2_ops==3.0.0) (1.13.1)
Requirement already satisfied: typing-extensions in /home/litao/anaconda3/envs/unidoormanip/lib/python3.8/site-packages (from torch>=1.4->pointnet2_ops==3.0.0) (4.12.1)
Requirement already satisfied: nvidia-cuda-runtime-cu11==11.7.99 in /home/litao/anaconda3/envs/unidoormanip/lib/python3.8/site-packages (from torch>=1.4->pointnet2_ops==3.0.0) (11.7.99)
Requirement already satisfied: nvidia-cudnn-cu11==8.5.0.96 in /home/litao/anaconda3/envs/unidoormanip/lib/python3.8/site-packages (from torch>=1.4->pointnet2_ops==3.0.0) (8.5.0.96)
Requirement already satisfied: nvidia-cublas-cu11==11.10.3.66 in /home/litao/anaconda3/envs/unidoormanip/lib/python3.8/site-packages (from torch>=1.4->pointnet2_ops==3.0.0) (11.10.3.66)
Requirement already satisfied: nvidia-cuda-nvrtc-cu11==11.7.99 in /home/litao/anaconda3/envs/unidoormanip/lib/python3.8/site-packages (from torch>=1.4->pointnet2_ops==3.0.0) (11.7.99)
Requirement already satisfied: setuptools in /home/litao/anaconda3/envs/unidoormanip/lib/python3.8/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch>=1.4->pointnet2_ops==3.0.0) (69.5.1)
Requirement already satisfied: wheel in /home/litao/anaconda3/envs/unidoormanip/lib/python3.8/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch>=1.4->pointnet2_ops==3.0.0) (0.43.0)
Building wheels for collected packages: pointnet2_ops
  Building wheel for pointnet2_ops (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> [38 lines of output]
      running bdist_wheel
      running build
      running build_py
      creating build
      creating build/lib.linux-x86_64-cpython-38
      creating build/lib.linux-x86_64-cpython-38/pointnet2_ops
      copying pointnet2_ops/pointnet2_modules.py -> build/lib.linux-x86_64-cpython-38/pointnet2_ops
      copying pointnet2_ops/pointnet2_utils.py -> build/lib.linux-x86_64-cpython-38/pointnet2_ops
      copying pointnet2_ops/_version.py -> build/lib.linux-x86_64-cpython-38/pointnet2_ops
      copying pointnet2_ops/__init__.py -> build/lib.linux-x86_64-cpython-38/pointnet2_ops
      running egg_info
      writing pointnet2_ops.egg-info/PKG-INFO
      writing dependency_links to pointnet2_ops.egg-info/dependency_links.txt
      writing requirements to pointnet2_ops.egg-info/requires.txt
      writing top-level names to pointnet2_ops.egg-info/top_level.txt
      reading manifest file 'pointnet2_ops.egg-info/SOURCES.txt'
      reading manifest template 'MANIFEST.in'
      writing manifest file 'pointnet2_ops.egg-info/SOURCES.txt'
      creating build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src
      creating build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/include
      copying pointnet2_ops/_ext-src/include/ball_query.h -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/include
      copying pointnet2_ops/_ext-src/include/cuda_utils.h -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/include
      copying pointnet2_ops/_ext-src/include/group_points.h -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/include
      copying pointnet2_ops/_ext-src/include/interpolate.h -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/include
      copying pointnet2_ops/_ext-src/include/sampling.h -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/include
      copying pointnet2_ops/_ext-src/include/utils.h -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/include
      creating build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/src
      copying pointnet2_ops/_ext-src/src/ball_query.cpp -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/src
      copying pointnet2_ops/_ext-src/src/ball_query_gpu.cu -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/src
      copying pointnet2_ops/_ext-src/src/bindings.cpp -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/src
      copying pointnet2_ops/_ext-src/src/group_points.cpp -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/src
      copying pointnet2_ops/_ext-src/src/group_points_gpu.cu -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/src
      copying pointnet2_ops/_ext-src/src/interpolate.cpp -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/src
      copying pointnet2_ops/_ext-src/src/interpolate_gpu.cu -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/src
      copying pointnet2_ops/_ext-src/src/sampling.cpp -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/src
      copying pointnet2_ops/_ext-src/src/sampling_gpu.cu -> build/lib.linux-x86_64-cpython-38/pointnet2_ops/_ext-src/src
      running build_ext
      error: [Errno 2] No such file or directory: ':/usr/local/cuda:/usr/local/cuda/bin/nvcc'
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for pointnet2_ops
  Running setup.py clean for pointnet2_ops
Failed to build pointnet2_ops
ERROR: Could not build wheels for pointnet2_ops, which is required to install pyproject.toml-based projects
```

官网寻找问题：
![image](https://github.com/LiTaobate/IsaacGym_Unidoormanip/assets/73519321/6f2c222b-2336-4ebc-9f31-961bf1ad66ad)


跑第二种方案，依然报错：

```jsx
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"                                                                                        ─╯

Collecting pointnet2_ops
  Cloning git://github.com/erikwijmans/Pointnet2_PyTorch.git to /tmp/pip-install-18othrsm/pointnet2-ops_b4243ffc2ff740bea1f0f42678bd0849
  Running command git clone --filter=blob:none --quiet git://github.com/erikwijmans/Pointnet2_PyTorch.git /tmp/pip-install-18othrsm/pointnet2-ops_b4243ffc2ff740bea1f0f42678bd0849
  fatal: unable to connect to github.com:
  github.com[0: 140.82.113.4]: errno=Connection timed out

  error: subprocess-exited-with-error
  
  × git clone --filter=blob:none --quiet git://github.com/erikwijmans/Pointnet2_PyTorch.git /tmp/pip-install-18othrsm/pointnet2-ops_b4243ffc2ff740bea1f0f42678bd0849 did not run successfully.
  │ exit code: 128
  ╰─> See above for output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× git clone --filter=blob:none --quiet git://github.com/erikwijmans/Pointnet2_PyTorch.git /tmp/pip-install-18othrsm/pointnet2-ops_b4243ffc2ff740bea1f0f42678bd0849 did not run successfully.
│ exit code: 128
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.

```

解决方案：

不用官方的指令：“
`pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"`

改用以下指令：

```jsx
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib" 
```

缺少Pytorch3D

```jsx
pip install pytorch3d -i https://pypi.tuna.tsinghua.edu.cn/simple  
```

### 1.1.2 unidoormanip 环境配置

```jsx
conda create -n unidoormanip python=3.8
conda activate unidoormanip
```

```jsx
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia  
```

### 1.1.3 isaacgym下载安装

目前isaacgym已经弃用，无法下载，复现时候使用之前下载的版本

安装cd /更改为你放置的目录/isaacgym/python/参考链接：

[20231126-超详细Isaac Gym安装教程（基于双系统版本）-CSDN博客](https://blog.csdn.net/m0_37802038/article/details/134629194)

```python
cd /isaacgym/python/
conda activate unidoormanip
pip install -e .
```

运行式例报错：

```jsx

cd examples 
python joint_monkey.py
```

报错：

“ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory” 

解决：
