# 附录A. Python开发环境配置

尽管在Windows操作系统部署Python环境是可行，但考虑到Linux操作系统对程序设计及调试支持性更好以及其他一些方面的优势，本书中全部程序代码均在Linux发行版Ubuntu操作系统下调试运行。本书中的代码同时在Linux和Windows环境下调试通过。

考虑到Python程序开发中可能会涉及多版本开源Package或Library的使用，为此，本书推荐使用Anaconda作为环境管理工具，以隔离和维护可能彼此间存在冲突的不同Python环境。考虑到Visual Studio Code（以下简称Vscode）的轻量化和其丰富的拓展功能，本书选择Vscode作为集成开发环境（Integrated Development Environment, IDE）。

## A.1 Ubuntu安装配置

读者可选择在虚拟机内构建Ubuntu系统，或为计算机安装第二操作系统。在有多台计算机的情况下，也可为第二台计算机安装Ubuntu系统并使用SSH工具远程连接进行程序开发。首先前往Ubuntu官方网站[^1]

下载系统的ISO镜像文件，建议读者选择带有用户图形界面的Desktop版本。

考虑到读者具体情况的不同以及Ubuntu安装过程的用户友好性，此处不再赘述Ubuntu系统安装过程，读者可参考网络资源完成安装。建议读者在安装过程中使用英文作为系统语言，尽管这不是必要的。首次完成安装后，建议读者首先更换Ubuntu国内软件源，随后在终端内执行以下指令以更新系统软件：

```bash
sudo apt update
sudo apt upgrate
```

## A.2 Anaconda安装配置

首先前往Anaconda官方网站[^2]下载其安装文件，随后在终端执行以下指令开始安装过程[^3]。

```bash
cd ~/Downloads 
bash Anaconda3-2022.10-Linux-x86_64.sh
```

在终端内参照提示按Enter键确认并持续按Enter键继续阅读软件许可，若无异议，输入yes并回车确认。随后参照提示选择默认安装路径或手动指定，按压Enter确认（注意，此处仅按压一次Enter键即可，否则后续将默认不在安装过程中初始化Anaconda，此时需要手动初始化）后将自动开始安装过程。当询问是否初始化Anaconda时，输入yes后按压Enter确认即可。

安装完成后再次打开终端会默认多出(base)前缀，这是因为Anaconda在安装过程中自动创建了名为base的虚拟环境，并在启动shell时默认激活该环境，(base)前缀提示用户在当前终端内将使用base环境下的Python解释器。执行以下指令即可关闭base环境的默认激活：

```bash
conda config --set auto_activate_base false
```

输入该指令后重启shell即可。为加速Conda的下载速度，需要更换Anaconda软件源，执行以下指令完成更换：

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2    
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/simpleitk  
```

在终端中输入如下指令即可查看Conda的基本信息，其中的channel URLs即包括刚刚增加的软件源。

```bash
conda info
```

## A.3 Pip配置

本书使用Python自带的Pip工具管理各Conda环境下的Python库。

执行以下指令安装Vim文本编辑器：

```bash
sudo apt install vim  
```

为加快pip下载速度，首先需要更换Pip工具的下载源，流程如下：

```bash
cd ~
mkdir .pip
cd .pip
touch pip.conf
vim pip.conf  
```

向pip.conf文件输入以下内容：

```bash
[global]
index-url=http://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host=mirrors.aliyun.com  
```

保存pip.conf文件即可完成Pip换源操作。该pip.conf文件的配置作用是全局的，在后续即将创建的各Conda虚拟环境下都将有效。

## A.4 Python虚拟环境配置

首先查看当前已有的Conda虚拟环境，执行以下命令：

```bash
conda env list  
```

执行该命令会返回当前所有的可用Conda环境名及其路径。默认情况下，Conda提供了一个名为base的虚拟环境，该环境中安装了常用的数据科学包。

本书不建议读者直接在base中安装其余需要的包，而是希望读者创建4个独立的虚拟环境，分别命名为stat，tf-cpu，torch-cpu和spark[^4]。以下给出Conda创建环境的基本流程：

使用Conda指令创建虚拟环境时需要为该环境指定特定版本的Python[^5]，执行以下指令创建名为stat环境：

```bash
conda create -n stat python=3.9.13
```

输入y确认将自动开始环境创建，结束后使用以下指令激活stat环境。

```bash
conda activate stat
```

若激活成功，当前终端提示符前出现(stat)前缀。使用以下指令更新Pip版本：

```bash
pip install --upgrade pip
```

在数据科学任务中，通常都会使用到numpy，pandas，matplotlib，scikit-learn，scipy等库，因此，创建虚拟环境完成之后，可以首先使用以下指令安装它们：

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install scipy  
```

在Pip的安装过程中，Pip将自动下载各包所需的依赖项。安装完成后使用以下指令可查看当前虚拟环境下已安装的Python包：

```bash
pip list
```

以下指令用于删除Conda环境。如果当前终端已激活待删除的环境中，则需要首先使用deactivate命令退出该环境。

```bash
conda deactivate
conda remove -n 待删除虚拟环境的名称 --all
```

以下指令用于清理Conda缓存，以帮助缓解硬盘压力。

```bash
conda clean -p  conda  clean -t
```

## A.5 Vscode安装配置

首先前往Vscode官方网站[^6]下载其安装文件，Ubuntu系统下选择下载.deb文件。执行以下指令安装Vscode：

```bash
cd ~/Downloads
sudo dpkg -i code_1.73.1-1667967334_amd64.deb
```

安装完成后在软件列表里即可找到Vscode图标并打开，读者也可以选择在终端内输入以下指令打开Vscode：

```bash
code .
```

该指令将以当前shell所在路径为Vscode默认的工作空间或工作路径。

安装完毕后需要为Python开发安装相应的拓展（Extensions）。点击Vscode左侧的EXTENSIONS选项卡，在搜索栏内输入Python，点击搜索结果内Python拓展右侧的install即可安装。该过程将自动安装其他可能需要的拓展。

上述工作完成后，使用Vscode打开Python脚本文件或Notebook文件即可查看到可用的Python环境。在Python脚本中，点击Vscode窗口右下角Python版本即可弹出Python解释器选择窗口，在其中可以看到Ubuntu内置Python环境以及使用Anaconda创建的虚拟环境。在Notebook文件中，点击文件右上角的Python版本即可弹出Kernel选择窗口，可选择不同Python环境。

在Jupyter Notebook内进行Python程序开发具有一定的优势，它允许使用者单独调试每一部分代码块，并能够将代码、执行结果和Markdown格式的笔记等融合在一个文件中。因此，本书中除部分通用的工具函数和较复杂的类定义使用Python脚本文件外，各案例的主要代码均使用Notebook文件实现。在读者熟练模型基本原理和方法后，可将程序迁移至Python脚本文件，这对较大的项目工程是有帮助的。

[^1]:  https://ubuntu.com/
[^2]: https://www.anaconda.com/
[^3]: bash后即为安装文件的文件名，取决于读者当前下载的版本，该文件名可能与书中此处有差异。
[^4]: 尽管没有必要为每一章的内容单独创建一个虚拟环境。但为尽可能降低冲突可能性，本书依然选择创建数个独立的环境。
[^5]: 由于本书代码中大量使用到了F字符串的新特性f’{expr=}’，以用于打印输出各类信息，请务必使用Python 3.8以上版本的Python发行版，否则需要读者重新编辑使用到f’{expr=}’的代码行。
[^6]: https://code.visualstudio.com/

# 附录B. Spark开发环境配置

## B.1 Java安装配置

安装Apache Spark前需要用户首先安装Java环境。

执行以下指令开始安装openjdk[^7]：

```bash
sudo apt-get install openjdk-8-jdk
```

执行以下指令验证是否安装成功：

```bash
java -version
```

若安装成功则会在终端中输出对应的版本信息：

```bash
openjdk version "1.8.0_352"
OpenJDK Runtime Environment (build 1.8.0_352-8u352-ga-1~22.04-b08)
OpenJDK 64-Bit Server VM (build 25.352-b08, mixed mode)
```

执行以下指令查找java路径：

```bash
sudo update-alternatives --config java
```

若读者系统内无安装其他版本Java，则应当得到以下输出：

```bash
There is only one alternative in link group java (providing /usr/bin/java): /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
Nothing to configure.
```

上述输出中/usr/lib/jvm/java-8-openjdk-amd64即是后续将要设置的环境变量JAVA_HOME的值。

## B.2 Scala和Hadoop安装配置

在本书中为最简化流程，此处不单独安装Scala和Apache Hadoop，而直接从磁盘读取数据文件进行后续建模分析。

## B.3 Spark安装配置

前往Apache Spark官方网站[^8]下载其安装文件。本书中选择3.3.1(Oct 25 2022)发行版，包类型为Pre-build for Apache Hadoop 3.3 and later。文件下载完成后使用如下指令将压缩包解压并转移至目标路径：

```bash
cd ~/Downloads
tar -zxvf spark-3.3.1-bin-hadoop3.tgz
sudo mv spark-3.3.1-bin-hadoop3 /usr/local 
```

此时已将Spark相关文件全部放置在/usr/local/ spark-3.3.1-bin-hadoop3路径下，为确保可以在任意位置使用Spark-shell，需要进行如下操作：

```bash
vim ~/.bashrc
```

在该文件中追加以下内容：

```bash
export JAVA_HOME='/usr/lib/jvm/java-8-openjdk-amd64'
export SPARK_HOME='/usr/local/spark-3.3.1-bin-hadoop3'
export PATH=$PATH:$SPARK_HOME/bin
```

保存文件回到终端，输入以下内容执行更新：

```bash
source ~/.bashrc
```

在终端中输入如下指令即可启动Scala语言的Spark交互环境：

```bash
spark-shell
```

按压Ctrl+D可退出上述交互式环境。

若无异常则终端会给出如下输出：

```bash
22/11/21 10:56:27 WARN Utils: Your hostname, yeli-virtual-machine resolves to a loopback address: 127.0.1.1; using 192.168.88.128 instead (on interface ens33)
22/11/21 10:56:27 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
22/11/21 10:56:35 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Spark context Web UI available at http://192.168.88.128:4040
Spark context available as 'sc' (master = local[*], app id = local-1668999396860).
Spark session available as 'spark'.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 3.3.1
      /_/
         
Using Scala version 2.12.15 (OpenJDK 64-Bit Server VM, Java 1.8.0_352)
Type in expressions to have them evaluated.
Type :help for more information.
```

由于具体环境差异，读者的输出通常会与上述不同。

若要启动Python语言的Spark交互环境，则输入以下指令：

```bash
pyspark
```

若无异常则终端会给出如下输出：

```bash
Python 3.10.6 (main, Nov  2 2022, 18:53:38) [GCC 11.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
22/11/21 11:02:45 WARN Utils: Your hostname, yeli-virtual-machine resolves to a loopback address: 127.0.1.1; using 192.168.88.128 instead (on interface ens33)
22/11/21 11:02:45 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
22/11/21 11:02:46 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 3.3.1
      /_/

Using Python version 3.10.6 (main, Nov  2 2022 18:53:38)
Spark context Web UI available at http://192.168.88.128:4040
Spark context available as 'sc' (master = local[*], app id = local-1668999768089).
SparkSession available as 'spark'.
```

至此，基本的Spark环境配置完成。

## B.4 PySpark安装配置

Spark支持Scala，Java，Python，R等编程语言，为在Conda虚拟环境中也使用Python语言操作Spark，需在在spark环境内单独安装Pyspark包。Pyspark可直接通过Pip指令安装。使用Pip指令前请先激活相应的Conda环境。

```bash
conda activate spark
pip install pyspark  
```

安装完毕后可执行以下代码以验证PySpark是否安装成功。

```python
# appx/appx.ipynb
# PySpark验证
from pyspark.sql import SparkSession
spark = SparkSession\
    .builder\
    .master('local[*]')\
    .appName('Time Series Forecasting')\
    .getOrCreate()
spark.stop()
```

若执行上述代码无异常，则证明Conda环境中的PySpark安装配置成功。

[^7]: 更新版本的openjdk应当也是受支持的。
[^8]: https://spark.apache.org/

# 附录C. 项目工程结构

本书中的全部数据和代码程序（演示程序、自定义模块和章节案例代码）的项目工程的目录层级如下所示：

```bash
.
├── appx
│   └── appx.ipynb
├── ch1
│   └── ch1.ipynb
├── ch2
│   ├── ch2_1_grid_sarma
│   │   ├── data
│   │   ├── fig
│   │   └── grid_sarima.ipynb
│   └── ch2_2_auto_sarima
│       ├── auto_sarima.ipynb
│       ├── data
│       └── fig
├── ch3
│   ├── ch3_1_knn
│   │   ├── fig
│   │   └── knn.ipynb
│   ├── ch3_2_mlr
│   │   ├── fig
│   │   └── mlr.ipynb
│   ├── ch3_3_svr
│   │   ├── fig
│   │   └── svr.ipynb
│   ├── ch3_4_dt
│   │   ├── dt.ipynb
│   │   └── fig
│   ├── ch3_5_ensemble
│   │   ├── ch3_5_1_rf
│   │   ├── ch3_5_2_gbrt
│   │   ├── ch3_5_3_xgboost
│   │   └── ch3_5_4_lightgbm
│   ├── ch3_6_spark
│   │   ├── fig
│   │   └── spark_mllib.ipynb
│   └── data
│       ├── data.ipynb
│       ├── data_pm2_5.csv
│       └── fig
├── ch4
│   ├── ch4_1_fnn
│   │   ├── data
│   │   ├── fig
│   │   └── fnn.ipynb
│   ├── ch4_2_rnn
│   │   ├── data
│   │   ├── fig
│   │   ├── gru.ipynb
│   │   └── lstm.ipynb
│   ├── ch4_3_cnn
│   │   ├── cnn.ipynb
│   │   ├── data
│   │   ├── fig
│   │   └── tcn.ipynb
│   ├── ch4_4_gnn
│   │   ├── data
│   │   ├── dataset_loader.py
│   │   ├── fig
│   │   ├── gat.ipynb
│   │   ├── gcn.ipynb
│   │   ├── model
│   │   └── model.py
│   └── ch4_5_att
│       ├── att.ipynb
│       ├── data
│       ├── dataset.py
│       ├── fig
│       ├── model
│       └── model.py
└── utils
    ├── dataset.py
    ├── dataset_spark.py
    ├── metrics.py
    ├── metrics_spark.py
    └── plot.py

46 directories, 28 files
```

其中appx文件夹用于存储附录中使用到的测试代码；ch1至ch4文件夹用于存储各章节中的时间序列数据（data），预测模型代码（Notebook文件和Python源文件），以及图片（fig），部分model文件夹用于存储训练完成的模型文件；utils文件夹用于存储自定义模块和工具函数等。

# 附录D. 缩写列表

| **缩写** | **中文全称**                 |
| -------- | ---------------------------- |
| ACF      | 自相关函数                   |
| ADF      | 增广迪基-富勒                |
| AIC      | 赤池信息准则                 |
| AR       | 自回归                       |
| ARCH     | 自回归条件异方差             |
| ARIMA    | 差分整合自回归滑动平均       |
| ARMA     | 自回归滑动平均               |
| BIC      | 贝叶斯信息准则               |
| BPNN     | 反向传播神经网络             |
| CART     | 分类与回归树                 |
| CNN      | 卷积神经网络                 |
| CV       | 交叉验证                     |
| CSV      | 逗号分隔值                   |
| DCL      | 密集连接层                   |
| DNN      | 深度神经网                   |
| DT       | 决策树                       |
| EFB      | 互斥特征捆绑                 |
| FCL      | 全连接层                     |
| FCNN     | 全连接神经网络               |
| FNN      | 前馈神经网络                 |
| GAL      | 图注意力层                   |
| GARCH    | 广义自回归条件异方差         |
| GAT      | 图注意力网络                 |
| GB       | 梯度提升                     |
| GBDT     | 梯度提升决策树               |
| GBRT     | 梯度提升回归树               |
| GCL      | 图卷积层                     |
| GCN      | 图卷积网络                   |
| GNN      | 图神经网络                   |
| GOSS     | 基于梯度的单边采样           |
| GRU      | 门控循环单元                 |
| ISP      | 网络服务提供商               |
| KNN      | K最近邻                      |
| LightGBM | 轻量梯度提升机               |
| LSTM     | 长短期记忆                   |
| MA       | 滑动平均                     |
| MAE      | 平均绝对误差                 |
| MAPE     | 平均绝对百分比误差           |
| MLP      | 多层感知机                   |
| MLR      | 多元线性回归                 |
| MSA      | 多头自注意力                 |
| MSE      | 均方误差                     |
| NLP      | 自然语言处理                 |
| PACF     | 偏自相关函数                 |
| PCC      | 皮尔逊相关系数               |
| PEP      | Python增强提案               |
| RF       | 随机森林                     |
| RMSE     | 均方根误差                   |
| RNN      | 循环神经网络                 |
| SARIMA   | 季节性差分整合自回归滑动平均 |
| SCADA    | 数据采集与监视控制           |
| SDE      | 误差标准差                   |
| SVM      | 支持向量机                   |
| SVR      | 支持向量回归                 |
| TCN      | 时间卷积网络                 |
| XGBoost  | 极度梯度提升                 |

