# time-series-analysis-and-python-practices

《时间序列分析与Python实例》中南大学出版社 配套代码

## 安装 Install

1. 在终端执行以下指令：

   ```bash
   git clone https://github.com/flywithliye/time-series-analysis-and-python-practices.git
   ```

   或点击 `Code`-`>Download ZIP`下载项目工程的压缩包文件

   若您的网络连接条件异常，可使用百度网盘[链接](https://pan.baidu.com/s/1OeBNIHFi3GmdPWZkn62cDQ?pwd=1vyk)下载
2. 切换至项目工程 `time-series-analysis-and-python-practices`

   ```bash
   cd time-series-analysis-and-python-practices
   ```
3. 在项目工程打开IDE（Visual Studio Code）

   ```bash
   code .
   ```

> 由于各功能包和模块相互间的调用关系，请在运行本项目时，将 `time-series-analysis-and-python-practices`作为IDE的工作空间路径。

## 环境 Environments

1. 本书配套项目使用Anaconde进行Python虚拟环境管理。
2. 建议在有GPU的硬件环境下运行本书第四章中的深度学习模型。
3. 本书配套项目中大量使用**f-字符串**，请确保python版本大于3.6。
4. 第二章使用的主要统计学习框架版本：

   ```bash
   statsmodels>=0.13.5
   pmdarima>=2.0.2
   ```
5. 第三章使用的主要机器学习框架版本：

   ```bash
   scikit-learn>=0.24.1
   xgboost>=1.7.1
   lightgbm>=3.3.3
   pyspark>=3.2.0
   ```
6. 第四章使用的主要深度学习框架版本：

   ```
   tensorflow>=2.6.0
   keras-tcn>=3.5.0
   pytorch>=1.9.1
   torch-geometric>=2.0.4
   torch-geometric-temporal>=0.52.0
   ```

   > 为最大限度降低冲突的可能性，建议分别为pytorch和keras构建独立的虚拟环境。
   >

## 附录 Appendix

为便于读者使用，本书的附录部分以Markdown格式文本提供，文 `appendix.md`件位于本仓库根路径下。

## 其他 Others

- 若对本书及其配套项目工程有任何疑问，可通过Github-Issues功能提问，或以电子邮件的形式联系原作者。
- 本项目工程中的代码可能同印刷版书籍中呈现的代码不完全一致，若有冲突，请以本仓库中实际代码为准。

## 引用 Citation

如果本书的内容对您有帮助，请考虑在您的工作中进行引用。
