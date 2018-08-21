# 项目说明

## 一、目录说明

整个项目由比赛目录、通用工具目录组成

### 比赛目录
kaggle项目下面是每一个具体的比赛，每个比赛一个文件夹，每个文件夹下面一个code文件，一个data文件，data文件里面是所有的数据，data文件下面又细分两个文件夹，一个是origin_data和processed_data，origin_data表示的是从kaggle直接下载的数据，processed_data表示的是经过处理之后的数据。

### 通用工具目录
目录名为util,里面是各个比赛都可以使用的通用工具

## 二、git ignore说明

- 目前所有的origin_data目录下面的数据不同步，这个表示的是kaggle上面可以直接下载的数据
- 所有test开头的文件不同步，这个表示的内部测试代码和内部测试文件，所以需要同步的代码不要用test命名或者test作为文件夹名字开头