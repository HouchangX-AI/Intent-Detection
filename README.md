## Intent-Detection  
  
### 0. 说明  
对话机器人的 意图识别/意图分类 模块。baseline采用fastText，改进模型采用CNN。  
  
### 1. 数据  
| 文件 | 说明 | 行数 |  
|--|--|--|  
| intentions_train.txt | 训练集 | 3926128 |  
| intentions_test.txt | 测试集 | 981658 |  
| tf_idf_all_csv.txt | 养生数据tf-idf | 1500 |  
|  tf_idf_all_tsv.txt | 对话数据tf-idf | 1000 |  
| stop_word_HIT.txt | 停用词-哈工大 | 767 |  
| stop_word_baidu.txt | 停用词-百度 | 1396 |  
| stop_word_chinese.txt | 停用词-中文 | 746 |  
| stop_word_sichuan.txt | 停用词-川大 | 976 |  
  
### 2. 模型  
#### 2.1 模型及进度  
| 模型 | 进度 |  
|--|--|  
| fastText | 完成 |  
| 基于pytorch的fastText | 待定 |  
| 基于pytorch的CNN | 正在写 |  
  
#### 2.2 模型结构图  
**2.2.1 基于pytorch的CNN**  
  
![avatar](https://i.niupic.com/images/2020/01/19/6l6i.png)  
  
### 3. 实验结果（Test set 评估结果）  
**3.1 模型fastText(Size: 800MB)**  
  
| Label | __label__0 | __label__1 |  
| -- | -- | -- |     
| Precision | 0.9778   | 0.9917 |  
| Recall | 0.9168 | 0.9979 |  
| F1-score | 0.9463 | 0.9948 |  
| F1-score | 0.9705 | -- |  
| Accuracy  | 0.9905 | -- |  
  
**3.2 基于pytorch的fastText**  
  
**3.3 基于pytorch的CNN**