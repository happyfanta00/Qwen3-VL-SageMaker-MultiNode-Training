# 基于SageMaker Training Job的Qwen3-VL多机多卡训练

## 1. 数据准备

数据格式请参考[Qwen3-VL官方repo](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)，数据集定义位于[src/qwenvl/data/__init__.py](src/qwenvl/data/__init__.py)文件中，本代码库提供了[COCO Caption](https://huggingface.co/datasets/lmms-lab/COCO-Caption2017)作为训练数据的样例，使用以下脚本下载并构建训练集：

```
pip install datasets pillow

python download_coco_caption.py
```

## 2. 模型训练

模型训练流程见[train.ipynb](train.ipynb)文件

训练代码基于[Qwen3-VL官方repo](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)，为了适配SageMaker Training Job，主要修改点如下：

* [src/requirements.txt](src/requirements.txt): 定义了同时适配SageMaker Framework Containers和Qwen3-VL训练的python环境

* [src/entry.py](src/scripts/sft_qwen3_4b.sh):定义了其他环境变量和训练脚本入口

* [src/scripts/sft_qwen3_4b.sh](src/scripts/sft_qwen3_4b.sh): 修改以适配SageMaker Training Job的容器环境

## 3. 训练时长参考

下表展示了使用上述样例数据，#epoch=10的训练时长

注意：该数据仅用于帮助验证训练流程是否符合预期，不作为性能benchmark

| Instance Type | Instance Number | Global Batch Size |  Time |
| ------------- | --------------- |------------------ |-------|
| g6e.48xlarge  |  1              | 128 | 25:05 |
| g6e.48xlarge  |  2              | 128 | 13:32 |
| g6e.48xlarge  |  4              | 256 | 7:21 |
| p5.48xlarge   |  1              | 256 | N/A |
