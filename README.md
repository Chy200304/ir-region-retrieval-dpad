# 局部证据与全图上下文的跨图区域检索

信息检索原理课程 Project。借鉴 CVPR 2026 DPAD 的局部区域/全图语义对照，检验它能否直接用作跨图区域检索分数。这是独立的冻结 CLIP 检索实验，不是 DPAD 强化学习训练或分割指标复现。

## 真实结果

CPU 上使用 CLIP ViT-B/32，编码 COCO128 的 128 张图片和 852 个有效标注框；71 个实际出现的类别各产生一个英文查询。同类框为相关项。没有训练或数据驱动调参，权重固定为 0.25、0.5、1.0。

|方法|Hit@1|Hit@10|Recall@10|AP@10|MRR|
|---|---:|---:|---:|---:|---:|
|整图余弦|0.3239|0.8028|0.3550|0.2466|0.4816|
|裁剪余弦|0.5915|0.9014|0.4345|0.3783|0.7098|
|裁剪 - 0.25×整图|0.4225|0.8451|0.3130|0.2381|0.5422|
|裁剪 - 0.5×整图|0.1831|0.5211|0.1400|0.0849|0.2926|
|裁剪 - 1.0×整图|0.0704|0.2254|0.0304|0.0216|0.1144|

**这是负结果：所有扣分版本都低于裁剪基线。** 全图包含目标自身，将其作为背景惩罚可能损伤真正的相关性；同一图像内减去相同常数不会改变区域排序。不能据此否定 DPAD 的训练方法，也没有提出新的最佳检索算法。

## 1. 复核指标，无需下载模型或图片

需要 Python 3.12。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-offline.txt
.venv/bin/python retrieval.py evaluate
```

仓库保留归一化特征与元数据；评测只需 numpy。首次安装依赖需要网络，之后评测可离线运行。

## 2. 本地演示，先准备图片

仓库不公开重分发原图和缩略图。以下命令下载约 6.7 MB 的 COCO128 压缩包，验证其 SHA256，并按已有框生成本机缩略图；此步不需要下载 CLIP。

```bash
.venv/bin/python -m pip install -r requirements-demo.txt
.venv/bin/python prepare_data.py
.venv/bin/python retrieval.py demo
```

打开终端显示的网址，默认 http://127.0.0.1:18765。选择类别，比较两种得分的前十结果。只支持预编码类别列表，不支持任意自由文本。停止服务按 Control+C。macOS 的 `启动演示.command` 也可启动服务，但须先完成图片准备。

## 3. 从原图重新提取 CLIP 特征

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prepare_data.py
.venv/bin/python retrieval.py build --dataset data/coco128 --cache model_cache
.venv/bin/python retrieval.py evaluate
```

首次下载模型需要网络；CPU 可运行，不需要收费 API。代码固定模型 revision 与原始权重格式。`requirements-lock.txt` 记录构建环境；模型、虚拟环境和下载数据均被 Git 忽略。

## 评价协议

英文查询为 `a photo of a {category}`；宽或高不足8像素的标注框被剔除。L2归一化后，`s(q,r)=cos(q,r)-lambda*cos(q,parent_image)`；按得分稳定降序排序。

Hit@k 表示前k至少有一个相关项；Recall@10 是前十相关数/总相关数；AP@10 是前十相关位置的精确率之和除以 `min(10, 总相关数)`；MRR 是首个相关项排名倒数。先逐查询计算，再按类别宏平均。

候选是理想标注框，查询是简单类别模板；不评价自动检测、复杂自然语言推理或无答案查询。COCO128 来自 COCO train2017，无法排除 CLIP 预训练重叠。因此这是小规模排序诊断，不是未见测试集上的泛化声明。整图基线会为同图区域产生并列分数，应作为上下文消融理解。

## 目录与验证

- `retrieval.py`：特征构建、评测、本地演示。
- `prepare_data.py`：校验公开数据并生成本地缩略图。
- `data/features.npz`、`metadata.json`：复核实验的向量与候选映射。
- `results/metrics.json`、`per_query.json`：汇总和全部逐查询指标。
- `tests/`：指标、排序性质、数据快照与缩略图准备测试。
- `DATA_SOURCES.md`、`data/UPSTREAM_LICENSE.txt`：来源、引用及上游许可。

图片准备完毕后运行：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile retrieval.py prepare_data.py
zsh -n 启动演示.command
```

Paper Reading 和 Project 的6页 PDF 报告由课程作者另行提交，未包含在本公开代码仓库中。代码和报告有助手参与生成，使用者应独立理解、核对并遵守课程要求。

## 参考

- Tao Yang, Qing Zhou, Yanliang Li, Qi Wang. **Discriminative Perception via Anchored Description for Reasoning Segmentation**, CVPR 2026. [Paper](https://arxiv.org/abs/2603.04002) · [Authors' code](https://github.com/mrazhou/DPAD).
- Alec Radford et al. **Learning Transferable Visual Models From Natural Language Supervision**, ICML 2021. [Paper](https://proceedings.mlr.press/v139/radford21a.html).
- [COCO128 data mirror](https://huggingface.co/datasets/LibreYOLO/coco128) · [CLIP model](https://huggingface.co/openai/clip-vit-base-patch32).
