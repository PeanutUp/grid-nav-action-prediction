# 基于机器学习的网格导航动作预测实验

这个项目把网格导航问题做成监督分类任务：给定当前位置、目标位置和局部障碍物信息，预测下一步动作。

## 目录

```text
scripts/
  make_data.py       生成并检查数据集 A/B
  train_eval.py      验证集调参、训练模型、单步动作预测评价
  rollout_eval.py    把模型放回地图里闭环导航
  ablation.py        所有学习算法的完整特征消融

src/gridnav/
  env.py             网格动作、合法位置判断
  search.py          BFS 专家路径
  features.py        16 维状态特征
  dataset.py         地图和样本生成
  models.py          模型候选参数、贪心基线
```

## 运行顺序

先生成数据：

```bash
PYTHONPATH=src python scripts/make_data.py --quick
```

确认能跑通后，再生成正式数据：

```bash
PYTHONPATH=src python scripts/make_data.py
```

训练并做单步评价：

```bash
PYTHONPATH=src python scripts/train_eval.py
```

闭环导航评价：

```bash
PYTHONPATH=src python scripts/rollout_eval.py
```

特征消融：

```bash
PYTHONPATH=src python scripts/ablation.py
```

结果会保存在：

```text
outputs/metrics/single_step.csv       单步动作预测结果
outputs/metrics/tuning.csv            各算法验证集调参记录
outputs/metrics/rollout.csv           闭环导航结果
outputs/metrics/ablation.csv          完整特征消融结果
outputs/metrics/ablation_tuning.csv   消融实验里的调参记录
outputs/models/                       训练好的模型
outputs/figures/                      混淆矩阵
```

## 实验内容

- 数据集 A：10x10 随机网格，障碍物比例约 0.10-0.30。
- 数据集 B：15x15/20x20 混合难度地图，包含随机障碍和窄通道地图，障碍物比例约 0.20-0.40。
- 模型：KNN、距离加权 KNN、GaussianNB、逻辑回归、决策树、K-means 多数标签、Bagging、随机森林、AdaBoost、Stacking。
- 单步评价：Accuracy、Macro-F1、四类动作 Precision/Recall、混淆矩阵、训练时间、预测时间、模型大小。
- 闭环评价：Success Rate、Average Steps、Path Ratio、Collision Rate、Timeout Rate。
- 特征消融：所有 10 个学习算法。
