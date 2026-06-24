# Paper 1 研究简报

## 题目暂定

**超越平均不确定性：面向可靠医学图像分割的空间失败区域建模**

英文暂定：

**Beyond Average Uncertainty: Spatial Failure-Region Modeling for Reliable Medical Image Segmentation**

核心方法名：

**SFRM: Spatial Failure-Region Modeling**

## 一、研究定位

本研究不以提出新的分割主干网络为目标，也不把论文写成一个单纯追求 AUC 的质量预测器论文。Paper 1 的定位是：

**建立一种医学图像分割可靠性审计的新表征框架，将模型失败从单一全局分数转化为空间结构化、可解释、可审计的失败区域描述。**

现有医学分割可靠性研究常把不确定性图压缩为 mean entropy、max entropy 或 case-level confidence。这种做法会丢失一个关键事实：医学图像中的错误不是均匀发生的，而是集中在边界模糊、结构粘连、拓扑异常、低对比度界面和混淆区域。

因此，本研究的基本立场是：

**医学分割可靠性不应只问“模型整体有多不确定”，而应问“风险出现在什么空间区域、对应什么医学失败模式、是否值得医生优先复核”。**

## 二、核心科学问题

本研究要回答的核心科学问题是：

**医学图像分割模型的失败是否具有可被结构化描述的空间几何与特征拓扑模式？这些空间模式在表达和发现模型失败时，是否比传统全局不确定性更具可解释性、可分性和临床复核价值？**

具体拆成三个问题：

1. **可分性问题**  
   SFRM 提取的空间失败区域特征，在好预测和坏预测之间是否存在稳定的统计学差异？

2. **定位问题**  
   SFRM 是否能比简单的高熵区域更准确地定位真正的错误区域、边界错误或对象级错误？

3. **临床复核价值问题**  
   在医生只能复核少量病例或区域时，SFRM 风险排序是否比全局不确定性更能捕获 critical segmentation failures？

## 三、核心贡献设计

### 贡献 1：SFRM 框架

提出空间失败区域建模框架，将分割预测中的高风险区域分解为可解释的失败区域家族：

- Boundary-risk region；
- Uncertainty-cluster region；
- Topology-risk region；
- Anatomical/topological consistency region；
- Feature-ambiguity region；
- 可选 Image-quality/artifact-risk region。

这些区域均要求在部署时不依赖 ground truth，只能由输入图像、预测概率图、预测掩码、不确定性图和可选冻结特征图计算得到。

### 贡献 2：Feature-Discrimination Audit

在训练复杂预测器之前，系统审计各类 SFRM 特征是否真的包含可靠性信息。

审计内容包括：

- 好预测与坏预测之间的特征分布差异；
- 单变量 AUROC/AUPRC；
- Spearman 相关性；
- Mann-Whitney U 检验；
- 特征相关矩阵与共线性分析；
- Lasso 或稀疏逻辑回归进行特征选择；
- 各失败区域家族的消融分析。

### 贡献 3：Human-Review Budget Simulation

模拟临床医生只能复核 5%、10%、20% AI 输出的场景，比较不同复核排序策略：

- random review；
- mean entropy；
- max entropy；
- foreground entropy；
- TTA disagreement；
- SFRM risk score。

核心评价指标：

- fixed-budget critical-error recall；
- accepted bad-case reduction；
- risk-coverage curve；
- boundary/object failure capture rate。

### 辅助贡献：轻量可靠性预测器

预测器不是论文主角，只作为验证工具。

优先使用：

- logistic regression；
- Lasso logistic regression；
- random forest；
- gradient boosting；
- calibrated logistic regression。

研究重点不是证明某个复杂预测器最强，而是证明：

**SFRM 描述符本身包含比全局不确定性更强的可部署可靠性信号。**

## 四、技术路线

整体技术路线如下：

```text
医学图像 X
   |
   v
已有分割模型 / 轻量 U-Net / nnU-Net
   |
   v
预测概率图 P + 预测掩码 M
   |
   +--> 不确定性图 U
   |       - entropy
   |       - margin uncertainty
   |       - TTA disagreement optional
   |
   +--> SFRM 空间失败区域挖掘
           - boundary-risk
           - uncertainty-cluster
           - topology-risk
           - anatomical/topological consistency
           - feature-ambiguity optional
   |
   v
Leakage-free deployable feature table
   |
   +--> Feature-discrimination audit
   |
   +--> Lightweight reliability predictor
   |
   +--> Human-review budget simulation
   |
   v
病例级可靠性评估 + 区域级失败定位 + 灰色区域分析
```

## 五、方法模块

### 1. Boundary-risk

目标：捕捉边界模糊、边界泄漏和轮廓不稳定。

候选特征：

- predicted boundary band area；
- boundary mean entropy；
- boundary max entropy；
- boundary margin uncertainty；
- boundary high-entropy ratio；
- boundary probability-gradient statistics。

注意：Boundary-risk 是 MoNuSeg 中最容易产生歧义的失败类型，因为 2-3 像素的边界滑移可能同时来自模型错误、标注者差异或真实组织边界模糊。

### 2. Uncertainty-cluster

目标：从像素级不确定性转向空间连通的不确定区域。

候选特征：

- high-entropy area fraction；
- high-entropy connected component count；
- largest high-uncertainty component area；
- uncertainty cluster compactness；
- uncertainty cluster boundary contact ratio。

核心思想：不是看单个高熵像素，而是看不确定性是否形成结构化区域。

### 3. Topology-risk

目标：捕捉更客观的结构性失败。

候选特征：

- connected component count；
- small island count；
- hole count / Euler number；
- largest component ratio；
- component area coefficient of variation；
- thin bridge / fragmentation surrogate；
- opening / closing residual。

Topology-risk 在密集细胞核分割中属于高置信失败信号，因为粘连、断裂、孔洞、小岛等错误更容易通过离散拓扑量化。

### 4. Anatomical / Topological Consistency

目标：检查预测结构是否符合基本医学或形态先验。

在 MoNuSeg 阶段主要使用轻量形态一致性：

- mild opening residual；
- mild closing residual；
- threshold perturbation stability；
- component count stability under thresholds 0.45 / 0.50 / 0.55；
- Euler number stability。

后续扩展到器官或病灶数据集时，可加入更明确的解剖位置、一致性和形态约束。

### 5. Feature-ambiguity

目标：捕捉“像目标但不稳定”的混淆区域。

可选输入：

- DINOv2；
- UNI；
- SAM image encoder；
- 其他冻结视觉特征。

Paper 1 初期可以先不强依赖该模块，避免把第一篇拖成 foundation feature 论文。若 MoNuSeg/CoNSeP 的基础 SFRM 特征已有效，再加入 feature-ambiguity 作为增强模块。

## 六、严格无泄漏设计

本研究必须严格区分：

### Deployable features

部署时可以计算，可进入可靠性预测器：

- 输入图像特征；
- 预测概率图特征；
- 预测掩码几何特征；
- 不确定性图特征；
- 预测结构拓扑特征；
- 可选冻结特征图统计。

### Evaluation metrics

只能用于审计、训练标签和评价，不能进入预测器输入：

- Dice；
- Boundary Dice；
- HD95；
- AJI / PQ；
- FP / FN region；
- error region mask；
- LECR。

缓存文件必须强制分离：

```python
cache_entry = {
    "patch_id": "...",
    "deployable_features": {},
    "evaluation_metrics": {},
}
```

CSV 中使用：

- `feat__...` 表示部署特征；
- `eval__...` 表示评价指标。

## 七、LECR 灰色区域诊断

LECR，即 Local Error Contribution Ratio，是审计指标，不是部署特征。

定义：

```text
Omega_error = prediction XOR ground_truth
```

第一阶段使用两个版本：

- `lecr_uncertainty`：错误区域内不确定性质量 / 全图不确定性质量；
- `lecr_boundary_error`：边界带内错误像素 / 全部错误像素。

用途：

- 识别高 Dice 但局部风险高的灰色区域；
- 区分边界模糊导致的 Dice 波动和真正结构性失败；
- 为 Figure C 的四象限分析提供诊断依据。

## 八、拟做实验

### Experiment 0：特征区分度审计

目的：验证 SFRM 特征是否在好预测和坏预测之间有统计学差异。

数据：

- 第一阶段：MoNuSeg test patches；
- 第二阶段：CoNSeP；
- 第三阶段：一个非病理数据集，如 BraTS/FeTS 或 MSD。

分析：

- good vs bad feature distribution；
- UMAP / t-SNE；
- Mann-Whitney U test；
- univariate AUROC；
- Spearman correlation；
- feature correlation matrix；
- Lasso feature selection。

停止标准：

- 如果 SFRM 特征不优于 mean entropy，不进入完整预测器；
- 如果提升只来自 GT 派生特征，立即停止并修正；
- 如果没有稳定贡献的特征家族，不进入完整论文阶段。

### Experiment 1：轻量可靠性预测器

目的：验证 SFRM 特征能否支持可解释的病例级失败预测。

模型：

- logistic regression；
- Lasso logistic regression；
- random forest；
- gradient boosting。

评价：

- AUROC；
- AUPRC；
- calibration curve；
- ECE；
- risk-coverage curve。

### Experiment 2：区域级失败定位

目的：评估 SFRM 是否能定位真实错误区域。

对比：

- top entropy pixels；
- high-entropy connected components；
- boundary-risk region；
- topology-risk region；
- full SFRM region。

指标：

- error-region Dice；
- region IoU；
- boundary-error recall；
- top-k failure-region hit rate；
- precision at fixed review budget。

### Experiment 3：人机复核预算模拟

目的：模拟有限临床审核资源下，SFRM 是否优于全局不确定性排序。

预算：

- top 5%；
- top 10%；
- top 20%。

对比：

- random；
- mean entropy；
- max entropy；
- foreground entropy；
- SFRM risk。

指标：

- critical-error recall；
- accepted bad-case reduction；
- selective risk-coverage curve。

### Experiment 4：特征家族消融

目的：证明每类失败区域的贡献。

消融：

- no boundary-risk；
- no topology-risk；
- no uncertainty-cluster；
- no anatomical/topological consistency；
- no feature-ambiguity。

### Experiment 5：跨数据集泛化

目的：证明 SFRM 不是只对 MoNuSeg 有效。

路线：

1. MoNuSeg 验证基本可行性；
2. CoNSeP 验证密集病理对象级失败；
3. 非病理数据集验证更一般的医学分割可靠性。

## 九、图表计划

### Figure 1：SFRM 技术路线图

展示从医学图像、分割输出、不确定性图到空间失败区域、可靠性评分和复核排序的完整流程。

### Figure 2：失败区域类型示意图

展示：

- boundary-risk；
- uncertainty-cluster；
- topology-risk；
- anatomical consistency；
- feature-ambiguity。

### Figure 3：Feature Separability

UMAP/t-SNE 展示 SFRM 特征空间中好预测与坏预测的分布关系。

### Figure 4：Feature Impact Radar

展示不同特征家族相对 mean entropy 的判别力。

### Figure 5：Gray-Zone Diagnostic Matrix

四象限：

- high Dice / low SFRM risk；
- high Dice / high SFRM risk；
- low Dice / low SFRM risk；
- low Dice / high SFRM risk。

重点展示高 Dice / 高 SFRM risk 的病例，证明 Dice 掩盖了局部风险。

### Figure 6：Review-Budget Curve

展示固定复核预算下，SFRM 是否比全局不确定性捕获更多 critical errors。

## 十、预期论文结论

如果实验成立，Paper 1 的核心结论应是：

**医学图像分割失败具有可量化的空间结构。与把不确定性压缩为全局平均值相比，SFRM 能更有效地表征、解释和定位模型失败，并在有限人工复核预算下捕获更多临床相关错误。**

更克制的英文表述：

> Structured failure-region descriptors provide more informative and interpretable reliability evidence than global uncertainty aggregation for medical image segmentation.

## 十一、风险与应对

### 风险 1：SFRM 特征与 mean entropy 差距不明显

应对：

- 检查失败标签是否过粗；
- 转向 boundary/object-level 失败；
- 加强 topology 和 gray-zone 分析；
- 不急于训练复杂 predictor。

### 风险 2：Boundary-risk 受标注者差异影响

应对：

- 把 Boundary-risk 定位为敏感但需校准的风险信号；
- 用 LECR 分析灰色区域；
- Discussion 中明确 GT 不是绝对真理。

### 风险 3：只在病理数据上有效

应对：

- CoNSeP 后加入一个非病理数据集；
- 把 MoNuSeg 定位为机制验证，不作为唯一泛化证据。

### 风险 4：预测器被审稿人要求更复杂

应对：

- 明确 predictor 是 empirical vehicle；
- 主贡献是 framework 和 feature audit；
- 使用简单可解释模型反而支持“信息来自特征表征”的论点。

## 十二、当前执行建议

第一步不训练复杂可靠性预测器。

推荐执行顺序：

1. 完成 MoNuSeg 50-100 patch 的 feature-discrimination audit；
2. 确认至少两类 SFRM 特征优于 global entropy；
3. 扩展到完整 MoNuSeg test split；
4. 复现到 CoNSeP；
5. 再训练轻量 reliability predictor；
6. 最后做人机复核预算模拟和跨数据集验证。

这一路线的关键是：

**先证明失败区域表征本身有效，再谈预测器和系统集成。**

