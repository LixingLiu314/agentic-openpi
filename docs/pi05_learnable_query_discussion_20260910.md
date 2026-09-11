# Learnable query 预测 subtask：针对当前实现的讨论

状态：设计讨论，未实现、未增加或替换当前已确认的两组实验。

这个方向可行。潜在收益是为subtask建立一组专门聚合任务、物体和阶段信息的表示。不能推导为“多经过几层就一定更准确”：当前S输入本身已经是B最后一层输出，图像/任务特征已通过B多层融合；S自身还做四层cross-attention。要检验的是专用query逐层聚合是否改善了关键信息的读取，而非原来完全没有深层特征。

## 两种不同方案

1. 在B最终特征之后给S加入learnable queries，再通过S或小型query decoder读取图像/文字。这属于Q-Former类聚合方式，改动小，CE仍只训练S，但queries没有穿过B的每层。[BLIP-2](https://arxiv.org/html/2301.12597v3)的learnable queries在独立Q-Former里与冻结图像特征交互；不能把它称为向整个VLM每层插query。
2. 在B多模态语言骨干的输入嵌入处增加4–8个连续向量，例如2048维，在各层更新query，最后取query位置特征作为subtask表示。更接近用户本次想法。它们经过B的语言/多模态层，不会自动经过前面的SigLIP视觉编码器，也不经过action expert。输入空间可学习token且保留预训练骨干的可行性可参考[Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)，但其视觉分类结果不能当作本项目的VLA效果保证。

## 我建议的最小版本

输入：三路图像特征、全局任务/状态token、少量learnable queries。查询初始向量可以结合上一时刻递归S记忆；保留会话重置和现有因果递归，避免相似当前图像丢失历史阶段信息。先保持8个通用query，不给每个query写死eggplant、potato或特定阶段。

在B每层使用非对称attention：原图像/任务token按原规则互相读取，不读取新增query；query可读取原token和query。这样原token特征与A所需的普通prefix不受query反向影响。最后query的特征经投影进入S，生成现有subtask文本；A继续使用原始观测＋生成文字重建普通prefix，不接query隐状态。

一枚query经过词表linear head只得到一个token分布，并不会直接变成完整subtask句子。如果闭集分类后映射文本，可用一个query，但类别固定不利于其他五类任务。优先保留现有自回归文本S，用4–8个query表示供它解码；也可用EOS终止的多位置输出方案，但需要另行处理长度和词间关系。

## 梯度实现是关键

Learnable query若位于B输入，CE必须能沿B中query的计算传回query参数。当前S分支在B输出处detach，因此不能只在现有输入拼query后照旧detach。冻结B权重与用no_grad切断整个计算图是两回事。

可以只让CE更新query、S和记忆，而保持B/A权重梯度规则：B主token的每层K/V先无梯度计算；query分支逐层读取这些detached K/V，并使用B层权重的常量视图计算query，保留对query的导数。这里已有limited LoRA仍只接受action梯度，不能简单移除detach后让CE也更新LoRA。此路由需要实际梯度检查和与非对称整段attention的等价检查。训练/推理额外成本需测量。

## 对当前问题的预期与局限

- 对“同一布局换目标却仍用同一只手”，query提供任务条件下的专用读取位置，但仍须通过同图换任务的指标验证。
- 对“文字说茄子，动作却抓紫薯”，需要query绑定实际物体实例。仅subtask CE可能学会抄任务物体名；已有稀疏定位/一致反事实监督可用于约束query。看query注意力图只能辅助诊断，不是正确定位证明。
- 对初始抓盖/末尾放盖图像相似，当前图像再深的query也不能凭空恢复历史；递归记忆应保留。
- S文本改进不能保证A的动作改进，尤其S当前已说对物体时。仍须分别看目标定位、S选臂/选物、实际前25步动作、真机抓取结果。

后续比较应沿用同一数据、5000步、seed42和辅助监督，对比“B末层之后的query聚合”与“B内部逐层query聚合”，先隔离query读取深度这一个差异。若需要CE训练B，那是另一种梯度合同，应该单独作为后续设计，不与本次两组混在一起。
