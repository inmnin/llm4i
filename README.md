# baseline cot总结

### 拥有不同拓扑结构的思维链
CoT思维链：[https://arxiv.org/abs/2205.11916]

ToT思维树：[https://arxiv.org/abs/2305.10601]

GoT思维图：[https://arxiv.org/abs/2308.09687]

FoT思维森林:[https://arxiv.org/abs/2412.09078]

*special*
BoT思维链缓冲区 https://arxiv.org/abs/2406.04271

### 使用蒙特卡洛搜索的思维链

蒙特卡洛树

EoT/XoT-Everything of Thought:[https://arxiv.org/pdf/2311.04254]

蒙特卡洛+self-refine+过程奖励模型（这篇研究的是小模型推理）

Rstar:https://arxiv.org/abs/2501.04519.

树搜索+过程奖励模型（prm）

Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

https://arxiv.org/abs/2408.03314

### 其他类型的思维链

CoT-SC 自洽性思维链[https://arxiv.org/abs/2203.11171]

auto-CoT 自动链式思维[https://arxiv.org/abs/2210.03493](https://arxiv.org/abs/2210.03493)

VerifyCoT验证链式思维[https://arxiv.org/abs/2305.03268](https://arxiv.org/abs/2305.03268)

CoF-CoT协作链式思维[https://arxiv.org/abs/2310.14623](https://arxiv.org/abs/2310.14623)

*这些思维链难以应对复杂的数学/coding问题*

### 使用推理增强方法 强化思维链

##### 反思与自我纠正 

- self-refine [https://arxiv.org/abs/2303.17651]
- 用cot矫正cot中的错误，论文暂时找不到了
  
##### 问题分解 
- ltm CoT(least to most) [https://arxiv.org/abs/2205.10625]
  
##### 高效推理 
- 推理中的投机解码：[https://arxiv.org/abs/2302.01318]
- 推理中的overthinking现象：[https://arxiv.org/pdf/2412.21187]


### 将思维链蒸馏到小模型中：
https://arxiv.org/abs/2212.10071










