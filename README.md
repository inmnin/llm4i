# baseline cot总结

### 拥有不同拓扑结构的思维链
CoT思维链：[https://arxiv.org/abs/2205.11916]

ToT思维树：[https://arxiv.org/abs/2305.10601]

GoT思维图：[https://arxiv.org/abs/2308.09687]

FoT思维森林:[https://arxiv.org/abs/2412.09078]

*special*

BoT思维链缓冲区 https://arxiv.org/abs/2406.04271

### 使用蒙特卡洛搜索的思维链

##### 蒙特卡洛树

EoT/XoT-Everything of Thought:[https://arxiv.org/pdf/2311.04254]

##### 蒙特卡洛+self-refine+过程奖励模型（这篇研究的是小模型推理）

Rstar:https://arxiv.org/abs/2501.04519.

##### 树搜索+过程奖励模型（prm）

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



### 参考文献
| 文献名                                                                                             | 链接                                   | 简介                                                                |
| ----------------------------------------------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------- |
| Learning Harmonized Representations for Speculative Sampling                                    | https://arxiv.org/abs/2408.15766     | 随机采样解码加速                                                          |
| Is Depth All You Need?<br>An Exploration of Iterative Reasoning in LLMs                         | https://arxiv.org/html/2502.10858v1  | 深度思考vs广度思考                                                        |
| When More is Less: Understanding Chain-of-Thought Length in LLMs                                | https://arxiv.org/abs/2502.07266     | CoT的长短黄金比，以及如何达到这种黄金比                                             |
| ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates                            | https://arxiv.org/abs/2502.06772     | 使用思维模版分层推理（with code）                                             |
| Kimi k1.5: Scaling Reinforcement Learning with LLMs                                             | https://arxiv.org/html/2501.12599v1  | kimi 1.5技术报告                                                      |
| Demystifying Long Chain-of-Thought Reasoning in LLMs                                            | https://arxiv.org/abs/2502.03373     | 长度奖励和重复惩罚                                                         |
| Rethinking External Slow-Thinking: From Snowball Errors to Probability of Correct Reasoning     | https://arxiv.org/abs/2501.15602     | 比较不同的外部慢思考对CoT思路与的矫正                                              |
| <br>                                                                                            | https://arxiv.org/abs/2501.18585     | 推理欠思考                                                             |
| Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs                            | https://arxiv.org/abs/2412.21187     | 推理过思考                                                             |
| Markov Chain of Thought for Efficient Mathematical Reasoning                                    | https://arxiv.org/abs/2410.17635     | 马克洛夫思维链                                                           |
| Small Models Struggle to Learn from Strong Reasoners                                            | https://arxiv.org/html/2502.12143v1  | 小模型CoT不能用太长的CoT数据进行SFT                                            |
| Teaching Small Language Models to Reason for Knowledge-Intensive Multi-Hop Question Answering   | https://arxiv.org/abs/2212.08410     | 蒸馏小模型的CoT,并让小模型实现多跳推理（小模型推理加速）                                    |
| Dynamic Chain-of-Thought: Towards Adaptive Deep Reasoning                                       | https://arxiv.org/abs/2502.10428     | D-cot，动态CoT,根据不同的任务动态调整CoT长度                                      |
| From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step                     | https://arxiv.org/abs/2405.14838     | 将显式思维链逐步内化为隐式思维链                                                  |
| **Think-to-Talk or Talk-to-Think?<br>When LLMs Come Up with an Answer in Multi-Step Reasoning** | https://arxiv.org/html/2412.01113v1  | 隐式思维链的可能性？ 先给出answer再给出推理                                         |
| LLMs Do Not Think Step-by-step In Implicit Reasoning                                            | https://arxiv.org/abs/2411.15862     | 同样是对隐式思维链的研究                                                      |
| Investigating Mysteries of CoT-Augmented Distillation                                           | https://arxiv.org/abs/2406.14511     | 1.也是研究了将answer放在CoT前面的可能性<br>2.研究了用特定的符号标记代替某些token的方法，算是一种CoT的压缩 |
| Compressed Chain of Thought: Efficient Reasoning Through Dense Representations                  | https://arxiv.org/abs/2412.13171     | 引入了"沉思标记" 压缩CoT                                                   |
| Token-Budget-Aware LLM Reasoning                                                                | https://arxiv.org/abs/2412.13171     | 通过在提示中包含合理的token预算，可以显著压缩CoT推理过程的token使用                          |
| Generative Verifiers: Reward Modeling as Next-Token Prediction                                  | https://arxiv.org/abs/2408.15240     | 生成式奖励model                                                        |
| Test-time Computing: from System-1 Thinking to System-2 Thinking                                | https://arxiv.org/abs/2501.02497     | 测试时计算方法综述                                                         |
| # TokenSkip: Controllable Chain-of-Thought Compression in LLMs                                  | https://arxiv.org/abs/2502.12067     | 高效可控的CoT压缩方法                                                      |
| Does Scaling Long-CoT Data Unlock Better Slow-Reasoning Systems                                 | https://arxiv.org/html/2501.11284v1  | 长链数据scaling能否提高模型慢推理的能力                                           |
| LOGIDYNAMICS: Unraveling the Dynamics of Logical Inference in Large Language Model Reasoning    | https://www.arxiv.org/pdf/2502.11176 | 各种逻辑推理范式在不同任务中的适应情况                                               |
| Small Models Struggle to Learn from Strong Reasoners                                            | https://arxiv.org/html/2502.12143v1  | 在蒸馏CoT数据给小模型时的讲究                                                  |
| CoT-Valve: Length-Compressible Chain-of-Thought Tuning                                          | https://arxiv.org/abs/2502.09601     | 动态地控制和压缩CoT的长度，以减少推理过程中的计算成本                                      |
| s1: Simple test-time scaling                                                                    | https://arxiv.org/html/2501.19393v2  | 优质多样的long-cot数据                                                   |
| Learning how hard to think: Input-adaptive allocation of lm computation                         | https://arxiv.org/abs/2410.04707     | 不同难度的问题采用不同的llm，agent方案                                           |
| SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs                                | https://arxiv.org/abs/2502.12134     | 连续表示空间内的思维链                                                       |




### Effient CoT
从chatgpt-o1到deepseek-r1, 长链思考已经证明了对于帮助模型解决复杂问题具有很好的效果。但是长思维链中存在着大量的冗余token,造成推理时间过长，推理资源的浪费，因此，如何高效的压缩思维链的长度，用更少的token去解决复杂推理问题，同时保持推理性能不要下降太多成为了一个重要的研究问题。


##### UnEffient CoT的表现形式
- 过度思考：模型推理到某一步时已经其实已经可以得出正确答案了，但是模型仍然在重复的探索新方案，或者对前面的结果不停做reflection，或者是说一些废话。
- 思路的频繁切换：模型在某个推理思路上走不远，浅尝辄止的做了几个step后就开始wait，然后丢弃之前的思路重新思考，这是一种欠思考，浅度思考的表现。
- 冗余文本：在CoT有大量的token其实是压根没有用的，将其去除甚至替换成其他的token，效果可能不降反增。

##### 对UnEfficient CoT 的分析

deepseek-R1为代表的长链推理模型 之所以会在长链CoT中出现大量冗余内容，是因为：

长链条推理生成与RL奖励机制之间的耦合不足。如果仅凭借计算结果奖励模型则会导致模型生成的链会越来越长（R1-Zero开源复现项目中观察到的），让模型养成偏好用长链解决问题的习惯。

先验知识：解决复杂数学/coding问题时，更长的链更容易做对。
在GRPO过程中，多次sampling中的长链更容易把提做对，然后导致整个组的sample都会被鼓励生成更长的链。逐步迭代，模型的链越来越长。

生成冗余长链是 训练模型能更好的推理出正确答案 的必然结果。如果为了减少冗余步骤而在RL中引入PRM的信号，则会影响模型推理答案的准确性。所以直接训练短链模型比较困难，最好是训练长链模型然后再压缩（类似于蒸馏优于大规模sft）， Efficient CoT可以大致分为以下几种方法：
1.**Postrain策略**:长度偏好RL / 用最短拒绝采样或者其他方法构造短链CoT数据,然后再SFT。
2.**推理时策略**：不对推理模型的参数做调整，仅改变模型推理时的策略。
3.**其他策略**：一些新颖奇特的Efficient CoT方法，比如Soft CoT这种。


- 有关precess rewarding model的讨论，尽管训练出长链推理模型最好不用prm，但prm可能仍然可以 对训练好之后的长链推理模型的CoT 进行压缩。


---------
*先观察一下CoT冗余的形式，以及对其的初步解决方案*
*o1-like模型中的两种现象：过思考和欠思考* 
*针对不同的任务难度 ，CoT冗余的类型不同*

----------


### 腾讯AILAB：简单问题的overthinking现象
**Postrain策略**
[Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs]
o1-like的模型在RL的训练过程中不断延长自己的思维链，这里有有一个先验知识：长推理有利于解决复杂的math/code问题，模型要不断地延长推理过程才能更好的拿到奖励信号。

因此产生了一个问题，模型会对一些简单的问题分配大量的计算资源，产生了overthinking的现象。总而言之，在推理中存在着大量冗余的steps,这些没用的token造成了推理资源的浪费。
*2+3=?  qwq的推理example*
![[截屏2025-02-20 13.18.56.png]] 
*2+3=?  deepseek-r1的推理example*
![[截屏2025-02-20 13.21.02.png]]
![[截屏2025-02-20 13.21.17.png]]

#### Overthinking的具体表现
- **答对后继续思考**：当模型给出了一个准确答案后，仍然在不停探索其他解决方案，或者反反复复的检查自己的答案
- **重复的方案探索**：鼓励模型通过多种思路，多种方案来解决问题也未必是一件坏事，但是模型探索的多种方案中存在着大量重复的现象。

#### 文章提出的解决方案
##### 新的指标
- 文章从结果和过程两个视角引入了新颖的效率指标，以评估o1-like模型对计算资源的合理使用。仅仅答对是不可以的，还需要用考虑推理的长度。
##### RL偏好优化
- 推理偏好优化（RPO）
- 简单偏好优化（SimPO）
##### 简化策略
-  **First-Correct Solutions (FCS)**：只保留到**第一次正确答案**出现就截停。
- **FCS+Reflection** ：仅保留第一次正确解答可能会导致o1类模型恢复到传统的LLM行为，因此保留到第二个正确答案生成，两次答案生成之间有一个**reflection**。
- **Greedily Diverse Solutions(GDS)**：FCS+Reflection通常使用相同的推理策略对第一个解决方案的答案进行双重检查，因此提出了一种简单的启发式方法，贪婪地扩展提供**新推理策略**的解决方案。
用这些新的reasoning data去sft模型。

#### 研究局限
只考虑了比较简单的问题，模型容易提前推理出答案，然后继续探索其他解决方案或者reflection。由于问题很简单，所以每个探索过程都很短。

但是如果是比较难的问题，没法提前得到答案，探索方案都很长，得到正确答案的探索方案很少。这种情况下如何压缩CoT还是没有得到很好的解决。

*******
### 腾讯AILAB：复杂问题下的underthinking现象
**推理时策略**
[Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs]
o1-like模型，所表现出的“underthinking”（浅层思考）问题。

underthinking是指这些模型在处理复杂推理任务时，倾向于过早放弃有希望的推理路径，导致推理深度不足，从而影响性能，尤其是在解决具有比较困难的数学问题时。
这种行为会导致模型频繁切换不同的推理思路，而没有充分探索可能带来正确解决方案的路径。

![[截屏2025-02-20 14.12.21.png]]
由图可以看到模型的错误推理中 会出现频繁的方案切换，有着大量探索失败的steps。
#### 抑制underthinking的方法
文中提出了一种新的解码策略—带有思考转换惩罚（Thought Switching Penalty, TIP）的方法，以鼓励模型在转换思路之前更深入地探索每个推理路径，从而提高准确性，

#### 研究局限
限制了方案切换频率会不会导致模型在错误的路上越走越远,并失去及时纠错的能力？

我们希望的是模型在探索正确的推理路径时，在某一条具体的探索路径中要多走几步，深入思考一下实在走不通再考虑切换思路寻找其他的可能正确的路径。但是，单纯的惩罚切换可能会使得模型陷入”滚雪球错误累计“的困境。

或许可以在切换时机上做一些工作，比如在模型在尝试探索某一个推理方案时，要限制放弃当前探索方案的token数量阈值，只要当前方案的token数超过了这个数量就可以自由选择继续探索或者切换思路，少于这个数量就必须就必须继续探索，这样或许能在**促进深度思考**和**抑制”一步错，步步错“的情况**之间取得一个阈值？

这个阈值可以是一个先验值，可以是一个可以动态调整的值，或许是一个由其他模型/大模型自己估计的一个值，也可以是其他等等。

在什么样的时机下应该鼓励模型继续探索，在什么的时机应该鼓励模型停止探索 是很值得研究的。

--------
*由此可见，针对不同的任务应选取不同的推理策略：
- 简单问题的冗余token来自于重复探索
- 复杂问题的冗余token来自于大量的欠思考的尝试
*推理策略要**适应问题难度**，适应问题难度的推理策略产生的CoT是最短的，最有效的*

---------
### 几个简单的适应策略：预先分配token
**推理时策略**
[Learning How Hard to Think: Input-Adaptive Allocation of LM Computation]
[Tokenbudget-aware llm reasoning]
[Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models]
上面几个方法通过预分配token的数量（引入外部模型来做估计）来做思维链长度的适应，这种解决方案过于生硬，而且在推理式模型上不适用，会导致推理"戛然而止"。我们还是希望有一种自适应的解决方案。

### 自适应解决方案1：动态调整思维链长度
**推理时策略**
[Dynamic Chain-of-Thought: Towards Adaptive Deep Reasoning]
通过一系列 复杂的门控/记忆机制+调用外部py库+引入外部模型 实现了CoT长度的动态调整。

![[截屏2025-02-21 22.12.08.png]]

### 自适应解决方案2：调整模型调参方向
**Postrain策略**
[CoT-Valve: Length-Compressible Chain-of-Thought Tuning]
控制微调参数的方向 Δθ来控制模型思维链的长度。
1.参数微调上
CoT-Valve 的核心思想是在模型的参数空间中找到一个方向（记为 Δθ），通过调整这个方向上的参数，可以有效地控制生成的推理链的长度。具体来说，当沿着这个方向调整参数时，模型可以生成从长到短不同长度的推理链。（实验性结论）
2.CoT数据构建
为了训练和优化 Δθ，论文构建了一个名为 **MixChain** 的数据集，其中每个问题都配有从长到短不同长度的推理链（用DeepSeek-R1-DistillLlama-8B生成）。


-------
*通过**大规模**post-train缩短o1-like模型的CoT长度也是一个比较好的方法，不过这个训练开销太大。*

-----
### 长度偏好和重复惩罚
**Postrain策略**
[Demystifying Long Chain-of-Thought Reasoning in LLMs]
该工作在RL中采用余弦长度缩放奖励机制，原文是鼓励模型生成生成长链，但是换一个角度来讲也可以鼓励模型生成短链，kimi k1.5的训练中就很好的运用到了这个技术。
### Long2short:kimi k1.5的技术报告
**Postrain策略**
kimi k1.5在训练中使用了多种训练方法将长链缩短。
- **模型融合**：将长文本 CoT 模型和短文本 CoT 模型的权重进行平均，得到一个新的模型。
- **最短拒绝采样**：从多个采样结果中选择最短且正确的答案。
- **DPO**：使用长文本 CoT 模型生成的答案作为偏好数据来训练短文本 CoT 模型。
- **Long2short RL**：在标准 RL 训练后，使用长度惩罚对模型进行，进一步提高短文本 CoT 模型的效率。
训练思路和R1有很多相同的地方，鼓励短链生成却仍然有着不错的性能。
![[截屏2025-02-20 21.04.12.png]]


-------
*传统的 efficient推理 中有一个经典方法：早停方法，这暂时没有找到将这种方法应用到o1-like模型中的工作*

-------

### 早停方法的研究
针对传统LLM的早停方法很早就有了，在合适的位置让大模型提前截断输出：
[Adaptive inference-time compute: Llms can predict if they can do better, even mid-generation]
[Making language models better reasoners with step-aware verifier]
对于**推理型**模型的早停策略还没有很深入的研究，在推理的中间步骤中，最终答案可能已经形成了？或者说，模型在当前的状态已经具备了输出正确答案的条件。在最终结果正确的推理路径上，可能存在着直接通往答案的捷径，直接给个\<answer\>就可以给出答案？

#### 早停的关键点
##### 产生Answer的时机：Think-to-Talk or Talk-to-Think?
[Think-to-Talk or Talk-to-Think?When LLMs Come Up with an Answer in Multi-Step Reasoning]
该工作初步探究了Answer的产生机制。作者想要确定模型在CoT推理过程中是先内部确定答案再进行解释（即“思考到说话”的模式），还是边解释边得出结论（即“说话到思考”的模式）

引入了外部信号来判断推理过程中哪些变量已经提前被算出。简单来说就是：
让模型推理解答A = 1+B，B = 2 + 3, A = ?这个简单的问题，并在推理过程中使用一些训练好的外部辅助模型（探针）去探测 预测出的每个token下 A 和 B的值已经被确定的概率。

![[截屏2025-02-20 20.38.29.png]]
![[Pasted image 20250220204057.png]]
##### 先确定答案，再生成解释？
[Investigating Mysteries of CoT-Augmented Distillation]
![[截屏2025-02-20 20.43.02.png]]
该工作中研究了小模型的思维范式，证明了在CoT的蒸馏中，用Post CoT比用Pre CoT效果更好。
另外，文章还表明了一些token确实是不必要的，乱序乃至替代成其他语言/符号，对模型性能的影响也不大。


上述工作或许可以表明在最终answer之前，在CoT中的某一个step正确答案已经生成了，给他加一个\<answer\>他就可以直接得到最终的答案。这点或许有可以探索的地方

*"捷径"数据自动生成*->*可以做推理时策略/也可以做SFT策略，最后做一个总结缓冲*

-------
*直接对CoT进行压缩也是一个简单粗暴的Efficient CoT的方法*

------
###  词句合并:用一个”沉思标记“替代一长串token
**其他策略**
[Compressed Chain of Thought: Efficient Reasoning Through Dense Representations]
将显示的词token压缩成大量的内容丰富且连续的“沉思令牌”（contemplation tokens）来提高推理性能，并减少推理成本。

这种方法有点像改变token的划分方式，将推理过程中的一些常见的组合词语或者句子这些由多个token表示的东西 压缩成一个Token进行表示。

在训练过程中，这些沉思token与完整的推理链的隐藏状态相对应进行对齐，分为两个部分：$CCoTφ$负责生成沉思令牌，$DECODEψ$负责解码答案。这种范式也使用next contemplation token方式进行生成，因此可以被扩展到大规模预训练中。

### 跳过冗余token:TokenSkip
**postrain**
#####  LLM TokenSkip
[TokenSkip: Controllable Chain-of-Thought Compression in LLMs] 
该工作引入外部信号，判断当前token是否冗余，这其实会出现上面的按照输入预分配token有一点好处：预分配token数量只考虑到了input，而本方法在计算每一个token是否要skip时同时考虑了input和之前的推理过程。

##### SLM TokenSkip
- 小模型学会多跳策略后推理效果提升很大，因为小模型的CoT无关token的占比特别大：
[Teaching Small Language Models to Reason for Knowledge-Intensive Multi-Hop Question Answering]
[Investigating Mysteries of CoT-Augmented Distillation]

### 投机解码
**推理时策略**
小模型生成对未来K个位置,每个位置生成大量的token候选（Multi-Token-Prediction），大模型来选择。结合**MTP**可以实现推理效率的极大提高，不过性能有所下降。

MTP? NEXT STEP GENERATION？NEXT STEP SKIP？


##### 压缩方法小结
压缩CoT的方法大致可以归纳为以下两种：
1.静态压缩
- 构造新的精简CoT数据重新对LLM做SFT。
- 在推理前对token总数进行约束。
2.动态压缩
- 在推理时引入外部信号对next one token或者next multi token进行限制。

静态的压缩缺乏灵活性，动态压缩造成推理时的额外开销。 可能做Step级而非token级的动态压缩可以缓解这种情况？




------
*直接丢弃传统的显式、离散文本空间内的CoT，转而使用隐式的、连续特征空间内的CoT，来减少推理过程中的冗余计算步骤*

---------

### 隐式CoT
**postrain策略**
[LLMs Do Not Think Step-by-step In Implicit Reasoning]
[From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step]
这类压缩CoT的方法比较极端，其认为模型的思考推理可以不必通过输出长链来完成，其对o1-like模型进行微调，逐步去掉思维链中的一些步骤，每次去除之后用这个更短的思维链重新微调一遍模型，意图"迫使模型将被移除的推理步骤内化到其隐藏状态中",最终实现隐式思维链的效果。

这个方向可能存在一定的问题：
因为推理模型之所以效果好很大程度上推理过程中额外增加的计算量，相关工作的实验也基本只在GSM8K数据集/甚至更简单的数学题上进行测试过。

### Soft CoT / Continuous CoT
**其他策略**
[SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs]
[Training large language models to reason in a continuous latent space]
这是一种非常新颖的CoT范式，其直接放弃了使用离散空间内的文本推理过程，而是用在连续空间内的特征表示来进行推理。
在文本式CoT中，因为许多Token主要用于文本连贯性，对推理本身贡献不大，而一些关键Token则需要复杂的推理计算，文本生成的NTP范式应对这个问题时比较吃力。而在这种软思维链的范式中，LLMs在不受限制的潜在空间而不是语言空间中进行推理，通过将LLM的最后一个隐藏状态直接作为下一个输入嵌入，而不是解码成单词标记。 该领域的工作认为这样可以减少冗余的推理成本（冗余token）。

该方法有几个问题：
- 其在特定领域数据集上做SFT时，极其容易发生灾难性遗忘。
- 可解释性不足。
- 在连续的特征空间中进行推理也会有冗余的信息呀，就像那些冗余的token一样，而且这类工作鲜少与其他CoT方法进行推理成本的对比。


### 结构化思维

### 小模型reason

### 大模型在其他场景的落地
搜索，推广的场景落地？

















