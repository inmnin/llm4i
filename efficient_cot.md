### 研究现状
从chatgpt-o1到deepseek-r1, 长链思考已经证明了对于帮助模型解决复杂问题具有很好的效果。但是长思维链中存在着大量的冗余token,造成推理时间过长，推理资源的浪费，因此，如何高效的压缩思维链的长度，用更少的token去解决复杂推理问题，同时保持推理性能不要下降太多成为了一个重要的研究问题。

生成冗余长链是 训练模型能更好的推理出正确答案 的必然结果。如果为了减少冗余步骤而在RL中引入PRM的信号，则会影响模型推理答案的准确性。所以直接训练短链模型比较困难，最好是训练长链模型然后再压缩（类似于蒸馏优于大规模sft）， Efficient CoT可以大致分为以下几种方法：

1.**Postrain策略**:长度偏好RL / 用最短拒绝采样或者其他方法构造短链CoT数据,然后再SFT。
2.**推理时策略**：不对推理模型的参数做调整，仅改变模型推理时的策略。比如预先分配token预算，tokenskip等策略。
3.**其他策略**：一些新颖奇特的Efficient CoT方法，比如Soft CoT这种，将cot从离散的文本空间映射到连续的表示空间

以上cot方法中，posttrain方法存在以下问题：
- 优质的first answer的sft数据难以获取。
- 在特定数据集上做SFT,会损害模型的泛化性能。
- 对训练好的r1-like模型做长度偏好RL会损害模型的一般性能。

当前的推理时策略存在以下问题：性能下降严重。无论是预先分配token预算还是用tokenskip的方式对思维链进行压缩，都会导致模型性能出现严重的下降，甚至在MATH-500/gsm-800k上都会出现大幅度掉点。

其他策略比如soft-cot这种性能下降更加严重，相关工作甚至只测试了gsm-800k的效果，可见其模型最基本的解决math问题的能力十分糟糕。

### 当前研究的问题
#### 现象发现：过度反思----->First is best
在数学问题中，即模型在得出第一个答案（first answer）之后不会停止思考，其会对答案进行反反复复的检查。

在模型的过度反思的步骤中，以下反思类型占比最大：

交叉检验（不可靠的反思）：
模型使用另外一种方法，将题目重新做一遍，然后得出新的答案，再与first answer进行比较。
模型的first answer往往会倾向于使用简单的方法解题，因此正确率很高，而模型的second answer/third answer...会倾向于使用复杂，怪辟的方法解题，出错的概率比较高，一旦推出了错误的答案，将带偏原本正确的first answer，带来极大的风险。
<img width="677" alt="截屏2025-04-25 11 48 13" src="https://github.com/user-attachments/assets/5a5ee60e-5b00-488c-8d34-369e4f066ee3" />

#### 研究点1：first is best现象的发现与理论分析
**first answer->first solution**
我们在GPQA（科学知识推理任务）上进行了测试，同样的发现了first is best的现象，过度的交叉检验反思有带偏原有正确solution的问题。
![Uploading 截屏2025-04-25 11.50.20.png…]()


我们或许可以推测，在其他的任务上，比如常识推理，符号逻辑推理上也有一样的现象呢？
更进一步，在ground truth不唯一的任务上，比如code任务上，是不是也存在着这种现象？first solution是代码结构最简洁，规范，时间复杂度最低的solution?
甚至是在无ground truth的任务上，比如写作任务，是不是 first solution最在文学意义上 是最优的。
first is best的现象可能在r1-like模型中广泛存在。
#### 研究点2：first is best现象的分析，为什么第一个最好，后面的不行？
- 拟从上下文学习icl中/强化学习grpo的角度进行分析。
#### 研究点3：如何控制r1-like模型在生成first answer后立刻终止？
##### prompt方法（train-free）
设计prompt，使llm得出first answer时添加一些特殊的标记
##### sft/rl（train）
- 优质的first answer的sft数据难以获取。
- 在特定数据集上做SFT,会损害模型的泛化性能。
- 对训练好的r1-like模型做长度偏好RL会损害模型的一般性能。
##### llm knows when the answer is correct
插入prompt的方式进行思维干预，诱导模型提前给出answer。
如果correct answer已经生成，多次诱导得到的answer应该相同，如果correct answer在当前思维链的上下文中还没有生成，则其倾向于给出随机答案。

### 目前的实验结果
<img width="456" alt="截屏2025-04-25 11 50 51" src="https://github.com/user-attachments/assets/76f79519-2be6-4091-aaec-c6ff25d430d7" />


- 拟进一步对baseline众多，也更加轻量化的的qwq-32b模型进行测试。
- 拟进一步对除了math问题以外的其他问题进行测试。
### 未来研究的问题
对first answer之前的思维链进行进一步的高效压缩。

因为大规模rl会倾向于生成长链，如果强行干扰这个链不断变长的过程会导致模型性能受损。因此我们倾向于使用已经训练好的r1-like模型，使用train-free的方法使其缩短思维链长度。

思维干预压缩：生成一段文本raw text之后让模型自行凝练思维链内容,得到compressed text，然后将原来的raw text全部mask掉，后序的next token prediction不再使用raw text，而仅仅使用更加精炼的compressed text。
