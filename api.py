# -*- coding: utf-8 -*-
import openai
import argparse

prompt = """
# 任务说明
你的任务是根据输入的用户画像（`profile`）（包含手机操作系统，省份，城市，年龄，性别，教育水平这几个属性），用户的最近几次搜索历史（`session`），当前搜索（`query`），输出改写query或者拒绝改写。你需要分析用户搜索历史session，将用户画像profile进行填充与更新；最后，你要结合当前搜索query思考是否需要用session中的语义信息以及更新后的用户画像profile，产生一定的信息增益。让我们一步一步地来：

一.**第一步**：分析判断用户历史session中是否包含属于用户自身的个性化信息，如年龄，性别，所在地域，短期兴趣点等，对用户画像profile进行补充与更新。请你对profile中的所有属性（手机操作系统，省份，城市，年龄，性别，教育水平）按照以下两步进行分析：
1.**填充缺失字段**: 如果你根据session推测出的某个属性值是profile中缺失的，则进行填充。
2.**更新冲突字段**: 如果你根据session推测出的某个属性值与profile中的属性值冲突，则将推测出的新值覆盖旧值（例如用session推理教育水平为博士，profile中的画像为大专，这时请你将profile中的学历水平字段由大专替换为博士）。
请你给出对所有属性的分析过程。

最后，请你输出修改后的用户画像profile。

二.**第二步**：让我们一步一步来：
1.**输出用户画像**: 例子如：
“用户的画像信息如下：用户是一个女性，年龄为中年，本科学历，住在北京市，使用设备为iphone”
2.**进行改写**: 请你根据上述用户画像，以及用户搜索历史session中的信息，对当前搜索query进行改写，要求分别对**用户画像**和**用户搜索历史session中的信息**进行思考，其是否存在信息增益点，如果存在信息增益点则要写出信息增益点。
如果没有信息增益点则拒绝改写。

## 改写要求

1. 改写query需要体现用户历史session或者用户画像profile。
2. 改写query需要保持语义清晰，无语法错误，且对用户搜索意图的理解准确无误。

## 输出格式要求

1. 请把任务执行过程中的分析，思考和推理过程写入你的分析思考过程中。
2. 输出改写query，只需要输出拒绝改写或者改写后的query，不需要输出其他内容。
3. 输出格式需要严格遵循给定的样例。

# 输入：
- **用户画像profile**：`{{
手机操作系统：android
省份：湖北
城市：孝感
年龄：25-34
性别：女
教育水平：高中及以下

}}`
- **用户的最近几次搜索历史session**：`{{
搜索query:岁岁常欢愉(校园1v1)结局
搜索query:独占糙汉1.v1书香(袋熊布丁)
搜索query:权臣hlH季舒奶
搜索query:首辅大人每天1v1
搜索query:佛子1H1Vpo

}}`
- **当前搜索query**：`{{
腹黑师兄的日常肉食动物不食草
}}`

# 请严格按以下格式完成最终输出输出：
你的分析思考过程：
你更新后的用户画像profile:
改写query：
"""

def call_openai_gpt4(prompt):
    openai.api_key = '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'
    try:
        response = openai.Completion.create(
            engine="gpt-4",  # Replace with "gpt-4" when it becomes available in your API access.
            prompt=prompt,
            max_tokens=150,
            temperature=0.3,
            top_p=1  # Correct parameter name
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    result = call_openai_gpt4(prompt)
    print(result)

if __name__ == "__main__":
    main()
