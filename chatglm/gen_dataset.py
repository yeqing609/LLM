# 构造微调训练数据集

# 借助deepseek我们可以实现自动化批量构造训练数据集。

#下面我们以中国古典哲学数据集为例，展示了自动构造训练集的主要流程：

# - 使用 LangChain 构造训练数据样例
#     - 基于 deepseek 设计 `System Role` 提示词
#     - 使用 `deepseek-chat` 生成基础数据
#     - 解析 deepseek-chat 生成的训练数据
#     - 持久化存储`dataset.csv`训练数据集文件
#     - 使用 deepseek 实现训练数据多样化
# - 自动化批量生成训练数据集
#     - 整理收集原始数据`raw_data.txt`
#     - 自动解析原始数据样例 `raw_data_content[]`
#     - 设计 `gen_data` 训练数据生成器函数
#     - 设计训练数据生成流水线


# 使用 LangChain 构造训练数据
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import os
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()
from langchain_deepseek import ChatDeepSeek
chat = ChatDeepSeek(
    model="deepseek-chat",  # DeepSeek API 支持的模型名称
    temperature=1,          # 控制输出的随机性
    max_tokens=4095,        # 最大输出长度
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量加载API key
)

system_content = """
你是中国古典哲学大师，尤其擅长周易的哲学解读。

接下来，你收到的都是关于周易卦象的解释，你需要整理润色，并生成用于大模型训练的内容和格式。

示例输入：

师卦，此卦是异卦相叠，下卦为坎，上卦为坤。“师”指军队。坎为水、为险；坤为地、为顺，喻寓兵于农。兵凶战危，用兵乃圣人不得已而为之，但它可以顺利无阻碍地解决矛盾，因为顺乎形势，师出有名，故能化凶为吉。占得此卦，对于军事上率师出征非常有利，必无灾祸。师卦是天马出群之卦，以寡伏众之象。
师卦位于讼卦之后，《序卦》之中这样解释道：“讼必有众起，故受之以师。师者，众也。”争讼的人越来越多，以致形成了军队。

期待结果：

content:"师卦"
summary:"在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。

师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。

师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"
"""

# 原始数据
raw_content = "蒙卦是教育启蒙的智慧，艮为山，坎为泉，山下出泉。泉水始流出山，则必将渐汇成江河,正如蒙稚渐启，又山下有险，因为有险停止不前，所以蒙昧不明。事物发展的初期阶段，必然蒙昧，所以教育是当务之急，养学生纯正无邪的品质，是治蒙之道。\n蒙卦，这个卦是异卦相叠，下卦为坎，上卦为艮。艮是山的形象，喻止；坎是水的形象，喻险。卦形为山下有险，仍不停止前进，是为蒙昧，故称蒙卦。但因把握时机，行动切合时宜;因此，具有启蒙和通达的卦象。\n《蒙》卦是《屯》卦这个始生卦之后的第二卦。《序卦》中说：“物生必蒙，故受之以蒙。蒙者，蒙也，特之稚也。”物之幼稚阶段，有如蒙昧未开的状态，在人则是指童蒙。\n《象》中这样解释蒙卦：山下出泉，蒙；君子以果行育德。"

messages = [
    SystemMessage(
        content=system_content
    ),
    HumanMessage(
        content=raw_content
    ),
]

ai_message = chat(messages)

# 解析DeepSeek 生成的训练数据
ai_message.content

text = ai_message.content
# 分割字符串来找到content和summary的位置
content_start = text.find('content:"') + len('content:"')
content_end = text.find('"\nsummary:')
summary_start = text.find('summary:"') + len('summary:"')
summary_end = text.rfind('"')

# 提取并存储content和summary
content = text[content_start:content_end].strip()
summary = text[summary_start:summary_end].strip()

# 持久化存储训练数据集文件
import csv

# 如果没有API_KEY，可以使用预定义的变量
# content = "蒙卦"
# summary = "在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"

# 新建CSV文件并写入数据
with open('test_dataset.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入标题行
    writer.writerow(['content', 'summary'])
    # 写入数据行
    writer.writerow([content, summary])


# 数据增强：构造多样化的提问方式
def generate_question_summary_pairs(content, summary):
    """
    生成20对提问和总结的配对。

    :param content: 内容（例如：“蒙卦”）。
    :param summary: 内容的总结。
    :return: 包含20对提问和总结的列表。
    """
    # 20种提问模板
    question_templates = [
        "{}代表什么？",
        "周易中的{}含义是什么？",
        "请解释一下{}。",
        "{}在周易中是什么象征？",
        "周易{}的深层含义是什么？",
        "{}和教育启蒙有什么联系？",
        "周易的{}讲述了什么？",
        "{}是怎样的一个卦象？",
        "{}在周易中怎样表达教育的概念？",
        "{}的基本意义是什么？",
        "周易中{}的解释是什么？",
        "{}在周易中代表了哪些方面？",
        "{}涉及哪些哲学思想？",
        "周易中{}的象征意义是什么？",
        "{}的主要讲述内容是什么？",
        "周易{}的核心思想是什么？",
        "{}和启蒙教育之间有何联系？",
        "在周易中，{}象征着什么？",
        "请描述{}的含义。",
        "{}在周易哲学中扮演什么角色？"
    ]

    # 使用content填充提问模板
    questions = [template.format(content) for template in question_templates]

    # 创建提问和总结的配对
    question_summary_pairs = [(question, summary) for question in questions]

    return question_summary_pairs


import csv

# 如果没有GPT API，可以使用预定义的变量
# content = "蒙卦"
# summary = "在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"
pairs = generate_question_summary_pairs(content, summary)

# 将结果写入CSV文件
with open('test_dataset.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['content', 'summary'])
    for pair in pairs:
        writer.writerow(pair)


# 自动化批量生成训练数据流水线
# 原始数据来源：https://www.zhouyi.cc/zhouyi/yijing64/4103.html

# 初始化一个空列表用于存储原始内容数据
raw_content_data = []

# 读取文件并分割数据样例
with open('data/raw_data.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    # 使用连续的换行符('\n\n')作为分隔符来分割文本
    data_samples = content.split('\n\n')

    # 遍历分割后的数据样例并添加到列表中
    for sample in data_samples:
        # 移除每个样例中的额外空白字符（如果有的话）
        cleaned_sample = sample.strip()
        # 仅添加非空样例
        if cleaned_sample:
            raw_content_data.append(cleaned_sample)



# 将以上的所有模块，整合到一起，自动化生成数据
from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# 初始化LangChain的deepseek-chat调用
chat = ChatDeepSeek(
    model="deepseek-chat",  # DeepSeek API 支持的模型名称
    temperature=1,          # 控制输出的随机性
    max_tokens=4095,        # 最大输出长度
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量加载API key
)

def gen_data(raw_content):
    """
    使用LangChain deepseek-chat调用处理单个数据样例。

    :param raw_content: 原始数据样例。
    :return: deepseek-chat模型生成的内容。
    """
    # 系统消息定义背景和任务
    system_message = SystemMessage(
        content="""
        你是中国古典哲学大师，尤其擅长周易的哲学解读。

        接下来，你收到的都是关于周易卦象的解释，你需要整理润色，并生成用于大模型训练的内容和格式。

        示例输入：

        师卦，此卦是异卦相叠，下卦为坎，上卦为坤。“师”指军队。坎为水、为险；坤为地、为顺，喻寓兵于农。兵凶战危，用兵乃圣人不得已而为之，但它可以顺利无阻碍地解决矛盾，因为顺乎形势，师出有名，故能化凶为吉。占得此卦，对于军事上率师出征非常有利，必无灾祸。师卦是天马出群之卦，以寡伏众之象。
        师卦位于讼卦之后，《序卦》之中这样解释道：“讼必有众起，故受之以师。师者，众也。”争讼的人越来越多，以致形成了军队。

        期待结果：

        content:"师卦"
        summary:"在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。

        师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。

        师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"

        返回格式要求：
        content:"{卦名}"
        summary:"{内容}"
        """
    )

    # 人类消息包含原始数据样例
    human_message = HumanMessage(
        content=raw_content
    )

    # 构建消息列表并进行模型调用
    messages = [system_message, human_message]
    ai_message = chat(messages)

    return ai_message.content


def dataset_parser(ai_message_content):
    """
    解析由gen_data函数生成的ai_message.content，提取content和summary。

    :param ai_message_content: gen_data函数返回的文本。
    :return: 提取的content和summary。
    """
    # 分割字符串来找到content和summary的位置
    content_start = ai_message_content.find('content:"') + len('content:"')
    content_end = ai_message_content.find('"\nsummary:')
    summary_start = ai_message_content.find('summary:"') + len('summary:"')
    summary_end = ai_message_content.rfind('"')

    # 提取并存储content和summary
    content = ai_message_content[content_start:content_end].strip()
    summary = ai_message_content[summary_start:summary_end].strip()

    return content, summary

import csv
import datetime
import os

def main():
    # 确保 data 目录存在
    if not os.path.exists('data'):
        os.makedirs('data')

    # 解析 data/raw_data.txt 得到 raw_content_data 列表
    raw_content_data = []
    with open('data/raw_data.txt', 'r', encoding='utf-8') as file:
        content = file.read()
        data_samples = content.split('\n\n')
        for sample in data_samples:
            cleaned_sample = sample.strip()
            if cleaned_sample:
                raw_content_data.append(cleaned_sample)

    # 创建带有时间戳的CSV文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/zhouyi_dataset_{timestamp}.csv"

    # 创建CSV文件并写入标题行
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['content', 'summary'])

        # 循环遍历 raw_content_data 数据样例
        for raw_content in raw_content_data:
            # 调用 gen_data 方法得到 ai_message_content
            ai_message_content = gen_data(raw_content)

            # 解析 ai_message_content 得到 content 和 summary
            content, summary = dataset_parser(ai_message_content)
            
            print("Content:", content)
            print("Summary:", summary)

            # 调用 generate_question_summary_pairs 得到20组 pairs
            pairs = generate_question_summary_pairs(content, summary)

            # 将 pairs 写入 csv 文件
            for pair in pairs:
                writer.writerow(pair)


# 执行主函数
main()
