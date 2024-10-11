import pandas as pd
import requests
import ast
import json
from datasets import Dataset
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate, adapt
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    # context_relevance,
)
def get_chat_record(application_id, chat_id, record_id):
    url = f'http://183.131.7.9:8003/api/application/{application_id}/chat/{chat_id}/chat_record/{record_id}'

    headers = {
        'AUTHORIZATION': 'eyJ1c2VybmFtZSI6ImFkbWluIiwiaWQiOiJmMGRkOGY3MS1lNGVlLTExZWUtOGM4NC1hOGExNTk1ODAxYWIiLCJlbWFpbCI6IiIsInR5cGUiOiJVU0VSIn0:1szBEw:_aORb6WmKRYem8mpJnX2ajOo39G_FVog4RxSc-VMEP8',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'http://183.131.7.9:8003/ui/application/cef470c6-603b-11ef-87f1-26cf8447a8c9/SIMPLE/setting',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
    }
    return requests.get(url, headers=headers, verify=False)

def send_chat_message(message, chat_uuid):
    url = f'http://183.131.7.9:8003/api/application/chat_message/{chat_uuid}'
    headers = {
        'AUTHORIZATION': 'eyJ1c2VybmFtZSI6ImFkbWluIiwiaWQiOiJmMGRkOGY3MS1lNGVlLTExZWUtOGM4NC1hOGExNTk1ODAxYWIiLCJlbWFpbCI6IiIsInR5cGUiOiJVU0VSIn0:1szBEw:_aORb6WmKRYem8mpJnX2ajOo39G_FVog4RxSc-VMEP8',
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'Origin': 'http://183.131.7.9:8003',
        'Referer': 'http://183.131.7.9:8003/ui/application/cef470c6-603b-11ef-87f1-26cf8447a8c9/SIMPLE/setting',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
    }

    data = {
        "message": message,
        "re_chat": False,
        "stream": False
    }

    return requests.post(url, headers=headers, json=data, verify=False)

def test_dialogue(llm, messages):
    """
    测试对话并打印结果。

    :param llm: 语言模型实例
    :param messages: 对话消息列表
    :return: 解析后的结果
    """
    result = llm.invoke(messages)
    parser = StrOutputParser()
    parsed_result = parser.invoke(result)
    print(parsed_result)
    return parsed_result


def ask_question(question, chat_id):
    """
    发送问题并打印答案。

    :param question: 要提出的问题
    :param chat_id: 聊天会话 ID
    :return: record_id
    """
    chat_resp = send_chat_message(question, chat_id)
    response_data = chat_resp.json()

    if chat_resp.status_code == 200:
        content = response_data['data']['content']
        print(f"问题: {question}")
        print(f"答案: {content}\n")
    else:
        print(f"接口调用失败 {chat_resp.status_code}")
        print(chat_resp.text)
    return response_data['data']['id']

def fetch_question_details(application_id, chat_id, record_id):
    """
    获取提问记录的详细信息，并处理响应。

    :param application_id: 应用程序 ID
    :param chat_id: 聊天会话 ID
    :param record_id: 提问记录 ID
    :return: (answers, contexts)
    """
    record_resp = get_chat_record(application_id, chat_id, record_id)
    answers = []
    contexts = []

    if record_resp.status_code == 200:
        record_data = record_resp.json()
        answers.append(record_data['data']['answer_text'])
        single_contexts = []
        for paragraph in record_data['data'].get('paragraph_list', []):
            # print(f"提问信息: {record_data}\n")
            context = paragraph.get('content')
            single_contexts.append(context)
        contexts.append(single_contexts)
    else:
        print(f"接口调用失败，响应返回: {record_resp.status_code}")
        print(record_resp.text)

    return answers, contexts


def read_excel(file_path):
    """
    从 Excel 文件中读取问题、答案和上下文。

    :param file_path: Excel 文件路径
    :return: (questions, answers, contexts)
    """
    df = pd.read_excel(file_path)
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    contexts = df['contexts'].tolist()
    # parsed_contexts = [ast.literal_eval(c) for c in contexts]
    parsed_contexts = [[c] for c in contexts]
    return questions, answers, parsed_contexts

# 初始化模型
langchain_embeddings = OpenAIEmbeddings(base_url='https://api2.aigcbest.top/v1',
                                        api_key='sk-0vc6oFIta4zDcjqS714dEd3d50854183902b7f857dEc07F4',
                                        model="text-embedding-ada-002")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="https://api2.aigcbest.top/v1",
                 api_key="sk-0vc6oFIta4zDcjqS714dEd3d50854183902b7f857dEc07F4")

# messages = [
#     HumanMessage(content="你好，我想了解如何使用Python进行数据分析。!"),
# ]
#
# print(test_dialogue(llm, messages))

# 评估参数
file_path = 'pg_ragas_result.xlsx'
questions, _, _ = read_excel(file_path)
# questions = [
#     '什么是CNAME，如何配置？',
#     '如何获取存储文件的外链？',
#     '七牛云的上传下载操作指南是什么？',
#     '如何测试域名使用规范？',
#     '如何使用base64将图片上传到七牛云？',
# ]
contexts = []
answers = []
ground_truths = ['要配置域名的 CNAME，您需要按照以下步骤操作：'
'在七牛云Portal创建加速域名：首先登录七牛云Portal，创建一个加速域名，这样您会获得一个 CNAME 域名，'
'例如 example.qiniudns.com。登录您的域名服务商账户：使用您的域名服务商提供的管理界面，登录您的账户。'
'添加 CNAME 记录：创建一条主机记录，名称填写 example（您加速域名中的主机部分）。'
'记录类型选择 CNAME。记录值填写 example.qiniudns.com（您在七牛云Portal获取的 CNAME 域名）。'
'保存更改：确认保存您的设置，这样您的 CNAME 记录就会被添加。'
'等待生效：DNS 记录的更改可能需要一些时间才能生效，通常在几分钟到48小时之间。'
'完成以上步骤后，您的域名请求就会指向七牛云 CDN，从而享受CDN加速效果。',
'要购买 SSL 证书，您可以按照以下步骤操作：',
'功能很多',
'七牛云存储是一个云存储服务，它提供了各种功能，如文件上传、下载、删除、文件元数据管理、文件访问控制等。',
'上传文件',
]

# 调接口
application_id = 'cef470c6-603b-11ef-87f1-26cf8447a8c9'
chat_id = 'e3b31394-87ac-11ef-9d0f-26cf8447a8c9'

for question in questions:
    record_id =  ask_question(question, chat_id)
    q_answers, q_contexts = fetch_question_details(application_id, chat_id, record_id)
    answers.extend(q_answers)
    contexts.extend(q_contexts)

# 构造数据集
data_samples = {
    'question': questions,
    'answer': answers,
    'contexts': contexts,
    # 'ground_truth': ground_truths
}

dataset = Dataset.from_dict(data_samples)
print(faithfulness.statement_prompt.to_string())
adapt(metrics=[faithfulness, answer_relevancy], language="chinese", llm=llm, cache_dir='.\\ragas\\cache\\')
print(faithfulness.statement_prompt.to_string())

# 进行评估
result = evaluate(dataset, metrics=[
    # context_relevancy,
    # context_recall,
    faithfulness,
    answer_relevancy,
], llm=llm, embeddings=langchain_embeddings)

df = result.to_pandas()

# 选择所需的列
# selected_columns = [
#     # 'context_relevancy',
#     # 'context_recall',
#     'faithfulness',
#     'answer_relevancy'
# ]
# df_selected = df[selected_columns]

# 设置 Pandas 显示选项以显示所有内容
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整宽度
pd.set_option('display.max_colwidth', None)  # 显示完整列内容

# 打印结果
# print(df_selected)
df.to_excel('new_ragas_result.xlsx', index=False)



'''
通过langchain生成llm和embeding模型实例，将数据和实例传入ragas中的evaluate方法中后会得到四个变量：分别为：

- 忠实度(faithfulness)
忠实度(faithfulness)衡量了生成的答案(answer)与给定上下文(context)的事实一致性。它是根据answer和检索到的context计算得出的。并将计算结果缩放到 (0,1) 范围且越高越好。

总结：答案中可以通过文档推断出的观点/答案中所有的观点
推荐模型：hhem

- 答案相关性(Answer relevancy)
评估指标“答案相关性”重点评估生成的答案(answer)与用户问题(question)之间相关程度。不完整或包含冗余信息的答案将获得较低分数。该指标是通过计算question和answer获得的，它的取值范围在 0 到 1 之间，其中分数越高表示相关性越好

总结：使用llm为大模型回答的答案反向生成多个问题，然后计算多个生成的问题和实际问题之间的平均余弦相似度

- 上下文精度(Context precision)
上下文精度是一种衡量标准，它评估所有在上下文(contexts)中呈现的与基本事实(ground-truth)相关的条目是否排名较高。理想情况下，所有相关文档块(chunks)必须出现在顶层。该指标使用question和计算contexts，值范围在 0 到 1 之间，其中分数越高表示精度越高。

总结：计算每个检索出来的文档和标准答案之间的相关性，然后计算平均值

- 上下文召回率(Context recall)
上下文召回率(Context recall)衡量检索到的上下文(Context)与人类提供的真实答案(ground truth)的一致程度。它是根据ground truth和检索到的Context计算出来的，取值范围在 0 到 1 之间，值越高表示性能越好。

总结：标准答案中可以通过检索文档得出的观点/标准答案中所有的观点
'''



