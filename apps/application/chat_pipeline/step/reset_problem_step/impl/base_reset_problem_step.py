# coding=utf-8
"""
    @project: maxkb
    @Author：虎
    @file： base_reset_problem_step.py
    @date：2024/1/10 14:35
    @desc:
"""
import logging
from typing import List

from langchain.chat_models.base import BaseChatModel
from langchain.schema import HumanMessage

from application.chat_pipeline.step.reset_problem_step.i_reset_problem_step import IResetProblemStep
from application.models import ChatRecord
from common.util.split_model import flat_map

from setting.models_provider.tools import get_model_instance_by_model_user_id

max_kb_error = logging.getLogger("max_kb_error")
max_kb = logging.getLogger("max_kb")

prompt = (
    '根据上下文:{context},回答用户问题:{question} 要求: 输出一个补全问题,并且放在<data></data>标签中')

class BaseResetProblemStep(IResetProblemStep):
    def execute(self, problem_text: str, history_chat_record: List[ChatRecord] = None, chat_model: BaseChatModel = None,
                **kwargs) -> str:
        # chat_model = get_model_instance_by_model_user_id(str(self.context['model_id']), str(self.context['user_id']), **kwargs)
        if chat_model is None:
            self.context['message_tokens'] = 0
            self.context['answer_tokens'] = 0
            return problem_text
        start_index = len(history_chat_record) - 3
        # , history_chat_record[index].get_ai_message()
        history_message = [[history_chat_record[index].get_human_message()]
                           for index in
                           range(start_index if start_index > 0 else 0, len(history_chat_record))]
        
        contexts = [[history_chat_record[index].get_human_message().content]
                           for index in
                           range(start_index if start_index > 0 else 0, len(history_chat_record))]
        
        flat_list = [item for sublist in contexts for item in (sublist if isinstance(sublist, list) else [sublist])]
        result = ','.join(flat_list)
        message_list = [*flat_map(history_message), HumanMessage(content=prompt.format(**{'context': result,'question': problem_text}))]
        response = chat_model.invoke(message_list) 
        
        padding_problem = problem_text
        if response.content.__contains__("<data>") and response.content.__contains__('</data>'):
            padding_problem_data = response.content[
                                   response.content.index('<data>') + 6:response.content.index('</data>')]
            if padding_problem_data is not None and len(padding_problem_data.strip()) > 0:
                padding_problem = padding_problem_data
        try:
            request_token = chat_model.get_num_tokens_from_messages(message_list)
            response_token = chat_model.get_num_tokens(padding_problem)
        except Exception as e:
            request_token = 0
            response_token = 0
        self.context['message_tokens'] = request_token
        self.context['answer_tokens'] = response_token
        max_kb.info(f"base_reset_problem_step: {padding_problem} request_token:{request_token} response_token:{response_token}")
        return padding_problem

    def get_details(self, manage, **kwargs):
        return {
            'step_type': 'problem_padding',
            'run_time': self.context['run_time'],
            'model_id': str(manage.context['model_id']) if 'model_id' in manage.context else None,
            'message_tokens': self.context['message_tokens'],
            'answer_tokens': self.context['answer_tokens'],
            'cost': 0,
            'padding_problem_text': self.context.get('padding_problem_text'),
            'problem_text': self.context.get("step_args").get('problem_text'),
        }
