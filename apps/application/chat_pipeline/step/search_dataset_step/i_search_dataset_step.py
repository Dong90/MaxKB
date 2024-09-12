# coding=utf-8
"""
    @project: maxkb
    @Author：虎
    @file： i_search_dataset_step.py
    @date：2024/1/9 18:10
    @desc: 检索知识库
"""
import re
from abc import abstractmethod
from typing import List, Type

from django.core import validators
from rest_framework import serializers

from application.chat_pipeline.I_base_chat_pipeline import IBaseChatPipelineStep, ParagraphPipelineModel
from application.chat_pipeline.pipeline_manage import PipelineManage
from common.util.field_message import ErrMessage
from application.models import ChatRecord
from application.serializers.application_serializers import NoReferencesSetting
from common.field.common import InstanceField


class ISearchDatasetStep(IBaseChatPipelineStep):
    class InstanceSerializer(serializers.Serializer):
        # 原始问题文本
        problem_text = serializers.CharField(required=True, error_messages=ErrMessage.char("问题"))
        # 系统补全问题文本
        padding_problem_text = serializers.CharField(required=False, error_messages=ErrMessage.char("系统补全问题文本"))
        # 需要查询的数据集id列表
        dataset_id_list = serializers.ListField(required=True, child=serializers.UUIDField(required=True),
                                                error_messages=ErrMessage.list("数据集id列表"))
        # 需要排除的文档id
        exclude_document_id_list = serializers.ListField(required=True, child=serializers.UUIDField(required=True),
                                                         error_messages=ErrMessage.list("排除的文档id列表"))
        # 需要排除向量id
        exclude_paragraph_id_list = serializers.ListField(required=True, child=serializers.UUIDField(required=True),
                                                          error_messages=ErrMessage.list("排除向量id列表"))
        # 需要查询的条数
        top_n = serializers.IntegerField(required=True,
                                         error_messages=ErrMessage.integer("引用分段数"))
        # 相似度 0-1之间
        similarity = serializers.FloatField(required=True, max_value=1, min_value=0,
                                            error_messages=ErrMessage.float("引用分段数"))
        search_mode = serializers.CharField(required=True, validators=[
            validators.RegexValidator(regex=re.compile("^embedding|keywords|blend$"),
                                      message="类型只支持register|reset_password", code=500)
        ], error_messages=ErrMessage.char("检索模式"))
        user_id = serializers.UUIDField(required=True, error_messages=ErrMessage.uuid("用户id"))
        model_id = serializers.UUIDField(required=True, error_messages=ErrMessage.uuid("llm模型id"))
        
        
        # 历史对答
        history_chat_record = serializers.ListField(child=InstanceField(model_type=ChatRecord, required=True),
                                                    error_messages=ErrMessage.list("历史对答"))
        # 多轮对话数量
        dialogue_number = serializers.IntegerField(required=True, error_messages=ErrMessage.integer("多轮对话数量"))
        # 最大携带知识库段落长度
        max_paragraph_char_number = serializers.IntegerField(required=True, error_messages=ErrMessage.integer(
            "最大携带知识库段落长度"))
        # 模板
        prompt = serializers.CharField(required=True, error_messages=ErrMessage.char("提示词"))
        # 补齐问题
        padding_problem_text = serializers.CharField(required=False, error_messages=ErrMessage.char("补齐问题"))
        # 未查询到引用分段
        no_references_setting = NoReferencesSetting(required=True, error_messages=ErrMessage.base("无引用分段设置"))

    def get_step_serializer(self, manage: PipelineManage) -> Type[InstanceSerializer]:
        return self.InstanceSerializer

    def _run(self, manage: PipelineManage):     
        paragraph_list = self.execute(**self.context['step_args'])
        manage.context['paragraph_list'] = paragraph_list
        manage.context["is_llm"] = self.context["is_llm"]
        self.context['paragraph_list'] = paragraph_list

    @abstractmethod
    def execute(self, problem_text: str, dataset_id_list: list[str], exclude_document_id_list: list[str],
                exclude_paragraph_id_list: list[str], top_n: int, similarity: float,
                dialogue_number: int,
                max_paragraph_char_number: int,
                prompt: str,
                history_chat_record: List[ChatRecord],
                no_references_setting=None,
                padding_problem_text: str = None,
                search_mode: str = None,
                model_id=None,
                user_id=None,
                **kwargs) -> List[ParagraphPipelineModel]:
        """
        关于 用户和补全问题 说明: 补全问题如果有就使用补全问题去查询 反之就用用户原始问题查询
        :param similarity:                         相关性
        :param top_n:                              查询多少条
        :param problem_text:                       用户问题
        :param dataset_id_list:                    需要查询的数据集id列表
        :param exclude_document_id_list:           需要排除的文档id
        :param exclude_paragraph_id_list:          需要排除段落id
        :param padding_problem_text                补全问题
        :param search_mode                         检索模式
        :param user_id                             用户id
        :return: 段落列表
        """
        pass
