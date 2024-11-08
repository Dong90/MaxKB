# coding=utf-8
"""
    @project: maxkb
    @Author：虎
    @file： base_search_dataset_step.py
    @date：2024/1/10 10:33
    @desc:
"""
import os
import logging
from typing import List, Dict


from django.db.models import QuerySet
from langchain.schema import BaseMessage, HumanMessage

from application.chat_pipeline.I_base_chat_pipeline import ParagraphPipelineModel
from application.chat_pipeline.step.search_dataset_step.i_search_dataset_step import (
    ISearchDatasetStep,
)
from common.config.embedding_config import VectorStore, ModelManage
from common.db.search import native_search
from common.util.file_util import get_file_content
from dataset.models import Paragraph, DataSet
from embedding.models import SearchMode
from setting.models import Model
from setting.models_provider import get_model
from smartdoc.conf import PROJECT_DIR
from setting.models_provider.tools import get_model_instance_by_model_user_id
from application.models import ChatRecord
from common.util.split_model import flat_map
from FlagEmbedding import FlagReranker

max_kb_error = logging.getLogger("max_kb_error")
max_kb = logging.getLogger("max_kb")

max_kb_error = logging.getLogger("max_kb_error")
max_kb = logging.getLogger("max_kb")


def get_model_by_id(_id, user_id):
    model = QuerySet(Model).filter(id=_id).first()
    if model is None:
        raise Exception("模型不存在")
    if model.permission_type == "PRIVATE" and str(model.user_id) != str(user_id):
        raise Exception(f"无权限使用此模型:{model.name}")
    return model


def get_embedding_id(dataset_id_list):
    dataset_list = QuerySet(DataSet).filter(id__in=dataset_id_list)
    if len(set([dataset.embedding_mode_id for dataset in dataset_list])) > 1:
        raise Exception("关联知识库的向量模型不一致，无法召回分段。")
    if len(dataset_list) == 0:
        raise Exception("知识库设置错误,请重新设置知识库")
    return dataset_list[0].embedding_mode_id


class BaseSearchDatasetStep(ISearchDatasetStep):

    def execute(
        self,
        problem_text: str,
        dataset_id_list: list[str],
        exclude_document_id_list: list[str],
        exclude_paragraph_id_list: list[str],
        top_n: int,
        similarity: float,
        dialogue_number: int,
        max_paragraph_char_number: int,
        prompt: str,
        history_chat_record: List[ChatRecord],
        no_references_setting=None,
        padding_problem_text: str = None,
        search_mode: str = None,
        user_id=None,
        model_id=None,
        **kwargs,
    ) -> List[ParagraphPipelineModel]:
        if len(dataset_id_list) == 0:
            return []
        exec_problem_text = (
            padding_problem_text if padding_problem_text is not None else problem_text
        )
        embedding_model_id = get_embedding_id(dataset_id_list)
        model = get_model_by_id(embedding_model_id, user_id)
        self.context["model_name"] = model.name
        embedding_model = ModelManage.get_model(
            embedding_model_id, lambda _id: get_model(model)
        )
        embedding_value = embedding_model.embed_query(exec_problem_text)
        vector_name = "pg_vector"
        vector = VectorStore.get_embedding_vector(vector_name)
        embedding_list = vector.query(
            exec_problem_text,
            embedding_value,
            dataset_id_list,
            exclude_document_id_list,
            exclude_paragraph_id_list,
            True,
            top_n,
            similarity,
            SearchMode(search_mode),
        )
        if embedding_list is None:
            return []
        if vector_name == "pg_vector":
            # PG查询返回的是dict，得转成对象兼容es和milvus
            class EmbeddingResult:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)

            result_objects = []
            for item in embedding_list:
                if isinstance(item, dict):
                    result_objects.append(EmbeddingResult(**item))
            embedding_list=result_objects
            
        paragraph_list = self.list_paragraph(embedding_list, vector)
        
        if len(paragraph_list) == 0:
            return []
       # 创建单例重排模型实例
        singleton_reranker = SingletonFlagReranker()

        queries = [exec_problem_text] * len(paragraph_list)
        candidates = [hit["content"] for hit in paragraph_list]

        # 计算重排得分
        scores = singleton_reranker.compute_score(list(zip(queries, candidates)), normalize=True)

        for i, hit in enumerate(paragraph_list):
            hit["score"] = scores[i]

        # 重排后进行排序
        paragraph_list = sorted(paragraph_list, key=lambda p: p["score"], reverse=True)
        result = [
            self.reset_paragraph(paragraph, embedding_list)
            for paragraph in paragraph_list
            if paragraph["score"] > similarity
        ]

        return result

    @staticmethod
    def reset_paragraph(
        paragraph: Dict, embedding_list: List
    ) -> ParagraphPipelineModel:
        filter_embedding_list = [
            embedding
            for embedding in embedding_list
            if str(embedding.paragraph_id) == str(paragraph.get("id"))
        ]
        if filter_embedding_list is not None and len(filter_embedding_list) > 0:
            find_embedding = filter_embedding_list[-1]
            return (
                ParagraphPipelineModel.builder()
                .add_paragraph(paragraph)
                .add_similarity(paragraph["score"])
                .add_comprehensive_score(paragraph["score"])
                .add_dataset_name(paragraph.get("dataset_name"))
                .add_document_name(paragraph.get("document_name"))
                .add_hit_handling_method(paragraph.get("hit_handling_method"))
                .add_directly_return_similarity(
                    paragraph.get("directly_return_similarity")
                )
                .build()
            )

    @staticmethod
    def get_similarity(paragraph, embedding_list: List):
        filter_embedding_list = [
            embedding
            for embedding in embedding_list
            if str(embedding.get("paragraph_id")) == str(paragraph.get("id"))
        ]
        if filter_embedding_list is not None and len(filter_embedding_list) > 0:
            find_embedding = filter_embedding_list[-1]
            return find_embedding.get("comprehensive_score")
        return 0

    @staticmethod
    def list_paragraph(embedding_list: List, vector):
        paragraph_id_list = [row.paragraph_id for row in embedding_list]
        if paragraph_id_list is None or len(paragraph_id_list) == 0:
            return []
        paragraph_list = native_search(
            QuerySet(Paragraph).filter(id__in=paragraph_id_list),
            get_file_content(
                os.path.join(
                    PROJECT_DIR,
                    "apps",
                    "application",
                    "sql",
                    "list_dataset_paragraph_by_paragraph_id.sql",
                )
            ),
            with_table_name=True,
        )
        # 如果向量库中存在脏数据 直接删除
        if len(paragraph_list) != len(paragraph_id_list):
            exist_paragraph_list = [row.get("id") for row in paragraph_list]
            for paragraph_id in paragraph_id_list:
                if not exist_paragraph_list.__contains__(paragraph_id):
                    vector.delete_by_paragraph_id(paragraph_id)
        # 如果存在直接返回的则取直接返回段落
        hit_handling_method_paragraph = [
            paragraph
            for paragraph in paragraph_list
            if (
                paragraph.get("hit_handling_method") == "directly_return"
                and BaseSearchDatasetStep.get_similarity(paragraph, embedding_list)
                >= paragraph.get("directly_return_similarity")
            )
        ]
        if len(hit_handling_method_paragraph) > 0:
            return hit_handling_method_paragraph
        return paragraph_list

    def get_details(self, manage, **kwargs):
        step_args = self.context["step_args"]

        return {
            "step_type": "search_step",
            "paragraph_list": [row.to_dict() for row in self.context["paragraph_list"]],
            "run_time": self.context["run_time"],
            "problem_text": (
                step_args.get("padding_problem_text")
                if "padding_problem_text" in step_args
                else step_args.get("problem_text")
            ),
            "model_name": self.context.get("model_name"),
            "message_tokens": 0,
            "answer_tokens": 0,
            "cost": 0,
        }

    @staticmethod
    def to_human_message(
        prompt: str,
        problem: str,
        max_paragraph_char_number: int,
        history: str,
        paragraph_list: List[ParagraphPipelineModel],
        no_references_setting: Dict,
    ):
        if paragraph_list is None or len(paragraph_list) == 0:
            if no_references_setting.get("status") == "ai_questioning":
                return HumanMessage(
                    content=no_references_setting.get("value").replace(
                        "{question}", problem
                    )
                )
            else:
                return HumanMessage(
                    content=prompt.replace("{data}", "").replace("{question}", problem)
                )
        temp_data = ""
        data_list = [history]
        for p in paragraph_list:
            # 排序。分数
            content = f"{p.title}:{p.content}"
            temp_data += content
            if len(temp_data) > max_paragraph_char_number:
                row_data = content[0 : max_paragraph_char_number - len(temp_data)]
                data_list.append(f"<data>{row_data}</data>")
                break
            else:
                data_list.append(f"<data>{content}</data>")
        data = "\n".join(data_list)
        return HumanMessage(
            content=prompt.replace("{data}", data).replace("{question}", problem)
        )

class SingletonFlagReranker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SingletonFlagReranker, cls).__new__(cls)
            cls._instance.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        return cls._instance

    def compute_score(self, pairs, normalize=False):
        return self.reranker.compute_score(pairs, normalize=normalize)