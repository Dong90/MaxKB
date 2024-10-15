# coding=utf-8
"""
    @project: maxkb
    @Author：虎
    @file： embedding_config.py
    @date：2023/10/23 16:03
    @desc:
"""
import time

from common.cache.mem_cache import MemCache


class ModelManage:
    cache = MemCache('model', {})
    up_clear_time = time.time()

    @staticmethod
    def get_model(_id, get_model):
        model_instance = ModelManage.cache.get(_id)
        if model_instance is None or not model_instance.is_cache_model():
            model_instance = get_model(_id)
            ModelManage.cache.set(_id, model_instance, timeout=60 * 30)
            return model_instance
        # 续期
        ModelManage.cache.touch(_id, timeout=60 * 30)
        ModelManage.clear_timeout_cache()
        return model_instance

    @staticmethod
    def clear_timeout_cache():
        if time.time() - ModelManage.up_clear_time > 60:
            ModelManage.cache.clear_timeout_data()

    @staticmethod
    def delete_key(_id):
        if ModelManage.cache.has_key(_id):
            ModelManage.cache.delete(_id)


class VectorStore:
    from embedding.vector.pg_vector import PGVector
    from embedding.vector.base_vector import BaseVectorStore
    from embedding.vector.es_kb_service import ESVector
    from embedding.vector.es_kb_service import ElasticSearchVectorFactory
    from embedding.vector.milvus_kb_service import MilvusVector
    from embedding.vector.milvus_kb_service import MilvusVectorFactory
    instance_map = {
        'pg_vector': PGVector,
        'elastic_search': ElasticSearchVectorFactory.init_vector([]),
        'milvus': MilvusVectorFactory.init_vector([])
    }
    instance = None

    @staticmethod
    def get_embedding_vector(vector_store_name: str = "pg_vector") -> BaseVectorStore:
        from embedding.vector.pg_vector import PGVector
      
        if VectorStore.instance is None:
            vector_store_class = VectorStore.instance_map.get(vector_store_name,PGVector)
            VectorStore.instance = vector_store_class()
            # es_vector = ElasticSearchVectorFactory.init_vector([])
            # VectorStore.instance = es_vector
            # milvus_vector=MilvusVectorFactory.init_vector([])
            # VectorStore.instance = milvus_vector
        return VectorStore.instance
