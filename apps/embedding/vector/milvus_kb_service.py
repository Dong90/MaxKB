import uuid
import logging
import json

from abc import ABC, abstractmethod
from typing import Optional, Dict, List

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, MilvusClient, \
    MilvusException

from embedding.models import Embedding, SearchMode
from embedding.vector.base_vector import BaseVectorStore
from embedding.vector.es_kb_service import elasticSearchVectorFactory, ESEmbedding
from common.util.ts_vecto_util import to_ts_vector, to_query

max_kb_error = logging.getLogger("max_kb_error")
max_kb = logging.getLogger("max_kb")


class MilvusEmbedding(BaseModel):
    id: str
    source_id: str
    source_type: str
    is_active: bool
    dataset: dict
    document: dict
    paragraph: dict
    meta: Optional[Dict[str, str]] = None
    embedding: Optional[List[float]] = None
    search_vector: str


@staticmethod
def embedding_to_milvus_embedding(embedding):
    milvusembedding = MilvusEmbedding(id=str(embedding.id),
                                      source_id=str(embedding.source_id),
                                      source_type=str(embedding.source_type),
                                      is_active=embedding.is_active,
                                      dataset=deep_model_to_dict(embedding.dataset, 2),
                                      document=deep_model_to_dict(embedding.document, 2),
                                      paragraph=deep_model_to_dict(embedding.paragraph, 2),
                                      meta=embedding.meta, embedding=embedding.embedding,
                                      search_vector=str(embedding.search_vector))
    return milvusembedding


@staticmethod
def milvus_embedding_to_embedding(milvus_embedding):
    embedding = Embedding(id=milvus_embedding.id,
                          dataset_id=milvus_embedding.dataset.get('id'),
                          document_id=milvus_embedding.document.get('id'),
                          paragraph_id=milvus_embedding.paragraph.get('id'),
                          is_active=milvus_embedding.is_active,
                          #   dataset=milvus_embedding.dataset,
                          #   document=milvus_embedding.document,
                          #   paragraph=milvus_embedding.paragraph,
                          meta=milvus_embedding.meta,
                          embedding=milvus_embedding.embedding,
                          search_vector=milvus_embedding.search_vector,
                          source_id=milvus_embedding.source_id,
                          source_type=milvus_embedding.source_type
                          )

    return embedding


from django.db.models.fields.related import ForeignKey, ManyToManyField, OneToOneField


def deep_model_to_dict(instance, depth=1):
    id_str = str(instance.pk)
    return {'id': id_str}
    # opts = instance._meta
    # data = {}
    # for f in opts.concrete_fields + opts.many_to_many:
    #     if isinstance(f, ManyToManyField):
    #         if instance.pk is None:
    #             data[f.name] = []
    #         else:
    #             data[f.name] = list(f.value_from_object(instance).values_list('pk', flat=True))
    #     elif isinstance(f, ForeignKey) or isinstance(f, OneToOneField):
    #         if getattr(instance, f.name) is not None and depth > 0:
    #             data[f.name] = deep_model_to_dict(getattr(instance, f.name), depth - 1)
    #         else:
    #             data[f.name] = None
    #     else:
    #         data[f.name] = f.value_from_object(instance)
    # return data


class MilvusConfig(BaseModel):
    host: str
    port: str
    username: str
    password: str

    # @model_validator(mode='before')
    # def validate_config(cls, values: dict) -> dict:
    #     if not values['host']:
    #         raise ValueError("config HOST is required")
    #     if not values['port']:
    #         raise ValueError("config PORT is required")
    #     if not values['username']:
    #         raise ValueError("config USERNAME is required")
    #     if not values['password']:
    #         raise ValueError("config PASSWORD is required")
    #     return values


class MilvusVector(BaseVectorStore):

    def __init__(self, index_name: str, config: MilvusConfig, attributes: list):
        self._collection_name = index_name.lower()
        self._client = self._init_client(config)
        self._attributes = attributes

    # 连接到 Milvus 服务器
    def _init_client(self, config: MilvusConfig) -> MilvusClient:
        try:
            client = MilvusClient(
                uri=f'http://{config.host}:{config.port}',
                # user=config.username,
                # password=config.password,
                db_name="default",
            )
            return client
        except Exception as e:
            raise ConnectionError(f"Vector database connection error: {str(e)}")

    def vector_is_create(self) -> bool:
        # 项目启动默认是创建好的 不需要再创建
        return self.vector_exists

    def vector_create(self):
       
        return True

    def _get_or_create_collection(self):
        if not Collection.exists(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)  # 假设嵌入向量的维度是 128
            ]
            schema = CollectionSchema(fields)
            return Collection(name=self.collection_name, schema=schema)
        return Collection(name=self.collection_name)

    def _save(self, text, source_type, dataset_id, document_id, paragraph_id, source_id, is_active, embedding):
        data_id = uuid.uuid1()
        text_embedding = embedding.embed_query(text)
        embedding = Embedding(id=data_id,
                              dataset_id=dataset_id,
                              document_id=document_id,
                              is_active=is_active,
                              paragraph_id=paragraph_id,
                              source_id=source_id,
                              embedding=text_embedding,
                              source_type=source_type,
                            #   search_vector=to_ts_vector(text))
                            search_vector=text)
        self.milvus_save(data_id, text_embedding, [embedding], lambda: True)
        return True

    def _batch_save(self, text_list: List[Dict], embedding: Embeddings, is_save_function):
        texts = [row.get('text') for row in text_list]
        embeddings = embedding.embed_documents(texts)
        data_id = uuid.uuid1()
        embedding_list = [embedding_to_milvus_embedding(Embedding(id=data_id,
                                                                  document_id=text_list[index].get('document_id'),
                                                                  paragraph_id=text_list[index].get('paragraph_id'),
                                                                  dataset_id=text_list[index].get('dataset_id'),
                                                                  is_active=text_list[index].get('is_active', True),
                                                                  source_id=text_list[index].get('source_id'),
                                                                  source_type=text_list[index].get('source_type'),
                                                                  embedding=embeddings[index],
                                                                #   search_vector=to_ts_vector(text_list[index]['text'])))
                                                                  search_vector = text_list[index].get('text')[:10]))
                          for index in
                          range(0, len(text_list))]
        return self.milvus_save(data_id, embeddings, embedding_list, is_save_function)

    def milvus_save(self, uuid, embeddings, embedding_list, is_save_function):
        # 检查集合是否存在，如果不存在则创建
        if not self._client.has_collection(self._collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=4096),
                FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="is_active", dtype=DataType.BOOL),
                FieldSchema(name="dataset", dtype=DataType.JSON, max_length=4096),
                FieldSchema(name="document", dtype=DataType.JSON, max_length=4096),
                FieldSchema(name="paragraph", dtype=DataType.JSON, max_length=4096),
                FieldSchema(name="meta", dtype=DataType.JSON, max_length=4096),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0])),
                FieldSchema(name="search_vector", dtype=DataType.VARCHAR, max_length=4096),
            ]
            schema = CollectionSchema(fields=fields)
            self._client.create_collection(collection_name=self._collection_name, schema=schema)
            index_params = self._client.prepare_index_params()

            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 1024}
            )
            self._client.create_index(
                collection_name=self._collection_name,
                index_params=index_params,
            )
            self._client.load_collection(
                collection_name=self._collection_name
            )
        if is_save_function():
            try:
                data_to_insert = [{
                    "id": em.document['id'],
                    "source_id": em.source_id,
                    "source_type": em.source_type,
                    "is_active": em.is_active,
                    "dataset": json.dumps(em.dataset),
                    "document": json.dumps(em.document),
                    "paragraph": json.dumps(em.paragraph),
                    "meta": json.dumps(em.meta),
                    "embedding": em.embedding,
                    "search_vector": em.search_vector
                } for em in embedding_list]
                self._client.insert(
                    collection_name=self._collection_name,
                    data=data_to_insert
                )
            except MilvusException as e:
                print(f"Milvus 异常: {e}")
        return True

    def hit_test(self, query_text, dataset_id_list: list[str], exclude_document_id_list: list[str], top_number: int,
                 similarity: float,
                 search_mode: SearchMode,
                 embedding: Embeddings):
        text_embedding = embedding.embed_query(query_text)
        return self._milvus_query(query_text, text_embedding, dataset_id_list, exclude_document_id_list, None, True,
                                  top_number, similarity, search_mode)

    def query(self, query_text: str, query_embedding: List[float], dataset_id_list: list[str],
              exclude_document_id_list: list[str],
              exclude_paragraph_list: list[str], is_active: bool, top_n: int, similarity: float,
              search_mode: SearchMode):

        return self._milvus_query(query_text, query_embedding, dataset_id_list, exclude_document_id_list,
                                  exclude_paragraph_list, is_active, top_n, similarity, search_mode)

    def _milvus_query(self, query_text: str, query_embedding: List[float], dataset_id_list: list[str],
                      exclude_document_id_list: list[str],
                      exclude_paragraph_list: list[str], is_active: bool, top_n: int, similarity: float,
                      search_mode: SearchMode):
        search_params = {
            "metric_type": "COSINE",#距离度量 L2为欧氏距离
            # "params":{
            #     "radius": 0.1,
            #     "range_filter": 0.8
            # }
            #https://milvus.io/docs/zh/consistency.md#Consistency-levels
            # 'consistency_level': 'STRONG'#数据一致性等级设为强一致性 
        }
        print(query_embedding)
        if search_mode == SearchMode.embedding:
            results = self._client.search(
                collection_name=self._collection_name,  # 目标集合
                data=[query_embedding],  # 查询向量
                limit=top_n,  # 返回的实体数量
                search_params=search_params,
                output_fields=["id", "source_id", "source_type", "is_active", "dataset", "document", "paragraph",
                               "meta", "search_vector"]
            )
            docs = []
            for sublist in results:
                for hit in sublist:
                    print(hit)
                    if hit['distance'] > similarity:
                        entity_data = hit['entity']
                        for field in ['dataset', 'document', 'paragraph', 'meta']:
                            if isinstance(entity_data.get(field), str):
                                entity_data[field] = json.loads(entity_data[field])
                        embedding = milvus_embedding_to_embedding(MilvusEmbedding(**entity_data))
                        embedding.meta['comprehensive_score'] = hit['distance']
                        embedding.meta['similarity'] = similarity
                        docs.append(embedding)

            docs = sorted(docs, key=lambda x: x.meta['similarity'], reverse=True)

            return docs

    def delete_by_dataset_id(self, dataset_id: str):
        self._client.delete(collection_name=self._collection_name, id=dataset_id)
        return
    def delete_by_dataset_id_list(self, dataset_id_list: List[str]):
        self._client.delete(collection_name=self._collection_name, ids=dataset_id_list)
        return
    def delete_by_source_ids(self, source_ids: List[str], source_type: str):
        self._client.delete(collection_name=self._collection_name, ids=source_ids)
        return

    def update_by_source_ids(self, source_ids: List[str], instance: Dict):
        return

    def update_by_source_id(self, source_id: str, instance: Dict):
        return

    def update_by_paragraph_id(self, paragraph_id: str, instance: Dict):
        return

    def update_by_paragraph_ids(self, paragraph_id: str, instance: Dict):
        return

    def delete_by_document_id(self, document_id: str):
        self._client.delete(collection_name=self._collection_name, ids = document_id)
        return

    def delete_bu_document_id_list(self, document_id_list: List[str]):
        self._client.delete(collection_name=self._collection_name, ids = document_id_list)
        return

    def delete_by_source_id(self, source_id: str, source_type: str):
        self._client.delete(collection_name=self._collection_name, ids = source_id)
        return
        
    def delete_by_paragraph_id(self, paragraph_id: str):
        self._client.delete(collection_name=self._collection_name, ids = paragraph_id)
        return

    def delete_by_paragraph_ids(self, paragraph_ids: List[str]):
        self._client.delete(collection_name=self._collection_name, ids = paragraph_ids)
        return


class MilvusVectorFactory:

    @staticmethod
    def init_vector(attributes: list) -> MilvusVector:
        collection_name = "milvus_db"
        return MilvusVector(
            index_name=collection_name,
            config=MilvusConfig(
                host="127.0.0.1",
                port="19530",
                username="",
                password="",
            ),
            attributes=[]
        )


milvusVectorFactory = MilvusVectorFactory.init_vector([])


class ISearch(ABC):
    @abstractmethod
    def support(self, search_mode: SearchMode):
        pass

    @abstractmethod
    def handle(self, query_set, query_text, query_embedding, top_number: int,
               similarity: float, search_mode: SearchMode):
        pass


class EmbeddingSearch(ISearch):
    def handle(self,
               query_set,
               query_text,
               query_embedding,
               top_number: int,
               similarity: float,
               search_mode: SearchMode):
 
        milvusVectorFactory.query(query_text, query_embedding, None, None, None, True, top_number, similarity,
                                  search_mode)
        return None

    def support(self, search_mode: SearchMode):
        return search_mode.value == SearchMode.embedding.value


# class KeywordsSearch(ISearch):
#     def handle(self,
#                query_set,
#                query_text,
#                query_embedding,
#                top_number: int,
#                similarity: float,
#                search_mode: SearchMode):
#         exec_sql, exec_params = generate_sql_by_query_dict({'keywords_query': query_set},
#                                                            select_string=get_file_content(
#                                                                os.path.join(PROJECT_DIR, "apps", "embedding", 'sql',
#                                                                             'keywords_search.sql')),
#                                                            with_table_name=True)
#         embedding_model = select_list(exec_sql,
#                                       [to_query(query_text), *exec_params, similarity, top_number])
#         return embedding_model

#     def support(self, search_mode: SearchMode):
#         return search_mode.value == SearchMode.keywords.value


# class BlendSearch(ISearch):
#     def handle(self,
#                query_set,
#                query_text,
#                query_embedding,
#                top_number: int,
#                similarity: float,
#                search_mode: SearchMode):
#         exec_sql, exec_params = generate_sql_by_query_dict({'embedding_query': query_set},
#                                                            select_string=get_file_content(
#                                                                os.path.join(PROJECT_DIR, "apps", "embedding", 'sql',
#                                                                             'blend_search.sql')),
#                                                            with_table_name=True)
#         embedding_model = select_list(exec_sql,
#                                       [json.dumps(query_embedding), to_query(query_text), *exec_params, similarity,
#                                        top_number])
#         return embedding_model

#     def support(self, search_mode: SearchMode):
#         return search_mode.value == SearchMode.blend.value

search_handle_list = [EmbeddingSearch()]

# if __name__ == '__main__':
#     factory = ElasticSearchVectorFactory.init()
#     esvector = factory.init_vector(['id', 'content'])
#     esvector.vector_create(["hello world", "hello world2"], [[1,2,3],[4,5,6]], SourceType.text, "test_dataset", "test_document", "test_paragraph", "test_source_id", True)