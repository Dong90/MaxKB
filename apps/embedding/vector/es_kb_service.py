import json
import os
import uuid
import logging
import requests
import sys,os
# sys.path.append("/root/dsj/MaxKB1")
from abc import ABC, abstractmethod
from typing import Dict, List
from pydantic import BaseModel, model_validator
from embedding.models import Embedding, SourceType, SearchMode, DataSet, Document, Paragraph
from embedding.vector.pg_vector import PGVector
from embedding.vector.base_vector import BaseVectorStore
from elasticsearch import Elasticsearch
from langchain_core.embeddings import Embeddings
from common.util.ts_vecto_util import to_ts_vector, to_query
from django.forms.models import model_to_dict
from django.apps import apps
import json

max_kb_error = logging.getLogger("max_kb_error")
max_kb = logging.getLogger("max_kb")


from json import JSONEncoder
import datetime

class ESEmbedding(BaseModel):
    
    id: str
    source_id: str
    source_type: str
    is_active: bool
    dataset: dict
    document: dict
    paragraph: dict
    meta: dict
    embedding: List[float]
    search_vector: str
    
@staticmethod
def embedding_to_es_embedding(embedding):
    esembedding = ESEmbedding(id=str(embedding.id), 
                              source_id=str(embedding.source_id), 
                              source_type=str(embedding.source_type), 
                              is_active=embedding.is_active, 
                              dataset=deep_model_to_dict(embedding.dataset, 2), 
                              document=deep_model_to_dict(embedding.document, 2), 
                              paragraph=deep_model_to_dict(embedding.paragraph,2), 
                              meta=embedding.meta, embedding=embedding.embedding, 
                              search_vector=str(embedding.search_vector))
    return esembedding

@staticmethod
def es_embedding_to_embedding(es_embedding):    
    embedding = Embedding(id=es_embedding.id, 
                          dataset_id=es_embedding.dataset.get('id'), 
                          document_id=es_embedding.document.get('id'),
                          paragraph_id=es_embedding.paragraph.get('id'),
                          is_active=es_embedding.is_active,   
                        #   dataset=es_embedding.dataset, 
                        #   document=es_embedding.document, 
                        #   paragraph=es_embedding.paragraph, 
                          meta=es_embedding.meta, 
                          embedding=es_embedding.embedding, 
                          search_vector=es_embedding.search_vector,
                          source_id=es_embedding.source_id,
                          source_type=es_embedding.source_type                     
    )
    
    
    return embedding

from django.db.models.fields.related import ForeignKey, ManyToManyField, OneToOneField
def deep_model_to_dict(instance, depth=1):
    opts = instance._meta
    data = {}
    for f in opts.concrete_fields + opts.many_to_many:
        if isinstance(f, ManyToManyField):
            if instance.pk is None:
                data[f.name] = []
            else:
                data[f.name] = list(f.value_from_object(instance).values_list('pk', flat=True))
        elif isinstance(f, ForeignKey) or isinstance(f, OneToOneField):
            if getattr(instance, f.name) is not None and depth > 0:
                data[f.name] = deep_model_to_dict(getattr(instance, f.name), depth-1)
            else:
                data[f.name] = None
        else:
            data[f.name] = f.value_from_object(instance)
    return data

class ElasticSearchConfig(BaseModel):
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

class ESVector(BaseVectorStore):
    
    def __init__(self, index_name: str, config: ElasticSearchConfig, attributes: list):
        self._collection_name = index_name.lower()  
        self._client = self._init_client(config)
        self._attributes = attributes
    
    def _init_client(self, config: ElasticSearchConfig) -> Elasticsearch:
        try:
            client = Elasticsearch(
                hosts=f'http://{config.host}:{config.port}',
                basic_auth=(config.username, config.password),
                request_timeout=100000,
                retry_on_timeout=True,
                max_retries=10000,
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Vector database connection error")

        return client
    
    def vector_is_create(self) -> bool:
        # 项目启动默认是创建好的 不需要再创建
        return self.vector_exists

    def vector_create(self):
        # if not self.vector_exists:
        #     for index in range(len(texts)):
        #         self.save(texts[index], source_type, dataset_id, document_id, paragraph_id, source_id, is_active, embeddings[index])
        #     self.vector_exists = True
        return True
    
    def _save(self, text, source_type: SourceType, dataset_id: str, document_id: str, paragraph_id: str, source_id: str,
              is_active: bool,
              embedding: Embeddings):
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
                              search_vector=to_ts_vector(text))
        self.es_save(data_id, text_embedding, [embedding], lambda: True)
        return True

    def _batch_save(self, text_list: List[Dict], embedding: Embeddings, is_save_function):
        texts = [row.get('text') for row in text_list]
        embeddings = embedding.embed_documents(texts)
        data_id = uuid.uuid1()   
        embedding_list = [embedding_to_es_embedding(Embedding(id=data_id,
                                    document_id=text_list[index].get('document_id'),
                                    paragraph_id=text_list[index].get('paragraph_id'),
                                    dataset_id=text_list[index].get('dataset_id'),
                                    is_active=text_list[index].get('is_active', True),
                                    source_id=text_list[index].get('source_id'),
                                    source_type=text_list[index].get('source_type'),
                                    embedding=embeddings[index],
                                    search_vector=to_ts_vector(text_list[index]['text']))) for index in
                          range(0, len(text_list))]
        return self.es_save(data_id, embeddings, embedding_list, is_save_function)
    
    def es_save(self, uuid ,embeddings, embedding_list, is_save_function):
        if not self._client.indices.exists(index=self._collection_name):
            dim = len(embeddings[0])    
            mapping = {
                "properties": {
                    "id": { "type": "text" },
                    "source_id": { "type": "text" },
                    "source_type": { "type": "text" },
                    "is_active": { "type": "boolean" },
                    # "dataset": { "type": "object" },
                    "document": { "type": "nested",
                                  	# "properties": 
                                    # { "dataset": {
                                    #     "type": "object",} 
                                    # },
                                },
                    "paragraph": { "type": "nested",
                                  	# "properties": 
                                    # { "dataset": {
                                    #     "type": "object",} 
                                    # },
                                    },
                    "meta": { "type": "nested" },
                    "embedding": { 
                        "type": "dense_vector",
                        "index": True,
                        "dims": dim,
                        "similarity": "l2_norm" 
                    },
                    "search_vector": { "type": "text" }
                }
            }
            
            

            self._client.indices.create(index=self._collection_name, mappings=mapping)
            
        if is_save_function():            
            em = embedding_list[0]
            self._client.index(index=self._collection_name,
                               id=uuid,
                               document={
                                   "id":em.id,
                                   "source_id":em.source_id,
                                   "source_type":em.source_type,
                                   "is_active":em.is_active,
                                   "dataset":em.dataset,
                                   "document":em.document,
                                   "paragraph":em.paragraph,
                                   "meta":em.meta,
                                   "embedding":em.embedding,
                                   "search_vector":em.search_vector,
                               })
        return True
        

    def hit_test(self, query_text, dataset_id_list: list[str], exclude_document_id_list: list[str], top_number: int,
                 similarity: float,
                 search_mode: SearchMode,
                 embedding: Embeddings):
        text_embedding = embedding.embed_query(query_text)
        return self._es_query(query_text, text_embedding, dataset_id_list, exclude_document_id_list, None, True, top_number, similarity, search_mode) 

    def query(self, query_text: str, query_embedding: List[float], dataset_id_list: list[str],
              exclude_document_id_list: list[str],
              exclude_paragraph_list: list[str], is_active: bool, top_n: int, similarity: float,
              search_mode: SearchMode):
        
        return self._es_query(query_text, query_embedding, dataset_id_list, exclude_document_id_list, exclude_paragraph_list, is_active, top_n, similarity, search_mode) 
        
    def _es_query(self, query_text: str, query_embedding: List[float], dataset_id_list: list[str],
              exclude_document_id_list: list[str],
              exclude_paragraph_list: list[str], is_active: bool, top_n: int, similarity: float,
              search_mode: SearchMode):
        if search_mode == SearchMode.embedding:  
            query_str = {
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                    "script": {
                        "source": "Math.max(0, Math.min(1, (cosineSimilarity(params.query_vector, 'embedding') + 1.0) / 2.0))",
                        "params": {
                            "query_vector": query_embedding
                            }
                        }
                    }
                },
                "size": top_n,
                "explain": True
            }
            
            results = self._client.search(index=self._collection_name, body=query_str)
            docs = []
            for hit in results['hits']['hits']:
                if hit['_score'] > similarity:
                    embeding = es_embedding_to_embedding(ESEmbedding(**hit['_source']))
                    embeding.meta['comprehensive_score'] = hit['_score'] 
                    embeding.meta['similarity'] = similarity 
                    docs.append(embeding)
                              
            docs = sorted(docs, key=lambda x: x.meta['similarity'], reverse=True)

            return docs
        
        elif search_mode == SearchMode.keywords:
            
            normalization_str = {
                    "size": 0,  
                    "query": {
                      "nested": {
                        "path": "paragraph",
                        "query": {
                          "bool": {
                            "must": [
                              { "match": { "paragraph.content": query_text } }
                            ]
                          }
                        }
                      }
                    },
                    "aggs": {
                      "max_score": {
                        "max": {
                          "script": {
                            "source": "_score"
                          }
                        }
                      },
                      "min_score": {
                        "min": {
                          "script": {
                            "source": "_score"
                          }
                        }
                    }
                }
            }
            
            normalization_results = self._client.search(index=self._collection_name, body=normalization_str)
            
            max_score = normalization_results['aggregations']['max_score']['value']
            min_score = normalization_results['aggregations']['min_score']['value']
            
            print(max_score, min_score)
            
            query_str = {
                "size": top_n,
                "query": {
                    "nested": {
                        "path": "paragraph", 
                        "query": {
                            "bool": {
                                "must": [
                                    { "match": { 
                                        "paragraph.content": query_text 
                                        }
                                    }
                                ]
                            }
                        }
                    }
                },
                "explain": True       
            }
            
            results = self._client.search(index=self._collection_name, body=query_str)
            results = results['hits']['hits']
            docs = []
            for hit in results:
                if hit['_score'] > similarity:
                    embeding = es_embedding_to_embedding(ESEmbedding(**hit['_source']))  
                    embeding.meta['comprehensive_score'] = round(min([(hit['_score'] - min_score) / (max_score - min_score),1]), 4)  
                    embeding.meta['similarity'] = similarity    
                    docs.append(embeding)
    
            return docs  
          
        elif search_mode == SearchMode.blend:
            normalization_str = {
                    "size": 0,  
                    "query": {
                        "function_score": {
                            "query": {
                               "nested": {
                                "path": "paragraph", 
                                "query": {
                                    "bool": {
                                        "must": [
                                            { "match": { 
                                                "paragraph.content": query_text 
                                                }
                                            }
                                        ]
                                    }
                                }
                            }
                            },
                            "functions": [
                              {
                                "script_score": {
                                  "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding')",
                                    "params": {
                                      "query_vector": query_embedding
                                    }
                                  }
                                }
                              }
                            ],
                            "score_mode": "sum"
                        }
                    },
                    "aggs": {
                      "max_score": {
                        "max": {
                          "script": {
                            "source": "_score"
                          }
                        }
                      },
                      "min_score": {
                        "min": {
                          "script": {
                            "source": "_score"
                          }
                        }
                    }
                }
            }
            
            normalization_results = self._client.search(index=self._collection_name, body=normalization_str)
            
            max_score = normalization_results['aggregations']['max_score']['value']
            min_score = normalization_results['aggregations']['min_score']['value']
            
            query_str =   {
                "query": {
                  "function_score": {
                    "query": {
                       "nested": {
                        "path": "paragraph", 
                        "query": {
                            "bool": {
                                "must": [
                                    { "match": { 
                                        "paragraph.content": query_text 
                                        }
                                    }
                                ]
                            }
                        }
                    }
                    },
                    "functions": [
                      {
                        "script_score": {
                          "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding')",
                            "params": {
                              "query_vector": query_embedding
                            }
                          }
                        }
                      }
                    ],
                    "score_mode": "sum"
                  }
                },
                "size": top_n,
                "explain": True
            }                   
            
            results = self._client.search(index=self._collection_name, body=query_str)
            
            docs = []
            for hit in results['hits']['hits']:
                if hit['_score'] > similarity:
                    embeding = es_embedding_to_embedding(ESEmbedding(**hit['_source']))
                    embeding.meta['comprehensive_score'] = round(min([(hit['_score'] - min_score) / (max_score - min_score),1]), 4) 
                    embeding.meta['similarity'] = similarity 
                    print(hit['_explanation']) 
                    docs.append(embeding)
    
            return docs 
    def delete_by_dataset_id(self, dataset_id: str):
        self._client.delete(index=self._collection_name, id=dataset_id)

    def delete_by_dataset_id_list(self, dataset_id_list: List[str]):
        for dataset_id in dataset_id_list:
            self._client.delete(index=self._collection_name, id=dataset_id)
    
    def delete_by_source_ids(self, source_ids: List[str], source_type: str):
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
        return

    def delete_bu_document_id_list(self, document_id_list: List[str]):
        return

    def delete_by_source_id(self, source_id: str, source_type: str):
        return

    def delete_by_paragraph_id(self, paragraph_id: str):
        return

    def delete_by_paragraph_ids(self, paragraph_ids: List[str]):
        return


class ElasticSearchVectorFactory:
    
    @staticmethod
    def init_vector(attributes: list) -> ESVector:
        collection_name = "es_db"
        return ESVector(
            index_name=collection_name,
            config=ElasticSearchConfig(
                host="127.0.0.1",
                port="9200",
                username="",
                password="",
            ),
            attributes=[]
        )

elasticSearchVectorFactory = ElasticSearchVectorFactory.init_vector([])

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
        # exec_sql, exec_params = generate_sql_by_query_dict({'embedding_query': query_set},
        #                                                    select_string=get_file_content(
        #                                                        os.path.join(PROJECT_DIR, "apps", "embedding", 'sql',
        #                                                                     'embedding_search.sql')),
        #                                                    with_table_name=True)
        # embedding_model = select_list(exec_sql,
        #                               [json.dumps(query_embedding), *exec_params, similarity, top_number])
          
        # elasticSearchVectorFactory.query(query_text, query_embedding, None, None, None, True, top_number, similarity, search_mode)
        elasticSearchVectorFactory.query(query_text,query_embedding,None,None,None,True,top_number,similarity,search_mode)
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