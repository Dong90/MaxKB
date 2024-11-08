from abc import ABC, abstractmethod

import numpy as np

from apps.common.config.embedding_config import ModelManage
from apps.common.util.split_model import get_split_model
from apps.setting.models_provider import get_model


class SplitStrategy(ABC):
    @abstractmethod
    def split(self, content: str) -> list:
        pass


class AllInOneSplitStrategy(SplitStrategy):
    def split(self, content: str) -> list:
        text = content
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = text.replace("\0", "")
        return [{"title": "a", "content": text}]


class ByTitleSplitStrategy(SplitStrategy):
    def split(self, content: str) -> list:
        return get_split_model("web.md").parse(content)


class AITitleExtractSplitStrategy(SplitStrategy):
    def split(self, content: str) -> list:
        # 假设这里有一个 AI 提取标题的方法
        return []


class CosineSplitStrategy(SplitStrategy):
    def split(self, content: str) -> list:
        pass
        # paragraphs = combine_sentences(get_split_model("web.md").parse(content))

        # model = get_embedding_model_by_dataset_id(dataset.id)
        # embedding_model_id = dataset.embedding_mode_id
        # embedding_model = ModelManage.get_model(
        #     embedding_model_id, lambda _id: get_model(model)
        # )

        # # 计算每个段落的嵌入向量
        # for i, paragraph in enumerate(paragraphs):
        #     paragraph["embedding"] = embedding_model.embed_query(
        #         paragraph["combined_content"]
        #     )

        # # 计算余弦距离
        # distances, paragraphs = calculate_cosine_distances(paragraphs)

        # if len(distances) == 0:
        #     return paragraphs

        # # 计算距离阈值
        # breakpoint_percentile_threshold = 30
        # breakpoint_distance_threshold = np.percentile(
        #     distances, breakpoint_percentile_threshold
        # )

        # # 获取超过阈值的距离的索引
        # indices_above_thresh = [
        #     i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
        # ]

        # # 初始化起始索引
        # start_index = 0

        # # 创建一个列表来保存分组后的段落
        # chunks = []

        # # 遍历断点以切分段落
        # for index in indices_above_thresh:
        #     # 结束索引为当前断点
        #     end_index = index

        #     # 从当前起始索引到结束索引切分段落
        #     group = paragraphs[start_index : end_index + 1]
        #     combined_text = " ".join([f"{p['title']} {p['content']}" for p in group])
        #     chunks.append(
        #         {"title": f"Chunk {len(chunks) + 1}", "content": combined_text}
        #     )

        #     # 更新起始索引以处理下一组
        #     start_index = index + 1

        # # 处理最后一组，如果有剩余的段落
        # if start_index < len(paragraphs):
        #     combined_text = " ".join(
        #         [f"{p['title']} {p['content']}" for p in paragraphs[start_index:]]
        #     )
        #     chunks.append(
        #         {"title": f"Chunk {len(chunks) + 1}", "content": combined_text}
        #     )

        # return chunks


class SplitStrategyFactory:
    @staticmethod
    def get_strategy(method: str) -> SplitStrategy:
        if method == "all_in_one":
            return AllInOneSplitStrategy()
        elif method == "by_title":
            return ByTitleSplitStrategy()
        elif method == "ai_extract":
            return AITitleExtractSplitStrategy()
        elif method == "cosine":
            return CosineSplitStrategy()
        else:
            raise ValueError(f"Unknown split method: {method}")


def cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def calculate_cosine_distances(paragraphs):
    distances = []
    for i in range(len(paragraphs) - 1):
        embedding_current = paragraphs[i]["embedding"]
        embedding_next = paragraphs[i + 1]["embedding"]
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding_current, embedding_next)
        # Convert to cosine distance
        distance = 1 - similarity
        distances.append(distance)
        # Store distance in the dictionary
        paragraphs[i]["distance_to_next"] = distance
    return distances, paragraphs


def combine_sentences(paragraphs, buffer_size=1):
    combined_sentences = [
        " ".join(
            f"{paragraphs[j]['title']} {paragraphs[j]['content']}"
            for j in range(
                max(i - buffer_size, 0), min(i + buffer_size + 1, len(paragraphs))
            )
        )
        for i in range(len(paragraphs))
    ]
    # 更新原始段落列表，添加组合后的句子
    for i, combined_sentence in enumerate(combined_sentences):
        paragraphs[i]["combined_content"] = combined_sentence
    return paragraphs
