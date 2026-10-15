"""多语言嵌入函数（fastembed + ONNX，无需 PyTorch）"""

import os
import warnings

# Hugging Face 国内镜像（模型下载加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from fastembed import TextEmbedding

# 多语言模型，支持中文 + 英文 + 50 种语言
_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 抑制 fastembed 的 pooling 警告
warnings.filterwarnings("ignore", message="The model.*now uses mean pooling.*")


class MultilingualEmbeddingFunction(EmbeddingFunction[Documents]):
    """支持中英双语的 ChromaDB 嵌入函数"""

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = TextEmbedding(
                model_name=_MODEL_NAME,
                providers=None,
            )
        return self._model

    def __call__(self, input: Documents) -> Embeddings:
        docs = list(input)
        if not docs:
            return []
        return list(self.model.embed(docs))
