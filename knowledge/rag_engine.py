# -*- coding: utf-8 -*-
"""
knowledge/rag_engine.py
=======================
RAG（检索增强生成）引擎：基于本地 **Chroma** 向量库实现领域知识的存储与检索，
为大模型注入 PyTorch 调试领域知识，缓解预训练知识过时/不精准的问题。

设计要点
--------
1. **本地化、轻量**：Chroma 持久化到本地磁盘，无需独立服务，数据不出本机，
   符合数据安全要求。
2. **重叠分块**：文档按 512 字符切块、相邻块重叠 128 字符，避免上下文在边界处断裂。
3. **入库前清洗**：去除冗余空白与无效字符，保证向量质量。
4. **稳健降级**：若未安装 chromadb / sentence-transformers，自动降级为
   **本地 TF（词频）检索**，保证演示环境零重型依赖也能召回知识。

对外接口
--------
    engine = RAGEngine()
    engine.add_documents([("来源名", "正文文本"), ...])   # 批量入库（自动分块）
    hits = engine.query("loss 不下降怎么办", top_k=3)      # 检索 Top-K
    engine.stats()                                          # 查看库规模
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from config.config import CONFIG


# ------------------------------------------------------------
# 检索结果
# ------------------------------------------------------------
@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float            # 相关度（越大越相关）

    def format(self) -> str:
        return f"[来源: {self.source} | 相关度: {self.score:.3f}]\n{self.text}"


# ------------------------------------------------------------
# 文本清洗与分块
# ------------------------------------------------------------
def clean_text(text: str) -> str:
    """入库前清洗：归一空白、去除不可见字符。"""
    text = text.replace("　", " ").replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """重叠滑窗分块。

    Args:
        text: 已清洗的全文
        chunk_size: 每块字符数，默认取配置（512）
        overlap: 相邻块重叠字符数，默认取配置（128）
    """
    chunk_size = chunk_size or CONFIG.chunk_size
    overlap = overlap or CONFIG.chunk_overlap
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ============================================================
# RAG 引擎（Chroma 优先 + 本地回退）
# ============================================================
class RAGEngine:
    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        self.persist_dir = Path(persist_dir or CONFIG.paths.vector_store)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = CONFIG.collection_name

        self._backend = "none"
        self._collection = None
        self._embedder = None
        # 本地回退存储
        self._fallback_chunks: List[Tuple[str, str]] = []   # (text, source)
        self._fallback_file = self.persist_dir / "fallback_store.tsv"

        self._init_backend()

    # ---------- 后端初始化 ----------
    def _init_backend(self) -> None:
        """优先 Chroma + sentence-transformers，失败则降级本地 TF。"""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(CONFIG.embedding_model)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._backend = "chroma"
        except Exception as e:  # noqa: BLE001  库缺失或模型下载失败 → 降级
            self._backend = "fallback"
            self._load_fallback()

    # ---------- 入库 ----------
    def add_documents(self, docs: List[Tuple[str, str]]) -> str:
        """批量入库（自动清洗 + 分块）。

        Args:
            docs: [(来源名, 正文), ...]
        Returns:
            入库结果说明
        """
        total_chunks = 0
        for source, content in docs:
            content = clean_text(content)
            chunks = chunk_text(content)
            if not chunks:
                continue
            if self._backend == "chroma":
                self._add_chroma(source, chunks)
            else:
                for c in chunks:
                    self._fallback_chunks.append((c, source))
            total_chunks += len(chunks)
        if self._backend == "fallback":
            self._save_fallback()
        return f"已入库 {len(docs)} 个文档，共 {total_chunks} 个文本块（后端: {self._backend}）"

    def _add_chroma(self, source: str, chunks: List[str]) -> None:
        existing = self._collection.count()
        embeddings = self._embedder.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"{source}_{existing + i}" for i in range(len(chunks))]
        metadatas = [{"source": source} for _ in chunks]
        self._collection.add(
            ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas
        )

    # ---------- 检索 ----------
    def query(self, question: str, top_k: int = None) -> List[RetrievedChunk]:
        """检索与问题最相关的 Top-K 文本块。"""
        top_k = top_k or CONFIG.retrieve_top_k
        if self._backend == "chroma":
            return self._query_chroma(question, top_k)
        return self._query_fallback(question, top_k)

    def _query_chroma(self, question: str, top_k: int) -> List[RetrievedChunk]:
        if self._collection.count() == 0:
            return []
        q_emb = self._embedder.encode([question]).tolist()
        res = self._collection.query(query_embeddings=q_emb, n_results=top_k)
        hits: List[RetrievedChunk] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append(RetrievedChunk(
                text=doc,
                source=(meta or {}).get("source", "未知"),
                score=1.0 - float(dist),     # cosine 距离 → 相似度
            ))
        return hits

    def _query_fallback(self, question: str, top_k: int) -> List[RetrievedChunk]:
        """本地词频/重叠打分检索（无需向量模型）。"""
        if not self._fallback_chunks:
            return []
        q_tokens = set(_tokenize(question))
        scored = []
        for text, source in self._fallback_chunks:
            c_tokens = _tokenize(text)
            if not c_tokens:
                continue
            c_set = set(c_tokens)
            overlap = len(q_tokens & c_set)
            if overlap == 0:
                continue
            # Jaccard 相似 + 命中词频加权
            score = overlap / (len(q_tokens | c_set) + 1e-9)
            scored.append((score, text, source))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(text=t, source=s, score=sc)
            for sc, t, s in scored[:top_k]
        ]

    # ---------- 本地回退持久化 ----------
    def _save_fallback(self) -> None:
        try:
            with open(self._fallback_file, "w", encoding="utf-8") as f:
                for text, source in self._fallback_chunks:
                    safe = text.replace("\t", " ").replace("\n", "\\n")
                    f.write(f"{source}\t{safe}\n")
        except Exception:  # noqa: BLE001
            pass

    def _load_fallback(self) -> None:
        if not self._fallback_file.exists():
            return
        try:
            for line in self._fallback_file.read_text(encoding="utf-8").splitlines():
                if "\t" in line:
                    source, text = line.split("\t", 1)
                    self._fallback_chunks.append((text.replace("\\n", "\n"), source))
        except Exception:  # noqa: BLE001
            pass

    # ---------- 统计 ----------
    def count(self) -> int:
        if self._backend == "chroma":
            return self._collection.count()
        return len(self._fallback_chunks)

    def stats(self) -> str:
        return f"知识库后端: {self._backend} | 文本块数量: {self.count()} | 持久化目录: {self.persist_dir}"

    def reset(self) -> None:
        """清空知识库（重新构建时使用）。"""
        if self._backend == "chroma":
            try:
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )
            except Exception:  # noqa: BLE001
                pass
        else:
            self._fallback_chunks = []
            if self._fallback_file.exists():
                self._fallback_file.unlink()


# ------------------------------------------------------------
# 简易中英文分词（回退检索用）
# ------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    text = text.lower()
    # 英文单词
    en = re.findall(r"[a-z_][a-z0-9_]+", text)
    # 中文按 2-gram 切分（无需分词库）
    zh_chars = re.findall(r"[一-鿿]", text)
    zh_bigrams = ["".join(pair) for pair in zip(zh_chars, zh_chars[1:])]
    return en + zh_chars + zh_bigrams


if __name__ == "__main__":
    engine = RAGEngine()
    print(engine.stats())
    engine.reset()
    print(engine.add_documents([
        ("故障手册", "当训练 loss 不下降时，常见原因包括学习率设置过大或过小、"
                     "数据未归一化、标签错误、模型容量不足。建议先用小学习率验证。"),
        ("PyTorch规范", "分类任务标签应为 LongTensor，使用 CrossEntropyLoss 时无需手动 softmax。"),
        ("显存手册", "CUDA out of memory 时可减小 batch_size、使用梯度累积或混合精度训练。"),
    ]))
    print(engine.stats())
    print("\n检索 'loss 不下降':")
    for h in engine.query("loss 不下降 怎么调", top_k=2):
        print(h.format())
        print("-" * 40)
