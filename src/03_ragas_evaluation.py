"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# Hai prompt này phải đồng nhất với Bước 2 để so sánh công bằng.
SYSTEM_V1 = """You are a concise AI assistant. Answer the question using only facts stated in the context below. Do not add outside knowledge or unsupported claims. Keep the answer direct and limited to 2-4 sentences. If the context is insufficient, state that clearly.

Context:
{context}"""
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = """You are an AI domain expert. Carefully identify the facts in the context that directly answer the question, then give a clear, well-organized answer in 3-5 sentences. Use only the supplied context, avoid speculation, and explicitly say when the context is insufficient.

Context:
{context}"""
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_PATHS = {
    "v1": PROJECT_ROOT / "data" / "ragas_outputs_v1.json",
    "v2": PROJECT_ROOT / "data" / "ragas_outputs_v2.json",
}
SCORE_PATHS = {
    "v1": PROJECT_ROOT / "data" / "ragas_scores_v1.json",
    "v2": PROJECT_ROOT / "data" / "ragas_scores_v2.json",
}
METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
]


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    # Giữ contexts ở dạng list[str] theo schema của RAGAS.
    # Gợi ý: contexts = [doc.page_content for doc in docs]
    contexts = [doc.page_content for doc in docs]

    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


def save_rag_outputs(rag_results: list, prompt_version: str) -> Path:
    """Lưu answers + contexts để có thể chạy lại RAGAS mà không gọi RAG lần nữa."""
    cache_path = CACHE_PATHS[prompt_version]
    cache_path.write_text(
        json.dumps(rag_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"💾 Đã cache {len(rag_results)} kết quả {prompt_version.upper()} → {cache_path}")
    return cache_path


def load_rag_outputs(prompt_version: str) -> list:
    """Đọc và kiểm tra cache của một prompt version."""
    cache_path = CACHE_PATHS[prompt_version]
    if not cache_path.exists():
        raise FileNotFoundError(f"Không tìm thấy cache: {cache_path}")

    rag_results = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(rag_results, list) or len(rag_results) != len(QA_PAIRS):
        raise ValueError(
            f"Cache {prompt_version.upper()} không hợp lệ: "
            f"cần {len(QA_PAIRS)} samples."
        )

    required_fields = {"question", "reference", "answer", "contexts"}
    for index, result in enumerate(rag_results, 1):
        if not isinstance(result, dict) or not required_fields.issubset(result):
            raise ValueError(
                f"Cache {prompt_version.upper()} thiếu trường ở sample {index}."
            )
        if not isinstance(result["contexts"], list):
            raise ValueError(
                f"Cache {prompt_version.upper()} có contexts sai kiểu ở sample {index}."
            )

    print(f"📂 Đã tải {len(rag_results)} kết quả {prompt_version.upper()} từ cache")
    return rag_results


def save_ragas_scores(scores: dict, prompt_version: str) -> Path:
    """Lưu score từng version để hai lượt RAGAS có thể chạy độc lập."""
    score_path = SCORE_PATHS[prompt_version]
    score_path.write_text(
        json.dumps(scores, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"💾 Đã cache scores {prompt_version.upper()} → {score_path}")
    return score_path


def load_ragas_scores(prompt_version: str) -> dict:
    """Đọc và kiểm tra score cache của một prompt version."""
    score_path = SCORE_PATHS[prompt_version]
    if not score_path.exists():
        raise FileNotFoundError(f"Không tìm thấy score cache: {score_path}")

    scores = json.loads(score_path.read_text(encoding="utf-8"))
    if not isinstance(scores, dict) or not set(METRIC_NAMES).issubset(scores):
        raise ValueError(f"Score cache {prompt_version.upper()} không hợp lệ.")
    for metric in METRIC_NAMES:
        if not np.isfinite(float(scores[metric])):
            raise ValueError(
                f"Score cache {prompt_version.upper()} chứa giá trị không hợp lệ: {metric}."
            )
    print(f"📂 Đã tải scores {prompt_version.upper()} từ cache")
    return {metric: float(scores[metric]) for metric in METRIC_NAMES}


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    # RAGAS cần output dài để trích xuất claims và sinh câu hỏi đánh giá.
    # Một số OpenAI-compatible routers có mặc định quá thấp, khiến
    # finish_reason="length" và phát sinh LLMDidNotFinishException.
    llm_eval.max_tokens = 16_384
    llm_eval.request_timeout = 600
    llm_eval.max_retries = 5
    emb_eval = get_embeddings()

    # 9router hiện chỉ trả một generation dù client yêu cầu n=3. Đặt
    # strictness=1 để RAGAS không yêu cầu ba generations rồi bỏ hai kết quả.
    answer_relevancy.strictness = 1

    # Giảm tải đồng thời lên local router/upstream và cho các evaluator dài
    # thêm thời gian hoàn tất. Retry xử lý các timeout tạm thời.
    run_config = RunConfig(
        timeout=600,
        max_retries=5,
        max_wait=60,
        max_workers=4,
    )

    # Truyền evaluator LLM/embeddings tường minh để dùng đúng provider.
    # Gợi ý:
    #   result = evaluate(
    #       dataset,
    #       metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    #       llm=llm_eval,
    #       embeddings=emb_eval,
    #   )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=run_config,
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in METRIC_NAMES:
        raw = result[key]
        valid_scores = [
            float(value)
            for value in raw
            if value is not None and np.isfinite(float(value))
        ]
        if not valid_scores:
            raise RuntimeError(f"Metric '{key}' không có score hợp lệ.")
        skipped = len(raw) - len(valid_scores)
        if skipped:
            print(f"  ⚠️  {key}: bỏ qua {skipped} sample bị lỗi/NaN")
        scores[key] = float(np.mean(valid_scores))

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


def write_final_report(v1_scores: dict, v2_scores: dict):
    """In so sánh và tạo report sau khi cả V1 và V2 đã được đánh giá."""
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in METRIC_NAMES:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "← V1" if s1 > s2 else "← V2" if s2 > s1 else "Hòa"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = PROJECT_ROOT / "data" / "ragas_report.json"
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    report_path.write_text(report_json, encoding="utf-8")

    evidence_dir = PROJECT_ROOT / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    (evidence_dir / "03_ragas_report.json").write_text(
        report_json,
        encoding="utf-8",
    )

    winners = []
    for metric in METRIC_NAMES:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "V1" if s1 > s2 else "V2" if s2 > s1 else "Hòa"
        winners.append(f"- **{metric}**: V1={s1:.4f}, V2={s2:.4f} — {winner} cao hơn.")
    summary = (
        "# Phân tích kết quả RAGAS\n\n"
        "V1 ưu tiên trả lời ngắn gọn; V2 dùng giọng chuyên gia và tổ chức câu trả lời rõ hơn. "
        "Cả hai prompt đều bị giới hạn chỉ dùng retrieved context nhằm giảm hallucination.\n\n"
        + "\n".join(winners)
        + f"\n\nFaithfulness tốt nhất là **{best_faith:.4f}**. "
        + ("Mục tiêu ≥ 0.8 đã đạt.\n" if best_faith >= 0.8 else "Mục tiêu ≥ 0.8 chưa đạt; cần tinh chỉnh retrieval/prompt.\n")
    )
    (evidence_dir / "README.md").write_text(summary, encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")
    print(f"💾 Đã lưu artifacts vào {evidence_dir}")


# ── 6. Main ────────────────────────────────────────────────────────────────
def rebuild_report_from_score_cache():
    """Tạo lại report chỉ từ hai score cache, không gọi RAG hoặc RAGAS."""
    try:
        v1_scores = load_ragas_scores("v1")
        v2_scores = load_ragas_scores("v2")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"❌ Không thể tạo báo cáo từ score cache: {error}")
        print("   Cần có đủ data/ragas_scores_v1.json và data/ragas_scores_v2.json.")
        sys.exit(1)

    for version, scores in (("V1", v1_scores), ("V2", v2_scores)):
        print(f"\n📊 Kết quả RAGAS — Prompt {version}:")
        for metric, value in scores.items():
            star = " ⭐" if metric == "faithfulness" and value >= 0.8 else ""
            print(f"  {metric:30s}: {value:.4f}{star}")

    write_final_report(v1_scores, v2_scores)


def main(
    eval_only: bool = False,
    collect_only: bool = False,
    eval_version: str = None,
    report_only: bool = False,
):
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if report_only:
        rebuild_report_from_score_cache()
        return

    if not config.validate():
        sys.exit(1)

    if eval_version:
        try:
            version_results = load_rag_outputs(eval_version)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            print(f"❌ Không thể đánh giá {eval_version.upper()}: {error}")
            print("   Hãy chạy: python 03_ragas_evaluation.py --collect-only")
            sys.exit(1)

        version_scores = run_ragas_eval(version_results, eval_version)
        save_ragas_scores(version_scores, eval_version)

        other_version = "v2" if eval_version == "v1" else "v1"
        try:
            other_scores = load_ragas_scores(other_version)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            print(
                f"\n✅ Đã hoàn tất {eval_version.upper()}. "
                f"Tiếp tục bằng: python 03_ragas_evaluation.py --eval-version {other_version}"
            )
            return

        if eval_version == "v1":
            write_final_report(version_scores, other_scores)
        else:
            write_final_report(other_scores, version_scores)
        return

    if eval_only:
        try:
            v1_results = load_rag_outputs("v1")
            v2_results = load_rag_outputs("v2")
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            print(f"❌ Không thể chạy --eval-only: {error}")
            print("   Hãy chạy: python 03_ragas_evaluation.py --collect-only")
            sys.exit(1)
    else:
        vectorstore = setup_vectorstore()

        # Cache ngay sau từng version; lỗi ở evaluation không làm mất dữ liệu RAG.
        v1_results = collect_rag_outputs(vectorstore, "v1")
        save_rag_outputs(v1_results, "v1")
        v2_results = collect_rag_outputs(vectorstore, "v2")
        save_rag_outputs(v2_results, "v2")

    if collect_only:
        print("\n✅ Đã sinh và cache dữ liệu V1/V2. Chưa chạy RAGAS.")
        print("   Tiếp tục bằng: python 03_ragas_evaluation.py --eval-only")
        return

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    save_ragas_scores(v1_scores, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")
    save_ragas_scores(v2_scores, "v2")
    write_final_report(v1_scores, v2_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sinh dữ liệu RAG và/hoặc đánh giá RAGAS."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--collect-only",
        action="store_true",
        help="Chỉ sinh V1/V2 và lưu cache, không chạy RAGAS.",
    )
    mode.add_argument(
        "--eval-only",
        action="store_true",
        help="Chỉ chạy RAGAS từ cache V1/V2 đã lưu.",
    )
    mode.add_argument(
        "--eval-version",
        choices=["v1", "v2"],
        help="Chỉ chạy RAGAS cho một version và lưu score cache.",
    )
    mode.add_argument(
        "--report-only",
        action="store_true",
        help="Chỉ đọc score cache V1/V2 để in và tạo lại report; không gọi LLM/RAGAS.",
    )
    args = parser.parse_args()
    main(
        eval_only=args.eval_only,
        collect_only=args.collect_only,
        eval_version=args.eval_version,
        report_only=args.report_only,
    )
