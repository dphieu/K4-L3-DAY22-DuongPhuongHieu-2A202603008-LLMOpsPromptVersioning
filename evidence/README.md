# Phân tích kết quả RAGAS

## Thông tin bài nộp

- GitHub repository: https://github.com/dphieu/K4-L3-DAY22-DuongPhuongHieu-2A202603008-LLMOpsPromptVersioning
- LangSmith project: https://smith.langchain.com/o/383fad7a-5cad-4f8c-9357-1905f6a84688/projects/p/e2a2c6c0-c950-4aa1-b148-503ee5375223
- Số root traces đã kiểm tra ngày 08/10/2026: **708**.

V1 ưu tiên trả lời ngắn gọn; V2 dùng giọng chuyên gia và tổ chức câu trả lời rõ hơn. Cả hai prompt đều bị giới hạn chỉ dùng retrieved context nhằm giảm hallucination.

- **faithfulness**: V1=0.8275, V2=0.8014 — V1 cao hơn.
- **answer_relevancy**: V1=0.9037, V2=0.9028 — V1 cao hơn.
- **context_recall**: V1=1.0000, V2=1.0000 — Hòa cao hơn.
- **context_precision**: V1=0.9533, V2=0.9500 — V1 cao hơn.

Faithfulness tốt nhất là **0.8275**. Mục tiêu ≥ 0.8 đã đạt.
