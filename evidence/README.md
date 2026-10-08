# Phân tích kết quả RAGAS

V1 ưu tiên trả lời ngắn gọn; V2 dùng giọng chuyên gia và tổ chức câu trả lời rõ hơn. Cả hai prompt đều bị giới hạn chỉ dùng retrieved context nhằm giảm hallucination.

- **faithfulness**: V1=0.8275, V2=0.8014 — V1 cao hơn.
- **answer_relevancy**: V1=0.9037, V2=0.9028 — V1 cao hơn.
- **context_recall**: V1=1.0000, V2=1.0000 — Hòa cao hơn.
- **context_precision**: V1=0.9533, V2=0.9500 — V1 cao hơn.

Faithfulness tốt nhất là **0.8275**. Mục tiêu ≥ 0.8 đã đạt.
