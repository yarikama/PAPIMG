# PAPIMG
**Prompt-Aware Pruning and Intent-aware Multimodal Guardrails**

> Chao Hsuan Ho (ch218) · Heng Jui Hsu (hh83) · Kerstin Sun (ks256)

---

## Motivation

Modern Multimodal Large Language Models (MLLMs) face a significant computational bottleneck when processing high-resolution images — the resulting explosion of visual tokens leads to prohibitive latency and high inference costs. While much of visual data is often redundant or task-irrelevant noise (e.g., background scenery when the task is counting fingers), standard **Global Pruning** techniques discard information indiscriminately, risking the loss of critical details.

Furthermore, existing pruning strategies frequently lack **intent-awareness**, posing a risk where safety-critical elements — such as pedestrians or hazard signs — are accidentally removed simply because they were not explicitly mentioned in the user's prompt.

This project develops a system that:
1. **Dynamically selects visual tokens** based on semantic relevance to the user's prompt
2. **Retains safety-critical information** through a dedicated Risk Awareness module

This ensures that the pursuit of efficiency does not come at the expense of safety and reliability.

---

## Architecture

```
Raw Image  →  ViT  →  Patch Tokens  V = {v1 ... vn}
Raw Text   →  CLIP Text Encoder  →  Text Embedding T
                          ↓
             Cross-Modal Cosine Similarity
                  +  Risk Awareness Module
                          ↓
             TopK/TopP Selection + Spatial Anchoring
                          ↓
             V_pruned  →  MLP Projection  →  LLaVA
```

### Stage 1 — Text and Vision Extraction

| Modality | Input | Process | Output |
|---|---|---|---|
| Text | Raw text query | Pretrained Transformer (e.g., CLIP Text Encoder) | Text Embedding `T` |
| Vision | Raw RGB image | ViT: divide image into N patches and project to embeddings | Patch Tokens `V = {v1 ... vn}` |

### Stage 2 — Cross-Modal Scoring & Pruning

1. **Cosine Similarity Scoring**: Compute dot product between `T` and every token in `V` to produce a Saliency Map
2. **Risk Scoring**: Safety prototype matching to flag safety-critical patches
3. **TopK / TopP Selection**: Retain the most task-relevant and safety-relevant tokens
4. **Spatial Anchoring**: Preserve spatial structure of surviving tokens
5. **Re-packing**: Compact the reduced token set

**Output**: Importance Map (normalized) + reduced token set `V_pruned`

### Stage 3 — LLM Inference

- Pass `V_pruned` through the remaining Vision Transformer layers
- **Modality Alignment**: Project filtered image tokens to LLaVA embedding space via MLP
- Perform final task (VQA, object detection, segmentation, etc.)

---

## Safety Guardrails

A key novelty of PAPIMG is that the Risk Awareness module **overrides efficiency-driven pruning** when patches are flagged as safety-relevant — even if they score low on prompt relevance. This decouples *task relevance* from *safety relevance*.

| Image Content | User Intent | Expected Behavior | Mechanism |
|---|---|---|---|
| Benign | Standard query | Accurate & low-latency response | Prompt-Aware Pruning: retains task-relevant patches, reducing FLOPs |
| Sensitive | Standard query (e.g., "What is this logo?") | Refusal of sensitive components | Safety Guardrail: identifies high-risk tokens via safety prototypes and applies masking |
| Benign | Adversarial / malicious query | Refusal of response | Safety-Aware Modulation: detects intent-risk alignment and suppresses visual token saliency |
| Jailbreak image | Hidden malicious instructions | Ignore / bypass hidden commands | Token Neutralization: identifies instruction-laden patches as task-irrelevant / high-risk and prunes them |

---

## Datasets

### GQA
Complex questions requiring multi-step spatial and semantic reasoning. Used to evaluate the trade-off between pruning aggressiveness and reasoning accuracy.

### VizWiz
Originating from visually impaired users, this dataset contains real-world, often cluttered or low-quality images. Used to assess the model's ability to recognize precise details in noisy environments — a stress test for over-aggressive pruning.

---

## Evaluation Plan

The project evaluates trade-offs between **reasoning efficiency** and **safety** across both datasets using:
- **Accuracy** on VQA tasks (vs. unpruned baseline)
- **Computational cost** (FLOPs, latency, token retention rate)
- **Safety compliance rate** across the four threat scenarios above
