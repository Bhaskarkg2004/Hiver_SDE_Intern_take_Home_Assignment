# 🚀 Hiver SDE Intern — AI Customer Support Agent

### An end-to-end AI support system built from real customer-support conversations

This project builds an AI customer-support agent for **Amazon Help** using real Twitter support conversations.

Instead of using an LLM as a simple chatbot, the system combines:

**Intent Classification → Semantic Retrieval → Grounded Reply Generation → Human Escalation**

The goal is to answer three practical support questions:

| Question | System Component |
|---|---|
| **What does the customer need?** | Intent classifier |
| **What should we reply?** | Retrieval + LLM generation |
| **Should AI handle this or involve a human?** | Escalation system |

---

## 📁 Repository Structure

```
notebook/    -> hiverAssignment.ipynb (full pipeline: data prep, classification,
                retrieval, reply generation, escalation logic, evaluation, baselines)
GoldenSet/   -> golden_set_final.csv (280 hand-labeled examples with model
                predictions for both intent and escalation)
report/      -> REPORT.md (this report, also included below) and decision log
README.md    -> this file
```

## ⚙️ Setup

1. Open `notebook/hiverAssignment.ipynb` in Google Colab.
2. Mount your Google Drive when prompted (first cell).
3. Add an NVIDIA NIM API key as a Colab secret named `NVIDIA_API_KEY`
   (key icon in the left sidebar → add new secret). NVIDIA's NIM API has a
   free tier and is OpenAI-SDK compatible, which is what this project uses
   for classification, reply generation, and escalation judging.
4. The raw dataset comes from the
   [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
   Kaggle dataset. If reproducing from scratch, download it and place the zip
   in your Drive at the path referenced in the first data-loading cell.

## ▶️ How to Run (reproducible in under 15 minutes)

The notebook is organized into clearly labeled sections (Setup, Data Pipeline,
Golden Set Creation, LLM Setup, Intent Classification, Evaluation, Retrieval,
Reply Generation, Escalation Logic, Baselines) so you can run top to bottom.

**To stay within 15 minutes, two slow steps can be skipped** by loading
pre-computed outputs already saved to Drive instead of recomputing them:

- **Batch intent classification** (looping the LLM over all 280 golden-set
  rows) — skip by loading `golden_set_with_predictions.csv` directly instead
  of re-running the classification loop cell.
- **Corpus embedding** (embedding the 15,000-message retrieval corpus) — skip
  by loading `corpus_embeddings.npy` and `retrieval_corpus.csv` directly
  instead of re-running the `embedder.encode(...)` cell.

With both of these skipped, the remaining cells (evaluation, retrieval
lookup, reply generation, escalation logic, baselines) run in a couple of
minutes total, making the whole notebook reviewable well within 15 minutes.

## 🏷️ Golden Set: Sampling and Labeling Method

280 messages (a buffer above the 250 target) were randomly sampled from the
cleaned, English-filtered Amazon Help customer-to-reply pairs. The sample was
exported to Google Sheets, where each message was manually labeled with an
`intent` (13 categories) and a separate boolean `escalate` flag. The final
labeled file is `GoldenSet/golden_set_final.csv`.

---



---
# AI Customer Support Agent for Amazon Help — Report

## 1. Problem Framing

This project builds an AI-powered customer support agent for Amazon's Twitter support handle, Amazon Help. The system takes an incoming customer message and performs three tasks:

1. **Classifies** the message into one of several support intent categories (e.g. delivery delay, refund request, account access issue).
2. **Drafts a reply** grounded in real historical resolutions — retrieving similar past customer complaints and the actual replies Amazon gave them, rather than generating a response from scratch.
3. **Decides** whether the message can be safely auto-handled or should be escalated to a human agent, along with a stated reason for that decision.

Amazon Help was chosen because of its high message volume and wide variety of realistic support scenarios — delivery issues, billing problems, account access, product questions — which allowed for a genuinely diverse intent taxonomy rather than a narrow one. The scope was deliberately limited to a representative sample of the full ~3M-tweet dataset, since the goal was a working, evaluable pipeline rather than processing the entire corpus.

## 2. Approach

**Data pipeline.** The raw Customer Support on Twitter dataset was filtered to Amazon Help's replies, then paired with the original customer messages that prompted them, producing true conversation pairs rather than isolated tweets. An ASCII-ratio heuristic (>0.9 ASCII characters) was used as a fast English-language filter. Known limitation: it lets through some European Latin-script languages (French, German, Portuguese, Italian, Spanish), which surfaced later as a real source of labeling error (see Failure Modes).

**Intent taxonomy and labeling.** A ten-category taxonomy was developed by sampling real messages:
`delivery_delay, order_not_received, refund_return, damaged_or_wrong_item, account_access, payment_issue, prime_subscription, product_content_request, customer_service_escalation, general_other`

Three additional categories (`technical_issue`, `follow_up_acknowledgment`, `non_english`) were added during labeling once real examples revealed gaps in the original list. A 280-message golden set was hand-labeled with both an `intent` and a separate boolean `escalate` flag, since escalation and intent are related but distinct signals — a message can need escalation regardless of its intent category.

**Classification.** Intent classification uses a free, hosted LLM (via NVIDIA's NIM API) with a zero-shot prompt listing all valid labels. Two models were swapped during development due to availability and latency issues (see Decision Log).

**Retrieval and reply generation.** A sentence-embedding model (`all-MiniLM-L6-v2`) embeds a 15,000-message sample of historical customer messages. For a new message, cosine similarity retrieves the top-k most similar past complaints along with Amazon's real replies, which are given to the LLM as grounding examples for drafting a new, context-appropriate reply. This was manually spot-checked on 5 varied messages (delivery, refund, account access, product question, angry repeat-contact) and produced replies that matched Amazon's real tone without inventing unavailable details (e.g. order numbers).

**Escalation logic.** A layered system:
- Fixed rules escalate high-risk intents (`account_access`, `refund_return`, `customer_service_escalation`) and messages with obvious frustration keywords, without any LLM call.
- Everything else is judged by the LLM based on tone and content, with a required one-sentence reason for every decision.

## 3. Results vs. Baselines

| Metric | Score |
|---|---|
| Intent classification accuracy (LLM, cleaned) | **54.64%** |
| Trivial baseline (always predict most common intent, `delivery_delay`) | 28.21% |
| Keyword-matching baseline | 19.29% |
| Escalation decision accuracy (vs. hand-labeled ground truth) | **69.42%** |

The LLM classifier nearly doubles the trivial baseline and clearly outperforms simple keyword matching. Notably, the keyword baseline underperforms even the trivial baseline — its hand-written rules created false positive matches across overlapping vocabulary and fell back to `general_other` too often. This is reported honestly rather than cherry-picking a flattering baseline.

Per-category performance (F1) for the classifier ranged from strong (`delivery_delay` 0.85, `refund_return` 0.84, `damaged_or_wrong_item` 0.80, `payment_issue` 0.76) to completely broken (`product_content_request` 0.00). See Failure Modes below.

For escalation, precision on "escalate" calls was high (0.79) but recall was moderate (0.54) — the system is conservative and tends to under-escalate rather than over-escalate.

## 4. Top 5 Failure Modes

1. **`product_content_request` complete failure (0% precision/recall).** The classifier never once predicted this label across 15–22 real examples. These messages (e.g. "when will Season 4 be available", "is this product real or fake") lack a strong shared linguistic pattern the way `delivery_delay` messages do, so the model defaults to more common categories instead.

2. **`general_other` used as a dumping ground (15% precision, 53% recall).** The model over-relies on this catch-all category whenever it's uncertain, meaning a majority of what gets labeled `general_other` doesn't actually belong there.

3. **Escalation under-recall (54% recall on true escalations).** The system misses roughly half of messages a human labeled as needing escalation, while rarely raising false alarms (79% precision). In a real deployment this asymmetry is risky — a missed escalation is usually costlier than an unnecessary one — and would need tuning toward higher recall.

4. **Non-English language leakage from the ASCII filter.** Investigation of `product_content_request` errors revealed that ~7 of ~19 "English" messages in that category were actually French, German, Portuguese, Italian, or Spanish. The ASCII-ratio filter cannot distinguish these from English since they use mostly Latin characters. This was a genuine pipeline flaw, not a labeling mistake, and was corrected in the final golden set (relabeled to `non_english`), improving intent accuracy from 53.93% to 54.64%.

5. **Raw model output required cleaning before scoring.** Some model outputs leaked internal "thinking" text (e.g. `general\n</think>delivery_delay`) even after a `chat_template_kwargs: {enable_thinking: False}` flag was set, and inconsistent whitespace caused `prime_subscription` to be silently split into two different label classes during scoring. Both required a post-processing cleanup step before evaluation could be trusted.

## 5. What's Misleading About My Headline Number

The headline "54.64% intent accuracy" understates how good the system is on the categories that matter most by volume, and overstates how good it is overall.

- **It hides bimodal performance.** Categories that make up the bulk of real traffic (delivery delay, refunds, damaged items, payment issues) score 76–85% F1 — genuinely strong, production-plausible performance. But one category (`product_content_request`) is at literal zero, and `general_other` is being systematically misused. A single blended accuracy number treats these as equally weighted, which misrepresents the system's real-world reliability on common cases.

- **It doesn't reflect the cost of different mistakes.** A message misclassified as `general_other` when it was actually a low-stakes product question is a mild annoyance. A message that should have triggered `customer_service_escalation` but got auto-handled instead (the ~46% of missed true escalations) is a materially more costly error. Accuracy treats both mistakes identically.

- **It doesn't account for known-fixable pipeline issues.** Part of the errors (the non-English leakage) came from a labeling artifact caused by an imperfect language filter, not from a genuine model failure. Correcting even this one known issue moved the number from 53.93% to 54.64%; there is likely more "fixable" error mixed into the reported number that a more careful audit would separate from genuine model confusion.

- **It says nothing about reply quality or escalation quality**, which are arguably more important to the end user than the intent label itself. The intent label is an internal routing signal; the customer only ever sees the reply and whether they got a human.

## 6. Next-Week Plan

If given another week, priorities would be:

1. **Fix `product_content_request`.** Either provide few-shot examples in the prompt specifically for this category, or reconsider whether it should be split into more distinguishable subcategories.
2. **Tune escalation toward higher recall.** Given the asymmetric cost of missed escalations, adjust the rule thresholds and/or prompt to be more conservative (escalate more readily) and re-measure the precision/recall trade-off.
3. **Build the LLM-as-judge evaluation for reply quality**, with a human-agreement check on a sample of the judge's own scores — this was scoped but not completed due to time constraints.
4. **Audit and separate "fixable" pipeline errors from genuine model confusion** — e.g. replace the ASCII-ratio language filter with a proper language-detection library (e.g. `langdetect` or `fasttext`) and re-score the golden set.
5. **Expand the golden set** for underrepresented categories (`prime_subscription`, `refund_return`, `damaged_or_wrong_item` each have under 15 examples) to get more statistically reliable per-category metrics.

---

# Decision Log

1. **Brand: Amazon Help.** Chosen for high message volume and diversity of realistic support scenarios (delivery, billing, account, product questions), enabling a richer intent taxonomy than a narrower brand would allow.
2. **ASCII-ratio (>0.9) used as English-language filter.** Fast and simple, but known to let through European Latin-script languages (French, German, Portuguese, Italian, Spanish). This limitation was documented up front and later confirmed as a real source of downstream labeling error.
3. **Taxonomy grew from 10 to 13 categories during labeling.** `technical_issue`, `follow_up_acknowledgment`, and `non_english` were added once real golden-set examples didn't fit the original 10, rather than forcing them into ill-fitting buckets.
4. **Escalation modeled as both its own intent category (`customer_service_escalation`) AND a separate boolean `escalate` column.** This is intentional: a message can be about, say, a refund (intent) while also independently needing escalation (the flag), so the two signals are not redundant but do overlap and required clear documentation.
5. **Subsampled the dataset rather than using the full ~3M tweets.** 280 messages for the golden set, 15,000 for the retrieval corpus — chosen to keep iteration and reproducibility fast (<15 min) without materially harming quality, since even a 15k sample gave strong retrieval results (0.76–0.79 cosine similarity on test queries).
6. **NVIDIA NIM API chosen over paid APIs (OpenAI/Anthropic)** to keep the project cost-free, using their generous free-tier hosted models via an OpenAI-compatible client.
7. **API key stored via Colab Secrets, not hardcoded**, to keep the notebook safe to make public in the repo.
8. **Model swapped multiple times during development:**
   - `meta/llama-3.1-8b-instruct` → failed with HTTP 410 (model reached end-of-life 2026-08-26).
   - `nvidia/nemotron-3.5-lightning-30b-a3b` → worked but defaulted to an internal "thinking mode" that leaked reasoning text into outputs, and was very slow (~15s/call), making the full batch run impractical.
   - `mistralai/mistral-nemotron` → ~8x faster (~1.8s/call), used for the golden-set batch classification run.
   - `nemotron-3.5-lightning` was later re-adopted for reply generation and escalation judging (with `enable_thinking: False` explicitly set) once its latency was found acceptable for lower-volume calls.
9. **Raw model outputs required a cleanup step before scoring** — stripping leaked `</think>` tags, trimming whitespace, and fuzzy-matching against the known label set — since evaluating raw text directly silently under-counted accuracy and fragmented categories (e.g. `prime_subscription` vs. `prime_subscription ` counted as different classes).
10. **Corrected 7 golden-set labels from `product_content_request` to `non_english`** after failure analysis revealed they were mislabeled non-English messages that slipped through the ASCII filter — a transparent, documented fix rather than silently improving the number.
11. **Escalation logic designed as layered rules + LLM**, not a single LLM call for every message: fixed rules handle high-risk intents and obvious frustration keywords cheaply and deterministically; the LLM is only invoked for genuinely ambiguous cases, reducing cost and latency.
12. **Chose two deliberately different baselines** (trivial most-frequent-class, and keyword matching) rather than one, to show the system beats both a "do nothing clever" baseline and a "simple heuristic" baseline — and reported honestly that the keyword baseline underperformed even the trivial one.
13. **Embeddings and retrieval kept lightweight** (sentence-transformers + cosine similarity in-memory) rather than standing up a full vector database, appropriate for a 15k-document corpus and a take-home timeline.
