# Conversational AI / Chatbot System

## Overview

Conversational AI and chatbot systems automate customer support, improve user experience, and reduce human agent costs. They combine intent classification, dialog management, retrieval-augmented generation (RAG), and safety guardrails to handle diverse user queries—from FAQ lookups to complex task completion. **Modern question asked at both Meta and Google.**

---

## 🎯 Problem Definition

### Business Goals

- **Automate customer support:** Handle 60-80% of routine queries without human agents
- **Improve user experience:** Instant 24/7 responses vs. queue wait times
- **Reduce human agent costs:** Lower cost per conversation while maintaining quality
- **Scale support capacity:** Handle traffic spikes without proportional staff increase
- **Collect insights:** Surface common pain points for product improvement

### Requirements

| Requirement | Specification |
|-------------|---------------|
| **Response Time** | < 2s p95 for first token; < 5s for complete response |
| **Concurrency** | 100K+ concurrent conversations |
| **Context** | Maintain conversation context (multi-turn, session resumption) |
| **Safety** | Block toxic content, prompt injection, PII leakage |
| **Accuracy** | High task completion rate; cite sources for knowledge queries |
| **Containment** | 70%+ conversations resolved without human escalation |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    Conversational AI / Chatbot System Architecture                         │
│                                                                                             │
│  ┌──────────────┐                                                                          │
│  │ User Message │                                                                          │
│  └──────┬───────┘                                                                          │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    SAFETY FILTER (Input)                                           │    │
│  │  [Toxic | Prompt Injection | Jailbreak | PII] → BLOCK / SANITIZE                   │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    INTENT CLASSIFICATION                                           │    │
│  │  [FAQ] [Task/Action] [Chitchat] [Out-of-Scope] [Escalation]                        │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    DIALOG MANAGER (State Tracking)                                 │    │
│  │  Slot filling, conversation memory, session state                                  │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ├─────────────────────────────┬─────────────────────────────┐                      │
│         ▼                             ▼                             ▼                      │
│  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐               │
│  │ FAQ Intent      │        │ Task Intent     │        │ Chitchat Intent │               │
│  │       │         │        │       │         │        │       │         │               │
│  │       ▼         │        │       ▼         │        │       ▼         │               │
│  │ Knowledge       │        │ API Call /      │        │ LLM Generation  │               │
│  │ Retrieval       │        │ Action          │        │ (Direct)        │               │
│  │ (RAG)           │        │ Execution       │        │                 │               │
│  └────────┬────────┘        └────────┬────────┘        └────────┬────────┘               │
│           │                          │                          │                         │
│           └──────────────────────────┼──────────────────────────┘                         │
│                                      │                                                     │
│                                      ▼                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    RESPONSE GENERATION                                            │    │
│  │  [Template] [RAG + LLM] [API result + NL] [LLM with tools]                         │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    SAFETY FILTER (Output)                                          │    │
│  │  [Harmful content | PII | Hallucination] → BLOCK / REDACT                          │    │
│  └──────┬───────────────────────────────────────────────────────────────────────────┘    │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────┐     Low confidence / Frustration / Sensitive topic                      │
│  │   Response   │─────────────────────────────────────▶ ESCALATE TO HUMAN AGENT         │
│  └──────────────┘                                                                          │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Intent Classification

**Hierarchical intent taxonomy:**
```
Root: Support Query
├── FAQ (knowledge lookup)
│   ├── billing
│   ├── shipping
│   ├── returns
│   └── product_usage
├── Task (action-oriented)
│   ├── change_password
│   ├── cancel_subscription
│   ├── track_order
│   └── update_address
├── Chitchat (open-ended)
│   ├── greeting
│   ├── thanks
│   └── small_talk
├── Out-of-scope
└── Escalation_request
```

**Few-shot classification with LLMs vs fine-tuned BERT:**
- **LLM few-shot:** Fast to iterate, no training data; slower and more expensive at scale
- **Fine-tuned BERT:** Lower latency, cheaper; requires labeled data and retraining for new intents

**Fallback detection:** Identify out-of-scope queries (e.g., "What's the weather?") and gracefully deflect or escalate.

### 2. RAG for Knowledge Retrieval

**Document indexing pipeline:**
- Ingest: FAQs, KB articles, policy docs
- Chunking: Semantic chunks (512–1024 tokens) with overlap
- Embedding: Dense vectors (e.g., sentence-transformers)
- Index: Vector store (Pinecone, Milvus, FAISS)

**Hybrid retrieval:**
- **Dense:** Semantic similarity (embeddings)
- **Sparse:** BM25, keyword matching
- **Fusion:** Reciprocal Rank Fusion (RRF) or weighted combination

**Context window management:**
- Fit retrieved docs within model context (e.g., 4K tokens for retrieval)
- Prioritize by relevance score; truncate low-rank chunks

**Citation and grounding:**
- Attach source doc IDs to each retrieved chunk
- Instruct LLM to cite: "According to [Doc X], ..."
- Verify claims against retrieved context to reduce hallucination

### 3. Dialog State Tracking

**Slot filling for task-oriented dialogs:**
- Slots: `order_id`, `reason`, `email`, etc.
- Track filled vs. missing; prompt user for missing slots

**Conversation memory:**
- **Short-term:** Last N turns in session (e.g., 10)
- **Long-term:** User profile, past orders, preferences (from DB)

**Multi-turn context management:**
- Summarize long conversations to fit context window
- Or use sliding window with critical facts preserved

**Session handling and resumption:**
- Session ID; store state in Redis/DB
- Timeout (e.g., 30 min); allow "continue where I left off"

### 4. Safety and Guardrails

**Input safety filter:**
- Toxic content detection
- Prompt injection (e.g., "Ignore previous instructions...")
- Jailbreak attempts
- PII in input (redact or block)

**Output safety filter:**
- Harmful content
- PII leakage
- Hallucination detection (claim not in retrieved context)

**Topic restrictions:**
- Domain boundaries (e.g., support only; refuse medical/legal advice)

**Confidence thresholds:**
- Low confidence → suggest rephrasing or escalate
- Escalation triggers: repeated failure, user frustration keywords, sensitive topics

### 5. Fallback to Human Agent

**Handoff triggers:**
- Low confidence (< 0.7)
- User says "speak to human", "agent", "representative"
- Sensitive topics (refunds above $X, complaints)
- Repeated misunderstood queries

**Context transfer:**
- Summary of conversation, intent, slot values, relevant KB snippets
- Seamless transition UX: "Connecting you to an agent..."

### 6. LLM Orchestration

**Prompt templates:**
- System prompt: role, constraints, format
- User prompt: current turn + context

**Chain-of-thought:** For reasoning-heavy tasks, instruct model to "think step by step"

**Tool use:**
- Function calling: `get_order_status(order_id)`, `cancel_subscription(user_id)`
- Parse tool calls from LLM output; execute; inject results back into prompt

---

## 💻 Python Code

### ChatbotOrchestrator

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any

class Intent(Enum):
    FAQ = "faq"
    TASK = "task"
    CHITCHAT = "chitchat"
    OUT_OF_SCOPE = "out_of_scope"
    ESCALATION = "escalation"

@dataclass
class TurnContext:
    session_id: str
    user_id: str
    message: str
    history: List[Dict[str, str]]
    slot_values: Dict[str, Any]

class ChatbotOrchestrator:
    """Orchestrates the full chatbot pipeline."""
    
    def __init__(
        self,
        intent_classifier: "IntentClassifier",
        rag_retriever: "RAGRetriever",
        safety_filter: "SafetyFilter",
        dialog_tracker: "DialogStateTracker",
        llm_client: Any,
    ):
        self.intent_classifier = intent_classifier
        self.rag_retriever = rag_retriever
        self.safety_filter = safety_filter
        self.dialog_tracker = dialog_tracker
        self.llm_client = llm_client
    
    def process(self, ctx: TurnContext) -> str:
        # 1. Input safety
        if not self.safety_filter.check_input(ctx.message):
            return self.safety_filter.get_safe_refusal_message()
        
        # 2. Intent classification
        intent, confidence = self.intent_classifier.classify(ctx.message, ctx.history)
        
        if confidence < 0.7:
            return "I'm not sure I understood. Could you rephrase, or would you like to speak with an agent?"
        
        if intent == Intent.ESCALATION:
            return self._initiate_handoff(ctx)
        
        if intent == Intent.OUT_OF_SCOPE:
            return "I can only help with [domain] questions. Would you like to speak with an agent?"
        
        # 3. Update dialog state
        self.dialog_tracker.update(ctx)
        
        # 4. Route by intent
        if intent == Intent.FAQ:
            docs = self.rag_retriever.retrieve(ctx.message, top_k=5)
            response = self._generate_rag_response(ctx, docs)
        elif intent == Intent.TASK:
            response = self._execute_task(ctx)
        else:  # Chitchat
            response = self._generate_chitchat_response(ctx)
        
        # 5. Output safety
        if not self.safety_filter.check_output(response):
            return self.safety_filter.get_safe_refusal_message()
        
        return response
    
    def _generate_rag_response(self, ctx: TurnContext, docs: List[Dict]) -> str:
        context = "\n\n".join(d["text"] for d in docs)
        prompt = f"""Answer based ONLY on the context below. Cite sources.

Context:
{context}

User: {ctx.message}

Assistant:"""
        return self.llm_client.generate(prompt, max_tokens=512)
    
    def _execute_task(self, ctx: TurnContext) -> str:
        slots = self.dialog_tracker.get_missing_slots(ctx)
        if slots:
            return f"To help you, I need: {', '.join(slots)}"
        # Execute API / action; format result
        return "Task completed."
    
    def _generate_chitchat_response(self, ctx: TurnContext) -> str:
        return self.llm_client.generate(
            f"User: {ctx.message}\n\nRespond briefly and helpfully.",
            max_tokens=128
        )
    
    def _initiate_handoff(self, ctx: TurnContext) -> str:
        summary = self.dialog_tracker.get_handoff_summary(ctx)
        # Send to human agent queue
        return "Connecting you to an agent. Please hold..."
```

### IntentClassifier

```python
class IntentClassifier:
    """Classifies user intent. Can use BERT fine-tuned or LLM few-shot."""
    
    def __init__(self, model_type: str = "bert", llm_client: Optional[Any] = None):
        self.model_type = model_type
        self.llm_client = llm_client
        # In production: load fine-tuned BERT from registry
        self.bert_model = None
        self.label_map = {i: i.name for i in Intent}
    
    def classify(self, message: str, history: List[Dict]) -> tuple[Intent, float]:
        if self.model_type == "llm":
            return self._classify_llm(message, history)
        return self._classify_bert(message)
    
    def _classify_bert(self, message: str) -> tuple[Intent, float]:
        # Placeholder: BERT forward pass
        # logits = self.bert_model(message)
        # pred = torch.argmax(logits)
        # conf = torch.softmax(logits, dim=-1)[pred].item()
        # return Intent(list(self.label_map.keys())[pred]), conf
        return Intent.FAQ, 0.9  # Placeholder
    
    def _classify_llm(self, message: str, history: List[Dict]) -> tuple[Intent, float]:
        few_shot = """
Examples:
User: How do I reset my password? -> faq
User: I want to cancel my order #12345 -> task
User: Thanks for your help! -> chitchat
User: What's the capital of France? -> out_of_scope
User: I need to talk to a human -> escalation
"""
        prompt = f"{few_shot}\nUser: {message}\nIntent (one word):"
        resp = self.llm_client.generate(prompt, max_tokens=5)
        intent_str = resp.strip().lower()
        intent = Intent.FAQ  # default
        for i in Intent:
            if i.name.lower() == intent_str or intent_str in i.name.lower():
                intent = i
                break
        return intent, 0.85
```

### RAGRetriever

```python
class RAGRetriever:
    """Hybrid retrieval (dense + sparse) for knowledge base."""
    
    def __init__(
        self,
        dense_index: Any,
        sparse_index: Optional[Any],
        embedder: Any,
        fusion_weight: float = 0.5,
    ):
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.embedder = embedder
        self.fusion_weight = fusion_weight
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        # Dense retrieval
        q_embed = self.embedder.encode(query)
        dense_results = self.dense_index.search(q_embed, top_k=top_k * 2)
        
        # Sparse retrieval (BM25)
        sparse_results = []
        if self.sparse_index:
            sparse_results = self.sparse_index.search(query, top_k=top_k * 2)
        
        # Reciprocal Rank Fusion
        fused = self._rrf_fusion(dense_results, sparse_results)
        return fused[:top_k]
    
    def _rrf_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        scores = {}
        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + self.fusion_weight / (k + rank + 1)
        for rank, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + (1 - self.fusion_weight) / (k + rank + 1)
        
        sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])
        doc_map = {d["id"]: d for d in dense_results + sparse_results}
        return [doc_map[did] for did in sorted_ids if did in doc_map]
```

### SafetyFilter

```python
class SafetyFilter:
    """Input and output safety checks."""
    
    def __init__(self, toxicity_model: Any, pii_detector: Any):
        self.toxicity_model = toxicity_model
        self.pii_detector = pii_detector
        self.injection_patterns = [
            r"ignore (previous|all) instructions",
            r"you are (now|a) .* (that|who)",
            r"disregard.*(above|before|prior)",
        ]
    
    def check_input(self, text: str) -> bool:
        if self._is_toxic(text):
            return False
        if self._has_injection(text):
            return False
        return True
    
    def check_output(self, text: str) -> bool:
        if self._is_toxic(text):
            return False
        if self.pii_detector.contains_pii(text):
            return False  # Or redact
        return True
    
    def _is_toxic(self, text: str) -> bool:
        # score = self.toxicity_model.predict_proba(text)[0][1]
        # return score > 0.8
        return False
    
    def _has_injection(self, text: str) -> bool:
        import re
        text_lower = text.lower()
        for pat in self.injection_patterns:
            if re.search(pat, text_lower):
                return True
        return False
    
    def get_safe_refusal_message(self) -> str:
        return "I'm not able to help with that. Is there something else I can assist you with?"
```

### DialogStateTracker

```python
class DialogStateTracker:
    """Tracks slots and conversation state for task-oriented dialogs."""
    
    def __init__(self, slot_definitions: Dict[str, type]):
        self.slot_definitions = slot_definitions
        self.session_states: Dict[str, Dict] = {}
    
    def update(self, ctx: TurnContext) -> None:
        sid = ctx.session_id
        if sid not in self.session_states:
            self.session_states[sid] = {"slots": {}, "history": []}
        
        state = self.session_states[sid]
        state["history"].append({"role": "user", "content": ctx.message})
        if len(state["history"]) > 20:
            state["history"] = state["history"][-20:]
        
        # Slot extraction (simplified; in prod use NER or LLM)
        for slot, dtype in self.slot_definitions.items():
            if slot not in state["slots"]:
                # Extract from message (placeholder)
                extracted = self._extract_slot(ctx.message, slot)
                if extracted:
                    state["slots"][slot] = extracted
    
    def get_missing_slots(self, ctx: TurnContext) -> List[str]:
        state = self.session_states.get(ctx.session_id, {})
        slots = state.get("slots", {})
        return [s for s in self.slot_definitions if s not in slots or slots[s] is None]
    
    def get_handoff_summary(self, ctx: TurnContext) -> str:
        state = self.session_states.get(ctx.session_id, {})
        history = state.get("history", [])
        return "\n".join(
            f"{h['role']}: {h['content'][:100]}" for h in history[-10:]
        )
    
    def _extract_slot(self, message: str, slot: str) -> Optional[Any]:
        # NER or regex; placeholder
        if slot == "order_id" and "#" in message:
            import re
            m = re.search(r"#?(\d{6,})", message)
            return m.group(1) if m else None
        return None
```

---

## 📈 Metrics & Evaluation

| Metric | Description | Target |
|--------|-------------|--------|
| **Task Completion Rate** | % of task intents successfully completed | > 85% |
| **User Satisfaction (CSAT)** | Survey after conversation | > 4.2/5 |
| **Containment Rate** | % resolved without human handoff | > 70% |
| **Avg Turns to Resolution** | Turns per resolved conversation | < 5 |
| **Safety Violation Rate** | Toxic/PII incidents per 1K convos | < 0.1 |
| **Response Latency p95** | Time to first token | < 2s |
| **Intent Accuracy** | Classification accuracy on held-out set | > 95% |

---

## ⚖️ Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Intent Classification** | Fine-tuned BERT (fast, cheap) | LLM few-shot (flexible, no training) |
| **Knowledge Retrieval** | Dense only | Hybrid (dense + sparse) |
| **Context Management** | Sliding window | Summarization |
| **Safety** | Conservative (more blocks) | Permissive (fewer blocks) |
| **Escalation** | Early (lower containment) | Late (higher frustration risk) |
| **RAG vs Pure LLM** | RAG (grounded, citable) | Pure LLM (broader, hallucination risk) |

---

## 🎤 Interview Tips

**Common Questions:**
1. How do you reduce hallucination in chatbot responses?
2. How would you handle 100K concurrent conversations?
3. How do you detect prompt injection and jailbreaks?
4. When should you escalate to a human agent?
5. How do you maintain conversation context across long sessions?
6. How do you evaluate a chatbot (beyond accuracy)?

**Key Points to Mention:**
- Multi-path routing: FAQ (RAG), Task (API), Chitchat (LLM)
- Safety at input and output
- Hybrid retrieval and citation for RAG
- Dialog state tracking for slot filling
- Containment vs. escalation trade-off
- Metrics: task completion, CSAT, containment, safety

---

## 🔗 Related Topics

- [LLM Serving Infrastructure](../../phase-5-advanced-topics/12-llm-genai-systems/01-llm-serving-infrastructure.md)
- [Retrieval-Augmented Generation](../../phase-5-advanced-topics/12-llm-genai-systems/02-retrieval-augmented-generation.md)
- [NLP Systems](./05-nlp-systems.md)
- [Content Moderation](./09-content-moderation.md)
- [Fairness & Responsible AI](../../phase-5-advanced-topics/13-fairness-responsible-ai/04-responsible-deployment.md)
