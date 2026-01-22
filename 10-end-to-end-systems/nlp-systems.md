# NLP Systems

## Overview

Natural Language Processing (NLP) systems process and understand human language for tasks like classification, information extraction, generation, and translation. Modern NLP is dominated by transformer-based models (BERT, GPT, T5) that require significant compute resources. NLP systems power chatbots, search, content moderation, and AI assistants.

---

## 🎯 Problem Definition

### Common Use Cases

| Application | Task | Model Type |
|-------------|------|------------|
| **Chatbots** | Generation, Intent | GPT, BERT |
| **Search** | Semantic similarity | Sentence transformers |
| **Moderation** | Classification | BERT, RoBERTa |
| **Translation** | Seq2Seq | T5, mBART |
| **Summarization** | Generation | BART, Pegasus |
| **NER** | Token classification | BERT, spaCy |

### Requirements (Text Classification Example)

| Requirement | Specification |
|-------------|---------------|
| **Latency** | < 100ms p99 |
| **Throughput** | 1K-10K requests/second |
| **Accuracy** | > 90% F1 |
| **Languages** | Multi-lingual support |
| **Text Length** | Up to 10K tokens |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NLP System Architecture                            │
│                                                                               │
│  ┌────────────────┐                                                          │
│  │  Text Input    │                                                          │
│  └───────┬────────┘                                                          │
│          │                                                                    │
│          ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Text Processing Layer                          │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │   Clean &    │──│   Language   │──│   Tokenize   │           │        │
│  │  │   Normalize  │  │   Detect     │  │              │           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  └───────────────────────────────┬─────────────────────────────────┘        │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Model Inference Layer                          │        │
│  │  ┌──────────────────────────────────────────────────────────┐   │        │
│  │  │  Transformer Model (BERT/GPT/T5)                         │   │        │
│  │  │  - Embedding lookup                                       │   │        │
│  │  │  - Attention computation                                  │   │        │
│  │  │  - Task head (classification/generation)                 │   │        │
│  │  └──────────────────────────────────────────────────────────┘   │        │
│  │                                                                  │        │
│  │  Optimization:                                                   │        │
│  │  - KV Cache (generation)                                        │        │
│  │  - Dynamic batching                                              │        │
│  │  - Quantization (INT8)                                          │        │
│  └───────────────────────────────┬─────────────────────────────────┘        │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Post-Processing Layer                          │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │   Decode     │──│   Format     │──│   Filter     │─▶ Output  │        │
│  │  │   Tokens     │  │   Response   │  │   (Safety)   │           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  └─────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Text Preprocessing

```python
from typing import List, Dict, Optional, Tuple
import re
import unicodedata
from dataclasses import dataclass

@dataclass
class TextConfig:
    """Text preprocessing configuration."""
    max_length: int = 512
    lowercase: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_html: bool = True
    normalize_unicode: bool = True

class TextPreprocessor:
    """Preprocess text for NLP models."""
    
    def __init__(self, config: TextConfig):
        self.config = config
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def preprocess(self, text: str) -> str:
        """Clean and normalize text."""
        
        # Remove HTML
        if self.config.remove_html:
            text = self.html_pattern.sub(' ', text)
        
        # Remove URLs
        if self.config.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        # Remove emails
        if self.config.remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        # Normalize unicode
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def truncate(self, text: str, tokenizer) -> str:
        """Truncate text to max tokens."""
        tokens = tokenizer.tokenize(text)
        
        if len(tokens) > self.config.max_length - 2:  # Account for [CLS] and [SEP]
            tokens = tokens[:self.config.max_length - 2]
            text = tokenizer.convert_tokens_to_string(tokens)
        
        return text

class LanguageDetector:
    """Detect language of text."""
    
    def __init__(self):
        from langdetect import detect, detect_langs
        self.detect = detect
        self.detect_langs = detect_langs
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language and confidence."""
        try:
            langs = self.detect_langs(text)
            if langs:
                return langs[0].lang, langs[0].prob
            return "unknown", 0.0
        except:
            return "unknown", 0.0
    
    def is_supported(self, text: str, supported: List[str]) -> bool:
        """Check if text is in supported language."""
        lang, conf = self.detect_language(text)
        return lang in supported and conf > 0.8
```

### 2. Text Classification

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict

class TextClassifier:
    """BERT-based text classifier."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        device: str = "cuda"
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._create_model(model_name, num_labels)
        self.model.to(self.device)
        self.model.eval()
    
    def _create_model(self, model_name: str, num_labels: int) -> nn.Module:
        """Create classification model."""
        
        class BertClassifier(nn.Module):
            def __init__(self, model_name, num_labels):
                super().__init__()
                self.bert = AutoModel.from_pretrained(model_name)
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
            
            def forward(self, input_ids, attention_mask):
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
                pooled = self.dropout(pooled)
                logits = self.classifier(pooled)
                return logits
        
        return BertClassifier(model_name, num_labels)
    
    @torch.no_grad()
    def predict(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """Predict class for texts."""
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoding = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            # Predict
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            # Format results
            for j, (pred, prob) in enumerate(zip(preds, probs)):
                results.append({
                    "predicted_class": pred.item(),
                    "probabilities": prob.cpu().tolist(),
                    "confidence": prob[pred].item()
                })
        
        return results

class MultiLabelClassifier(TextClassifier):
    """Multi-label text classifier."""
    
    @torch.no_grad()
    def predict(
        self,
        texts: List[str],
        threshold: float = 0.5
    ) -> List[Dict]:
        """Predict multiple labels for texts."""
        
        results = []
        
        for text in texts:
            encoding = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            logits = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            
            # Get labels above threshold
            labels = (probs > threshold).squeeze().cpu().tolist()
            
            results.append({
                "labels": labels,
                "probabilities": probs.squeeze().cpu().tolist()
            })
        
        return results
```

### 3. Named Entity Recognition

```python
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class NERSystem:
    """Named Entity Recognition system."""
    
    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        device: str = "cuda"
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_list = self.model.config.id2label
    
    @torch.no_grad()
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text."""
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].tolist()
        
        # Predict
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
        
        # Extract entities
        entities = []
        current_entity = None
        
        for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
            label = self.label_list[pred]
            
            if offset == (0, 0):  # Special token
                continue
            
            if label.startswith("B-"):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "type": label[2:],
                    "start": offset[0],
                    "end": offset[1],
                    "text": text[offset[0]:offset[1]]
                }
            elif label.startswith("I-") and current_entity:
                # Continue entity
                if label[2:] == current_entity["type"]:
                    current_entity["end"] = offset[1]
                    current_entity["text"] = text[current_entity["start"]:offset[1]]
            else:
                # End entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
```

### 4. Text Generation (LLM Serving)

```python
from typing import List, Dict, Optional, AsyncGenerator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio

class TextGenerator:
    """LLM-based text generation with streaming."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda",
        max_length: int = 1024
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.max_length = max_length
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True
    ) -> str:
        """Generate text from prompt."""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens
        ).to(self.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Return only new tokens
        return generated_text[len(prompt):]
    
    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming output."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generated_ids = inputs["input_ids"]
        
        for _ in range(max_new_tokens):
            outputs = self.model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Decode and yield
            token_text = self.tokenizer.decode(next_token[0])
            yield token_text
            
            # Allow other tasks
            await asyncio.sleep(0)

class ConversationManager:
    """Manage multi-turn conversations."""
    
    def __init__(
        self,
        generator: TextGenerator,
        system_prompt: str = "You are a helpful assistant."
    ):
        self.generator = generator
        self.system_prompt = system_prompt
        self.conversations: Dict[str, List[Dict]] = {}
    
    def format_prompt(
        self,
        conversation_id: str,
        user_message: str
    ) -> str:
        """Format conversation history into prompt."""
        
        history = self.conversations.get(conversation_id, [])
        
        prompt_parts = [f"System: {self.system_prompt}"]
        
        for turn in history[-10:]:  # Last 10 turns
            prompt_parts.append(f"User: {turn['user']}")
            prompt_parts.append(f"Assistant: {turn['assistant']}")
        
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    async def chat(
        self,
        conversation_id: str,
        user_message: str
    ) -> str:
        """Process chat message and generate response."""
        
        prompt = self.format_prompt(conversation_id, user_message)
        response = self.generator.generate(prompt, max_new_tokens=512)
        
        # Update history
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "user": user_message,
            "assistant": response
        })
        
        return response
```

### 5. Semantic Search with Embeddings

```python
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    """Semantic search using sentence embeddings."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        vector_store = None
    ):
        self.model = SentenceTransformer(model_name)
        self.vector_store = vector_store  # Pinecone, Milvus, etc.
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    def index_documents(
        self,
        documents: List[Dict],
        batch_size: int = 100
    ):
        """Index documents into vector store."""
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            texts = [d["text"] for d in batch]
            embeddings = self.encode(texts)
            
            # Upsert to vector store
            vectors = [
                {
                    "id": d["id"],
                    "values": emb.tolist(),
                    "metadata": {k: v for k, v in d.items() if k != "text"}
                }
                for d, emb in zip(batch, embeddings)
            ]
            
            self.vector_store.upsert(vectors)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Dict = None
    ) -> List[Dict]:
        """Search for similar documents."""
        
        # Encode query
        query_embedding = self.encode([query])[0]
        
        # Search vector store
        results = self.vector_store.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            filter=filters,
            include_metadata=True
        )
        
        return [
            {
                "id": r.id,
                "score": r.score,
                "metadata": r.metadata
            }
            for r in results.matches
        ]
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Re-rank documents by semantic similarity."""
        
        query_emb = self.encode([query])[0]
        doc_embs = self.encode(documents)
        
        # Compute similarities
        similarities = np.dot(doc_embs, query_emb)
        
        # Rank
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, similarities[idx]) for idx in ranked_indices]
```

### 6. Complete NLP Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI()

class ClassificationRequest(BaseModel):
    texts: List[str]
    model: str = "default"

class ClassificationResponse(BaseModel):
    predictions: List[Dict]
    latency_ms: float

class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    entities: List[Dict]
    latency_ms: float

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    text: str
    latency_ms: float

class NLPService:
    """Complete NLP service."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor(TextConfig())
        self.classifier = TextClassifier()
        self.ner = NERSystem()
        self.generator = TextGenerator()
        self.semantic_search = SemanticSearch()
    
    async def classify(self, request: ClassificationRequest) -> ClassificationResponse:
        """Classify texts."""
        start = time.time()
        
        # Preprocess
        processed = [self.preprocessor.preprocess(t) for t in request.texts]
        
        # Classify
        predictions = self.classifier.predict(processed)
        
        return ClassificationResponse(
            predictions=predictions,
            latency_ms=(time.time() - start) * 1000
        )
    
    async def extract_entities(self, request: NERRequest) -> NERResponse:
        """Extract named entities."""
        start = time.time()
        
        # Preprocess
        processed = self.preprocessor.preprocess(request.text)
        
        # Extract
        entities = self.ner.extract_entities(processed)
        
        return NERResponse(
            entities=entities,
            latency_ms=(time.time() - start) * 1000
        )
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text."""
        start = time.time()
        
        # Generate
        text = self.generator.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return GenerationResponse(
            text=text,
            latency_ms=(time.time() - start) * 1000
        )

# API Endpoints
nlp_service = NLPService()

@app.post("/classify", response_model=ClassificationResponse)
async def classify(request: ClassificationRequest):
    return await nlp_service.classify(request)

@app.post("/ner", response_model=NERResponse)
async def extract_entities(request: NERRequest):
    return await nlp_service.extract_entities(request)

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    return await nlp_service.generate(request)
```

---

## 📈 Metrics & Evaluation

### Model Metrics

| Task | Metric | Target |
|------|--------|--------|
| **Classification** | F1, Accuracy | > 90% |
| **NER** | Entity F1 | > 85% |
| **Generation** | Perplexity, BLEU | Task-specific |
| **Similarity** | MRR, Recall@K | > 80% |

### System Metrics

```python
from prometheus_client import Histogram, Counter

inference_latency = Histogram(
    'nlp_inference_latency_ms',
    'Inference latency',
    ['task', 'model'],
    buckets=[10, 50, 100, 200, 500, 1000, 2000]
)

tokens_processed = Counter(
    'nlp_tokens_processed_total',
    'Total tokens processed',
    ['task']
)

model_errors = Counter(
    'nlp_model_errors_total',
    'Model inference errors',
    ['task', 'error_type']
)
```

---

## ⚖️ Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Model Size** | Large (accurate) | Small (fast) |
| **Precision** | FP32 (accurate) | FP16/INT8 (fast) |
| **Context** | Long context (better) | Short (faster) |
| **Custom vs Pretrained** | Fine-tuned (domain) | Pretrained (general) |

---

## 🎤 Interview Tips

**Common Questions:**
1. How do you handle long documents?
2. How do you optimize transformer inference?
3. How would you build a chatbot system?
4. How do you handle multi-lingual text?
5. How do you evaluate generation quality?

**Key Points:**
- Tokenization is critical
- Batching for throughput
- KV-cache for generation
- Quantization for efficiency
- Prompt engineering for LLMs

---

## 🔗 Related Topics

- [Search Systems](./search-systems.md)
- [Model Optimization](../07-scalability-performance/optimization-techniques.md)
- [Model Serving](../05-model-serving/serving-patterns.md)
