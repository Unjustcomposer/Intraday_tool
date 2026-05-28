import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import hashlib
from collections import OrderedDict
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SentimentEngine:
    """
    FinBERT NLP model for financial news sentiment.
    
    Production fixes:
      - Uses GPU if available to prevent latency spikes
      - Implemented LRU cache for duplicate headlines
      - Supports batch processing
    """
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing FinBERT on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.is_ready = True
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self.is_ready = False
            
        # LRU Cache to avoid recomputing sentiment for same headlines
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_cache_size = 1000

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze single text string"""
        if not self.is_ready or not text:
            return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.0}
            
        # Check cache
        text_hash = hashlib.md5(text.strip().encode()).hexdigest()
        if text_hash in self.cache:
            self.cache.move_to_end(text_hash)
            return self.cache[text_hash]
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # FinBERT classes: 0=positive, 1=negative, 2=neutral
            probs_np = probs.cpu().numpy()[0]
            pred_class = int(torch.argmax(probs, dim=-1).item())
            
            labels = ['positive', 'negative', 'neutral']
            sentiment = labels[pred_class]
            
            # Map to 0-1 score where 1 is highly positive, 0 is highly negative, 0.5 is neutral
            if sentiment == 'positive':
                score = 0.5 + (probs_np[0] * 0.5)
            elif sentiment == 'negative':
                score = 0.5 - (probs_np[1] * 0.5)
            else:
                score = 0.5
                
            result = {
                'sentiment': sentiment,
                'score': float(score),
                'confidence': float(probs_np[pred_class]),
                'raw_probs': probs_np.tolist()
            }
            
            # Update cache
            if len(self.cache) >= self.max_cache_size:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
            self.cache[text_hash] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.0}

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze batch of texts efficiently"""
        if not self.is_ready or not texts:
            return [{'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.0} for _ in texts]
            
        # Check cache first
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.strip().encode()).hexdigest()
            if text_hash in self.cache:
                self.cache.move_to_end(text_hash)
                results[i] = self.cache[text_hash]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                
        if not uncached_texts:
            return results
            
        # Process uncached in batch
        try:
            inputs = self.tokenizer(uncached_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
                
            for idx, text_idx in enumerate(uncached_indices):
                prob = probs[idx]
                pred_class = int(np.argmax(prob))
                labels = ['positive', 'negative', 'neutral']
                sentiment = labels[pred_class]
                
                if sentiment == 'positive':
                    score = 0.5 + (prob[0] * 0.5)
                elif sentiment == 'negative':
                    score = 0.5 - (prob[1] * 0.5)
                else:
                    score = 0.5
                    
                res = {
                    'sentiment': sentiment,
                    'score': float(score),
                    'confidence': float(prob[pred_class]),
                    'raw_probs': prob.tolist()
                }
                
                # Update cache
                text_hash = hashlib.md5(uncached_texts[idx].strip().encode()).hexdigest()
                if len(self.cache) >= self.max_cache_size:
                    self.cache.popitem(last=False)
                self.cache[text_hash] = res
                    
                results[text_idx] = res
                
        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {e}")
            for idx in uncached_indices:
                results[idx] = {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.0}
                
        return results
