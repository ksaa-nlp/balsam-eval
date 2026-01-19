"""
Universal MTEB Evaluation Framework
Supports: Local models, HuggingFace models, API models, and custom tasks
"""

import mteb
import numpy as np
import requests
import torch
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from datasets import Dataset
import time
import os
import hashlib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================================
# BASE MODEL INTERFACE
# ============================================================================

class BaseModel(ABC):
    """Base class for all model types"""
    
    @abstractmethod
    def encode(self, sentences: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Encode sentences into embeddings.
        
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (len(sentences), embedding_dim)
        """
        pass


# ============================================================================
# LOCAL MODEL (PyTorch/Custom)
# ============================================================================

class LocalModel(BaseModel):
    """Wrapper for local PyTorch models"""
    
    def __init__(self, model, tokenizer=None, device=None, normalize=True):
        """
        Args:
            model: Your PyTorch model
            tokenizer: Your tokenizer (optional)
            device: Device to use ('cuda', 'cpu', or None for auto)
            normalize: Whether to normalize embeddings
        """
        self.model = model
        self.tokenizer = tokenizer
        self.normalize = normalize
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, sentences: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode sentences using local model"""
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                
                # Tokenize
                if self.tokenizer:
                    inputs = self.tokenizer(
                        batch, 
                        padding=True, 
                        truncation=True, 
                        return_tensors='pt',
                        max_length=512
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get embeddings
                    outputs = self.model(**inputs)
                    
                    # Mean pooling
                    embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                else:
                    # Assume model handles raw text
                    embeddings = self.model(batch)
                
                # Normalize if requested
                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# ============================================================================
# HUGGINGFACE MODEL
# ============================================================================

class HuggingFaceModel(BaseModel):
    """Wrapper for HuggingFace models using sentence-transformers"""
    
    def __init__(self, model_name: str, device: Optional[str] = None, normalize: bool = True):
        """
        Args:
            model_name: HuggingFace model name or path
            device: Device to use
            normalize: Whether to normalize embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        self.model_name = model_name
        self.normalize = normalize
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = SentenceTransformer(model_name, device=device)
    
    def encode(self, sentences: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode using HuggingFace model"""
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            **kwargs
        )
        return embeddings


# ============================================================================
# API MODEL (Generic)
# ============================================================================

class APIModel(BaseModel):
    """Generic wrapper for API-based models"""
    
    def __init__(
        self, 
        api_url: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        rate_limit_delay: float = 0.1,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            api_url: API endpoint URL
            api_key: API key for authentication
            model_name: Model name to use
            headers: Additional headers
            max_retries: Max retry attempts
            rate_limit_delay: Delay between requests (seconds)
            cache_dir: Directory for caching embeddings
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.rate_limit_delay = rate_limit_delay
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Setup headers
        self.headers = headers or {}
        self.headers["Content-Type"] = "application/json"
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def encode(self, sentences: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode using API with caching support"""
        if self.cache_dir:
            return self._encode_with_cache(sentences, batch_size, **kwargs)
        else:
            return self._encode_no_cache(sentences, batch_size, **kwargs)
    
    def _encode_no_cache(self, sentences: List[str], batch_size: int, **kwargs) -> np.ndarray:
        """Encode without caching"""
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            embeddings = self._encode_batch(batch, **kwargs)
            all_embeddings.extend(embeddings)
            time.sleep(self.rate_limit_delay)
        
        return np.array(all_embeddings)
    
    def _encode_with_cache(self, sentences: List[str], batch_size: int, **kwargs) -> np.ndarray:
        """Encode with caching"""
        embeddings = []
        to_encode = []
        to_encode_indices = []
        
        # Check cache
        for idx, sentence in enumerate(sentences):
            cached = self._get_cached_embedding(sentence)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                to_encode.append(sentence)
                to_encode_indices.append(idx)
        
        # Encode uncached sentences
        if to_encode:
            new_embeddings = self._encode_no_cache(to_encode, batch_size, **kwargs)
            
            # Update cache and results
            for idx, embedding in zip(to_encode_indices, new_embeddings):
                embeddings[idx] = embedding
                self._save_to_cache(sentences[idx], embedding)
        
        return np.array(embeddings)
    
    def _encode_batch(self, sentences: List[str], **kwargs) -> List[np.ndarray]:
        """
        Encode a batch via API. Override this method for custom API formats.
        Default format assumes OpenAI-like API.
        """
        payload = {
            "input": sentences,
        }
        
        if self.model_name:
            payload["model"] = self.model_name
        
        payload.update(kwargs)
        
        try:
            response = self.session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract embeddings (OpenAI format)
            return [item["embedding"] for item in data["data"]]
            
        except Exception as e:
            print(f"API Error: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
        
        if os.path.exists(cache_file):
            return np.load(cache_file)
        return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to cache"""
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
        np.save(cache_file, embedding)


# ============================================================================
# OPENAI MODEL
# ============================================================================

class OpenAIModel(BaseModel):
    """OpenAI embeddings API wrapper"""
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def encode(self, sentences: List[str], batch_size: int = 100, **kwargs) -> np.ndarray:
        """Encode using OpenAI API"""
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name
            )
            
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)


# ============================================================================
# COHERE MODEL
# ============================================================================

class CohereModel(BaseModel):
    """Cohere embeddings API wrapper"""
    
    def __init__(self, api_key: str, model_name: str = "embed-english-v3.0", input_type: str = "search_document"):
        try:
            import cohere
        except ImportError:
            raise ImportError("Please install cohere: pip install cohere")
        
        self.client = cohere.Client(api_key)
        self.model_name = model_name
        self.input_type = input_type  # "search_document", "search_query", "classification"
    
    def encode(self, sentences: List[str], batch_size: int = 96, **kwargs) -> np.ndarray:
        """Encode using Cohere API"""
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            response = self.client.embed(
                texts=batch,
                model=self.model_name,
                input_type=self.input_type
            )
            
            all_embeddings.extend(response.embeddings)
        
        return np.array(all_embeddings)


# ============================================================================
# GOOGLE VERTEX AI MODEL
# ============================================================================

class GoogleVertexAIModel(BaseModel):
    """Google Vertex AI embeddings wrapper"""
    
    def __init__(
        self, 
        project_id: str,
        location: str = "us-central1",
        model_name: str = "text-embedding-004",
        credentials_path: Optional[str] = None
    ):
        """
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            model_name: Model name (text-embedding-004, text-multilingual-embedding-002, etc.)
            credentials_path: Path to service account JSON file
        """
        try:
            from vertexai.language_models import TextEmbeddingModel
            import vertexai
        except ImportError:
            raise ImportError("Please install google-cloud-aiplatform: pip install google-cloud-aiplatform")
        
        # Initialize Vertex AI
        if credentials_path:
            import os
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        vertexai.init(project=project_id, location=location)
        
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.model_name = model_name
    
    def encode(self, sentences: List[str], batch_size: int = 5, **kwargs) -> np.ndarray:
        """
        Encode using Google Vertex AI.
        Note: Vertex AI has a limit of 5 texts per request for some models
        """
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Get embeddings
            embeddings = self.model.get_embeddings(batch)
            
            # Extract values
            batch_embeddings = [emb.values for emb in embeddings]
            all_embeddings.extend(batch_embeddings)
            
            time.sleep(0.1)  # Rate limiting
        
        return np.array(all_embeddings)


# ============================================================================
# GOOGLE GENERATIVE AI MODEL (Gemini API)
# ============================================================================

class GoogleGenAIModel(BaseModel):
    """Google Generative AI (Gemini) embeddings wrapper"""
    
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        """
        Args:
            api_key: Google API key
            model_name: Model name (models/text-embedding-004, models/embedding-001, etc.)
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.genai = genai
    
    def encode(self, sentences: List[str], batch_size: int = 100, **kwargs) -> np.ndarray:
        """Encode using Google Generative AI"""
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Embed content
            result = self.genai.embed_content(
                model=self.model_name,
                content=batch,
                task_type="retrieval_document"  # or "retrieval_query", "semantic_similarity", "classification"
            )
            
            all_embeddings.extend(result['embedding'])
            time.sleep(0.1)  # Rate limiting
        
        return np.array(all_embeddings)


# ============================================================================
# VOYAGE AI MODEL (Anthropic's Recommended Embedding Provider)
# ============================================================================

class VoyageAIModel(BaseModel):
    """Voyage AI embeddings wrapper - Anthropic's recommended embedding provider"""
    
    def __init__(self, api_key: str, model_name: str = "voyage-3"):
        """
        Args:
            api_key: Voyage AI API key (get from https://www.voyageai.com/)
            model_name: Model name (voyage-3, voyage-3-lite, voyage-code-3, etc.)
        """
        try:
            import voyageai
        except ImportError:
            raise ImportError("Please install voyageai: pip install voyageai")
        
        self.client = voyageai.Client(api_key=api_key)
        self.model_name = model_name
    
    def encode(self, sentences: List[str], batch_size: int = 128, **kwargs) -> np.ndarray:
        """Encode using Voyage AI"""
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            result = self.client.embed(
                texts=batch,
                model=self.model_name,
                input_type="document"  # or "query"
            )
            
            all_embeddings.extend(result.embeddings)
            time.sleep(0.1)
        
        return np.array(all_embeddings)


# ============================================================================
# CLAUDE-BASED EMBEDDING MODEL (Experimental - Not Recommended)
# ============================================================================

class ClaudeEmbeddingModel(BaseModel):
    """
    Experimental: Use Claude to generate embeddings through prompting.
    
    NOTE: This is NOT recommended for production use because:
    1. Claude is not designed for embeddings
    2. Very slow and expensive
    3. Inconsistent results
    4. Use Voyage AI instead (Anthropic's recommended provider)
    
    This is only for educational/experimental purposes.
    """
    
    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20241022"):
        """
        Args:
            api_key: Anthropic API key
            model_name: Claude model name
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        
        print("\n" + "="*80)
        print("WARNING: Using Claude for embeddings is experimental and NOT recommended!")
        print("For production use, please use Voyage AI (Anthropic's recommended provider)")
        print("="*80 + "\n")
    
    def encode(self, sentences: List[str], batch_size: int = 1, embedding_dim: int = 768, **kwargs) -> np.ndarray:
        """
        Generate embeddings by asking Claude to create numerical representations.
        This is very slow and not recommended - use Voyage AI instead!
        """
        all_embeddings = []
        
        for i, sentence in enumerate(sentences):
            if i % 10 == 0:
                print(f"Processing {i}/{len(sentences)} sentences...")
            
            try:
                # Create a prompt asking Claude to generate a numerical representation
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2048,
                    messages=[{
                        "role": "user",
                        "content": f"""Generate a {embedding_dim}-dimensional numerical embedding vector for this text: "{sentence}"

Return ONLY a Python list of {embedding_dim} floating point numbers between -1 and 1, nothing else.
Example format: [0.23, -0.45, 0.67, ...]"""
                    }]
                )
                
                # Parse the response
                response_text = message.content[0].text.strip()
                
                # Try to extract the list
                import ast
                embedding = ast.literal_eval(response_text)
                
                if len(embedding) != embedding_dim:
                    # Pad or truncate
                    if len(embedding) < embedding_dim:
                        embedding = embedding + [0.0] * (embedding_dim - len(embedding))
                    else:
                        embedding = embedding[:embedding_dim]
                
                all_embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error processing sentence: {e}")
                # Return random embedding as fallback
                all_embeddings.append(np.random.randn(embedding_dim).tolist())
            
            time.sleep(1)  # Rate limiting
        
        return np.array(all_embeddings)


# ============================================================================
# CUSTOM TASKS
# ============================================================================

class CustomClassificationTask(mteb.AbsTaskClassification):
    """Custom classification task template"""
    
    metadata = {
        "name": "CustomClassification",
        "description": "Custom classification task",
        "type": "Classification",
        "category": "s2s",
        "eval_splits": ["train", "test"],
        "eval_langs": ["eng-Latn"],
        "main_score": "accuracy",
    }
    
    def __init__(self, data: Dict[str, Any], task_name: str = "CustomClassification", **kwargs):
        """
        Args:
            data: Dictionary with 'train' and 'test' keys containing datasets
                  Each dataset should have 'text' and 'label' columns
            task_name: Name of the task
        """
        super().__init__(**kwargs)
        self.data = data
        self.metadata["name"] = task_name
    
    def load_data(self, **kwargs):
        """Load custom data"""
        datasets = {}
        
        for split in self.metadata["eval_splits"]:
            if split in self.data:
                datasets[split] = Dataset.from_dict(self.data[split])
        
        return datasets


class CustomClusteringTask(mteb.AbsTaskClustering):
    """Custom clustering task template"""
    
    metadata = {
        "name": "CustomClustering",
        "description": "Custom clustering task",
        "type": "Clustering",
        "category": "s2s",
        "eval_splits": ["test"],
        "eval_langs": ["eng-Latn"],
        "main_score": "v_measure",
    }
    
    def __init__(self, data: Dict[str, Any], task_name: str = "CustomClustering", **kwargs):
        """
        Args:
            data: Dictionary with 'test' key containing dataset
                  Dataset should have 'sentences' and 'labels' columns
            task_name: Name of the task
        """
        super().__init__(**kwargs)
        self.data = data
        self.metadata["name"] = task_name
    
    def load_data(self, **kwargs):
        """Load custom data"""
        datasets = {}
        
        for split in self.metadata["eval_splits"]:
            if split in self.data:
                datasets[split] = Dataset.from_dict(self.data[split])
        
        return datasets


class CustomRetrievalTask(mteb.AbsTaskRetrieval):
    """Custom retrieval task template"""
    
    metadata = {
        "name": "CustomRetrieval",
        "description": "Custom retrieval task",
        "type": "Retrieval",
        "category": "s2s",
        "eval_splits": ["test"],
        "eval_langs": ["eng-Latn"],
        "main_score": "ndcg_at_10",
    }
    
    def __init__(self, corpus: Dict, queries: Dict, qrels: Dict, task_name: str = "CustomRetrieval", **kwargs):
        """
        Args:
            corpus: Dict of {doc_id: {"text": doc_text}}
            queries: Dict of {query_id: query_text}
            qrels: Dict of {query_id: {doc_id: relevance_score}}
            task_name: Name of the task
        """
        super().__init__(**kwargs)
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.metadata["name"] = task_name
    
    def load_data(self, **kwargs):
        """Load custom data"""
        return {
            "test": {
                "corpus": self.corpus,
                "queries": self.queries,
                "qrels": self.qrels
            }
        }


class CustomSTSTask(mteb.AbsTaskSTS):
    """Custom Semantic Textual Similarity task template"""
    
    metadata = {
        "name": "CustomSTS",
        "description": "Custom STS task",
        "type": "STS",
        "category": "s2s",
        "eval_splits": ["test"],
        "eval_langs": ["eng-Latn"],
        "main_score": "cosine_spearman",
    }
    
    def __init__(self, data: Dict[str, Any], task_name: str = "CustomSTS", **kwargs):
        """
        Args:
            data: Dictionary with 'test' key containing dataset
                  Dataset should have 'sentence1', 'sentence2', and 'score' columns
            task_name: Name of the task
        """
        super().__init__(**kwargs)
        self.data = data
        self.metadata["name"] = task_name
    
    def load_data(self, **kwargs):
        """Load custom data"""
        datasets = {}
        
        for split in self.metadata["eval_splits"]:
            if split in self.data:
                datasets[split] = Dataset.from_dict(self.data[split])
        
        return datasets


# ============================================================================
# MODEL FACTORY
# ============================================================================

class ModelFactory:
    """Factory class to create different model types"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """
        Create a model instance based on type.
        
        Args:
            model_type: Type of model ('local', 'huggingface', 'api', 'openai', 'cohere', 
                                       'google_vertex', 'google_genai', 'voyage', 'claude')
            **kwargs: Arguments specific to each model type
            
        Returns:
            Model instance
        """
        if model_type == "local":
            return LocalModel(**kwargs)
        elif model_type == "huggingface":
            return HuggingFaceModel(**kwargs)
        elif model_type == "api":
            return APIModel(**kwargs)
        elif model_type == "openai":
            return OpenAIModel(**kwargs)
        elif model_type == "cohere":
            return CohereModel(**kwargs)
        elif model_type == "google_vertex":
            return GoogleVertexAIModel(**kwargs)
        elif model_type == "google_genai":
            return GoogleGenAIModel(**kwargs)
        elif model_type == "voyage":
            return VoyageAIModel(**kwargs)
        elif model_type == "claude":
            return ClaudeEmbeddingModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

class MTEBEvaluator:
    """Main class to run MTEB evaluations"""
    
    def __init__(self, model: BaseModel, output_dir: str = "./results"):
        """
        Args:
            model: Model instance
            output_dir: Directory to save results
        """
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(
        self, 
        tasks: Union[List[str], List[Any]], 
        task_langs: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Run evaluation on specified tasks.
        
        Args:
            tasks: List of task names or task instances
            task_langs: Language codes for tasks
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Evaluation results
        """
        # Get tasks
        if isinstance(tasks[0], str):
            # Task names provided
            task_objects = mteb.get_tasks(tasks=tasks, languages=task_langs)
        else:
            # Task objects provided
            task_objects = tasks
        
        # Run evaluation
        results = mteb.MTEB(tasks=task_objects).run(
            self.model,
            output_folder=self.output_dir,
            **kwargs
        )
        
        return results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: HuggingFace Model with built-in task
    print("=" * 80)
    print("Example 1: HuggingFace Model")
    print("=" * 80)
    
    model = ModelFactory.create_model(
        model_type="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    evaluator = MTEBEvaluator(model, output_dir="./results/huggingface")
    results = evaluator.evaluate(tasks=["Banking77Classification"])
    print("Results:", results)
    
    # Example 2: Local Model with custom task
    print("\n" + "=" * 80)
    print("Example 2: Custom Classification Task")
    print("=" * 80)
    
    # Create custom data
    custom_data = {
        "test": {
            "text": [
                "I love this product",
                "This is terrible",
                "Amazing experience",
                "Waste of money",
                "Highly recommend"
            ],
            "label": [1, 0, 1, 0, 1]  # 1 = positive, 0 = negative
        }
    }
    
    # Create custom task
    custom_task = CustomClassificationTask(
        data=custom_data,
        task_name="SentimentAnalysis"
    )
    
    # Evaluate
    results = evaluator.evaluate(tasks=[custom_task])
    print("Results:", results)
    
    # Example 3: API Model
    print("\n" + "=" * 80)
    print("Example 3: API Model (Generic)")
    print("=" * 80)
    
    # Note: Replace with your actual API endpoint
    api_model = ModelFactory.create_model(
        model_type="api",
        api_url="https://api.example.com/embeddings",
        api_key="your-api-key",
        model_name="your-model",
        cache_dir="./cache"
    )
    
    # evaluator = MTEBEvaluator(api_model)
    # results = evaluator.evaluate(tasks=["Banking77Classification"])
    
    # Example 4: OpenAI Model
    print("\n" + "=" * 80)
    print("Example 4: OpenAI Model")
    print("=" * 80)
    
    # openai_model = ModelFactory.create_model(
    #     model_type="openai",
    #     api_key="your-openai-api-key",
    #     model_name="text-embedding-3-small"
    # )
    
    # evaluator = MTEBEvaluator(openai_model)
    # results = evaluator.evaluate(tasks=["Banking77Classification"])
    
    # Example 5: Custom STS Task
    print("\n" + "=" * 80)
    print("Example 5: Custom STS Task")
    print("=" * 80)
    
    sts_data = {
        "test": {
            "sentence1": ["The cat sits on the mat", "A dog runs in the park"],
            "sentence2": ["A cat is on the mat", "The dog is running in the park"],
            "score": [0.9, 0.85]  # Similarity scores between 0 and 1
        }
    }
    
    sts_task = CustomSTSTask(data=sts_data, task_name="MySTS")
    results = evaluator.evaluate(tasks=[sts_task])
    print("Results:", results)
    
    # Example 6: Google Vertex AI Model
    print("\n" + "=" * 80)
    print("Example 6: Google Vertex AI Model")
    print("=" * 80)
    
    # google_vertex_model = ModelFactory.create_model(
    #     model_type="google_vertex",
    #     project_id="your-project-id",
    #     location="us-central1",
    #     model_name="text-embedding-004",
    #     credentials_path="/path/to/service-account.json"  # Optional
    # )
    
    # evaluator = MTEBEvaluator(google_vertex_model)
    # results = evaluator.evaluate(tasks=["Banking77Classification"])
    
    # Example 7: Google Generative AI (Gemini) Model
    print("\n" + "=" * 80)
    print("Example 7: Google Generative AI Model")
    print("=" * 80)
    
    # google_genai_model = ModelFactory.create_model(
    #     model_type="google_genai",
    #     api_key="your-google-api-key",
    #     model_name="models/text-embedding-004"
    # )
    
    # evaluator = MTEBEvaluator(google_genai_model)
    # results = evaluator.evaluate(tasks=["Banking77Classification"])
    
    # Example 8: Voyage AI Model (RECOMMENDED by Anthropic)
    print("\n" + "=" * 80)
    print("Example 8: Voyage AI Model (Anthropic's Recommended)")
    print("=" * 80)
    
    # voyage_model = ModelFactory.create_model(
    #     model_type="voyage",
    #     api_key="your-voyage-api-key",  # Get from https://www.voyageai.com/
    #     model_name="voyage-3"  # or voyage-3-lite, voyage-code-3
    # )
    
    # evaluator = MTEBEvaluator(voyage_model)
    # results = evaluator.evaluate(tasks=["Banking77Classification"])
    
    # Example 9: Claude Model (EXPERIMENTAL - NOT RECOMMENDED!)
    print("\n" + "=" * 80)
    print("Example 9: Claude Model (Experimental - Use Voyage AI Instead!)")
    print("=" * 80)
    
    # WARNING: This is extremely slow and expensive!
    # Use Voyage AI instead for production use
    
    # claude_model = ModelFactory.create_model(
    #     model_type="claude",
    #     api_key="your-anthropic-api-key",
    #     model_name="claude-3-5-sonnet-20241022"
    # )
    
    # evaluator = MTEBEvaluator(claude_model)
    # results = evaluator.evaluate(tasks=["Banking77Classification"])
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)