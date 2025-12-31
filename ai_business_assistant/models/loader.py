"""
ML Model Loader for Business AI Assistant.
Provides startup loading, caching, and lazy loading for heavy ML models.
"""

import logging
import importlib
import asyncio
from typing import Any, Dict, Optional
from pathlib import Path

from ai_business_assistant.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class ModelLoader:
    """
    Handles loading and caching of ML models.
    Supports lazy loading for optional dependencies.
    """
    _models: Dict[str, Any] = {}
    _is_loaded: bool = False

    @classmethod
    def _import_optional(cls, module_name: str) -> Optional[Any]:
        """Try to import an optional dependency."""
        try:
            return importlib.import_module(module_name)
        except ImportError:
            logger.warning(f"Optional dependency '{module_name}' not found.")
            return None

    @classmethod
    async def load_all(cls):
        """Load all configured heavy models at startup."""
        if cls._is_loaded:
            return
        
        logger.info("Loading ML models into memory...")
        
        # Ensure model cache directory exists
        settings.ensure_dirs()
        
        # Define loading tasks for different model types
        # In a real application, these would load actual model weights from settings.MODEL_CACHE_DIR
        
        # 1. Transformers (Optional)
        cls._models["transformers"] = cls._import_optional("transformers")
        if cls._models["transformers"]:
            logger.info("Transformers loaded successfully")
            
        # 2. PyTorch (Optional)
        cls._models["torch"] = cls._import_optional("torch")
        if cls._models["torch"]:
            logger.info("PyTorch loaded successfully")

        # 3. TensorFlow (Optional)
        cls._models["tensorflow"] = cls._import_optional("tensorflow")
        if cls._models["tensorflow"]:
            logger.info("TensorFlow loaded successfully")

        # 4. Prophet (Optional)
        cls._models["prophet"] = cls._import_optional("prophet")
        if cls._models["prophet"]:
            logger.info("Prophet loaded successfully")

        # 5. scikit-learn (Required for many modules)
        cls._models["sklearn"] = cls._import_optional("sklearn")
        if cls._models["sklearn"]:
            logger.info("scikit-learn loaded successfully")
            
        cls._is_loaded = True
        logger.info("ML models loading completed.")

    @classmethod
    def get_model(cls, model_name: str) -> Any:
        """Get a loaded model by name."""
        if model_name not in cls._models:
            # Try to load it on demand if not already loaded
            cls._models[model_name] = cls._import_optional(model_name)
        return cls._models.get(model_name)

    @classmethod
    def clear_cache(cls):
        """Clear the model cache."""
        cls._models.clear()
        cls._is_loaded = False
