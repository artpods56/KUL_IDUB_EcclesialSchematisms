from pydantic import BaseModel, Field

from core.config.registry import register_config
from core.config.constants import ConfigType, ModelsConfigSubtype


@register_config(ConfigType.MODELS, ModelsConfigSubtype.OCR)
class OcrModelConfig(BaseModel):
    """Configuration for OCR model using PyTesseract."""
    
    language: str = Field(
        default="lat+pol+rus", 
        description="Languages passed to Tesseract (e.g. 'eng+deu', 'lat+pol+rus')"
    )
    enable_cache: bool = Field(
        default=True, 
        description="Enable caching of OCR results"
    )
    psm_mode: int = Field(
        default=6, 
        description="Page segmentation mode for Tesseract (--psm parameter)"
    )
    oem_mode: int = Field(
        default=3, 
        description="OCR Engine Mode for Tesseract (--oem parameter)"
    )
