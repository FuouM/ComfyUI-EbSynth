from .run import (
    ES_Guides7,
    ES_Translate
)

NODE_CLASS_MAPPINGS = {
    "ES_Guides7": ES_Guides7,
    "ES_Translate": ES_Translate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ES_Guides7": "ES Guides 7",
    "ES_Translate": "ES Translate",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
