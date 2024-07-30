from .run import (
    ES_Guides7,
    ES_Translate,
    ES_VideoTransfer,
    ES_VideoTransferExtra,
)

NODE_CLASS_MAPPINGS = {
    "ES_Guides7": ES_Guides7,
    "ES_Translate": ES_Translate,
    "ES_VideoTransfer": ES_VideoTransfer,
    "ES_VideoTransferExtra": ES_VideoTransferExtra,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ES_Guides7": "ES Guides 7",
    "ES_Translate": "ES Translate",
    "ES_VideoTransfer": "ES Video Transfer",
    "ES_VideoTransferExtra": "ES Video Transfer Extra",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
