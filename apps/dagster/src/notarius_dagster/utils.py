class AssetKeyHelper:
    @staticmethod
    def build_key(*parts: str) -> str:
        """Build asset key from parts"""
        return "__".join(parts)

    @staticmethod
    def build_prefixed_key(layer: str, source: str, *parts: str) -> str:
        """Build asset key with layer and source prefix"""
        return f"{layer}__{source}__{AssetKeyHelper.build_key(*parts)}"
