from .abstraction import FeaturesExtractor

def get_source_features_extractor() -> "FeaturesExtractor":
    from .source import SourceFeaturesExtractor
    return SourceFeaturesExtractor()