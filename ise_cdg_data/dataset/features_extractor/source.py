from .abstraction import FeaturesExtractor
import pandas as pd

class SourceFeaturesExtractor(FeaturesExtractor):
    def __init__(self) -> None:
        pass

    def extract(self, column: "pd.Series") -> "pd.Series":
        return super().extract(column)