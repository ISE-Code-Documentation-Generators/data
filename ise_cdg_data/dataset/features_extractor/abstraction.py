from abc import ABC, abstractmethod
import pandas as pd

class FeaturesExtractor(ABC):
    @abstractmethod
    def extract(self, column: "pd.Series") -> "pd.Series":
        pass