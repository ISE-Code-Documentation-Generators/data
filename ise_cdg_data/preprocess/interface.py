from abc import ABC, abstractmethod



class PreprocessInterface(ABC):
    @abstractmethod
    def preprocess(self) -> None:
        pass


def get_preprocessor_for_ast_augmentation(
    path,
    batch_number: int = 0,
    batch_size: int = 1000,
) -> PreprocessInterface:
    from ise_cdg_data.preprocess.preprocessor import ASTPreprocessor
    return ASTPreprocessor(path, batch_number=batch_number, batch_size=batch_size)
