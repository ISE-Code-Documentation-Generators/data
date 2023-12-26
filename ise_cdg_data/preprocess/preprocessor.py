from abc import ABC
import typing
import pandas as pd

import torch
from tqdm import tqdm
from ise_cdg_data.dataset.interface import Facade

from ise_cdg_data.preprocess import PreprocessInterface
from ise_cdg_data.utils import LazyFileName, get_range


class ASTPreprocessor(PreprocessInterface):
    @classmethod
    def init_dataset(cls, path):
        dataset = Facade(Facade.DatasetMode.WITH_PREPROCESS).get_md4def(path)
        geo_dataset = Facade(Facade.DatasetMode.WITH_PREPROCESS).get_source_graph(
            path, dataset.src_vocab
        )
        assert len(dataset) == len(geo_dataset), 'Oops something went wrong since the `geo_dataset` was not of the same size as the `dataset`!'
        return dataset, geo_dataset

    def __init__(
        self,
        path,
        batch_number: int = 0,
        batch_size: typing.Optional[int] = None, 
        output_name: typing.Optional[str] = None,
        logger: typing.Callable = print,
    ) -> None:
        super().__init__()
        self.path = path
        self.dataset, self.geo_dataset = self.init_dataset(self.path)
        self.output_name = output_name or LazyFileName(self.path).name
        self.logger = logger

        self.batch_number = batch_number
        self.batch_size = batch_size or len(self.dataset)

    def preprocess(self) -> None:
        self.vocab_stage()
        self.dataset_stage()

    def vocab_stage(self) -> None:
        torch.save(self.dataset.src_vocab, f"src_vocab_{self.output_name}.pt")
        self.logger("source code vocabulary has been saved.")
        torch.save(self.dataset.md_vocab, f"md_vocab_{self.output_name}.pt")
        self.logger("markdown vocabulary has been saved.")

    def save_dataset(self, df_dict: dict, use_prev: bool):
        output_dataset_path = f'preprocessed_dataset_{self.output_name}.csv'

        new_df = pd.DataFrame(df_dict)
        if not use_prev:
            to_be_saved_df = new_df
        else:
            prev_df = pd.read_csv(output_dataset_path)
            to_be_saved_df = pd.concat([prev_df, new_df], ignore_index=True, sort=False)

        to_be_saved_df.to_csv(output_dataset_path, index=False)
    
    def dataset_stage(self) -> None:
        dataset_size = len(self.dataset)
        for batch in range(self.batch_number, dataset_size//self.batch_size):
            df_dict = {
                'source': [],
                'markdown': [],
                'ast_nodes': [],
                'ast_edges': [],
            }
            for i in tqdm(get_range(self.batch_size, batch, dataset_size)):
                try:
                    data = self.dataset[i]
                    src, md = data
                    geo_data = self.geo_dataset[i]
                except:
                    continue
                df_dict['source'].append(src.tolist())
                df_dict['markdown'].append(md.tolist())
                df_dict['ast_nodes'].append(geo_data.x.tolist())
                df_dict['ast_edges'].append(geo_data.edge_index.tolist())
            self.logger(f'end of batch number {batch}')
            self.save_dataset(df_dict, batch != 0)
