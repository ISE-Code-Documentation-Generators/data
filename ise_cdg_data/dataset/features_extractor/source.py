from .abstraction import FeaturesExtractor
import pandas as pd

from tqdm.notebook import tqdm
tqdm.pandas()

from .source_utils import *
class SourceFeaturesExtractor(FeaturesExtractor):
    def __init__(self) -> None:
        pass

    def extract(self, source_column: "pd.Series") -> "pd.Series":
        df = pd.DataFrame()
        df['source'] = source_column
        self.extract_feature_columns(code_df=df)
        df = df.drop(columns=['API', 'source'])
        print("features_types")
        print(self.get_types(df))
        return df.apply(self.aggregate_features, axis=1)

    def aggregate_features(self, row):
        return list(row)

    def get_types(self, df: "pd.DataFrame"):
        types = dict()
        for column in df.columns:
            types[column] = type(df[column].iloc[0])
        return types
        
    def extract_feature_columns(self, code_df):
        # TODO Refactor: make a class for each set of features
        # TODO Refactor: refine "progress_apply" behavior
        code_df["LOC"] = code_df["source"].apply(
            lambda x: x.count("\n") + 1 if type(x) == str else 0
        )
        code_df["BLC"] = code_df["LOC"].apply(lambda x: 1 if x == 0 else 0)
        code_df["UDF"] = code_df["source"].apply(
            lambda x: sum([len(re.findall("^(?!#).*def ", y)) for y in x.split("\n")])
            if type(x) == str
            else 0
        )
        code_df["I"] = code_df["source"].apply(
            lambda x: x.count("import ") if type(x) == str else 0
        )
        code_df["EH"] = code_df["source"].apply(
            lambda x: x.count("try:") if type(x) == str else 0
        )
        import statistics as sts

        code_df["ALLC"] = code_df["source"].apply(
            lambda x: sts.mean([len(y) for y in x.split("\n")]) if type(x) == str else 0
        )
        # code_df['NDD'] = code_df['output_type'].progress_apply(lambda x: x.count('display_data') if type(x)==str else 0)
        code_df["NDD"] = 0
        # code_df['NEC'] = code_df['output_type'].progress_apply(lambda x: x.count('execute_result') if type(x)==str else 0)
        code_df["NEC"] = 0

        ### -----

        code_df['S'] = code_df['source'].progress_apply(lambda x: statements_count(str(x)))

        code_df['P'] = code_df['source'].progress_apply(lambda x: python_arguments(str(x)))

        code_df['KLCID'] = code_df['source'].progress_apply(lambda x: klcid(str(x)))

        code_df['NBD'] = code_df['source'].progress_apply(lambda x: nested_depth(str(x)))

        code_df['OPRND'] = code_df['source'].progress_apply(lambda x: extract_operand_count(str(x)))

        ### ----

        code_df['OPRATOR'] = code_df['source'].progress_apply(lambda x: extract_operator_count(str(x)))

        code_df['UOPRND'] = code_df['source'].progress_apply(lambda x: extract_unique_operand_count(str(x)))

        code_df['UOPRATOR'] = code_df['source'].progress_apply(lambda x: extract_unique_operator_count(str(x)))

        code_df['ID'] = code_df['source'].progress_apply(lambda x: extract_identifier_count(str(x)))

        code_df['ALID'] = code_df['source'].progress_apply(lambda x: extract_avg_len_identifier(str(x)))

        code_df['MLID'] = code_df['source'].progress_apply(lambda x: extract_max_len_identifier(str(x)))

        code_df ['CyC'] = code_df['source'].progress_apply(lambda x: complexity_analysis2(str(x)))

        ### ---

        code_df['API'] = code_df['source'].apply(lambda x: capture_imports(str(x)))
        eap_score = eap_score_function_generator(code_df['API'])
        code_df['EAP'] = code_df['API'].apply(lambda x: eap_score(set(x)))

        ### Comment Metrics

        code_df['LOCom'] = code_df['source'].progress_apply(lambda x: extract_line_comments(str(x)))

        code_df['CW'] = code_df['source'].progress_apply(lambda x: count_comment_word(str(x)))