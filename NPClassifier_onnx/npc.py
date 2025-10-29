import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors,Mol
from tqdm import tqdm
from typing import List,Dict,Tuple,Literal,Union

class NPClassifier():

    def __init__(
        self,
        model_dir_path: str,
        model_set: List[str] = ["class","pathway","superclass"],
    ) -> None:
        self.sessions,self.mapping = self.load_sessions(model_dir_path, model_set)
    
    @staticmethod
    def get_mol(inchi_or_smiles: str) -> Mol:
        if inchi_or_smiles.startswith('InChI'):
            # 处理InChI字符串
            mol = Chem.MolFromInchi(inchi_or_smiles)
            if mol is None:
                raise ValueError("无法从InChI生成分子对象")
        else:
            # 处理SMILES字符串
            mol = Chem.MolFromSmiles(inchi_or_smiles)
            if mol is None:
                raise ValueError("无法从SMILES生成分子对象")
        mol = Chem.AddHs(mol)
        return mol
    
    @staticmethod
    def calculate_fingerprint(mol:Mol) -> Tuple[np.ndarray, np.ndarray]:
        binary = np.zeros((2048*(2)), int)
        formula = np.zeros((2048),int)
        mol_bi = {}
        for r in range(2+1):
            mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=r, bitInfo=mol_bi, nBits = 2048)
            mol_bi_QC = []
            for i in mol_fp.GetOnBits():
                num_ = len(mol_bi[i])
                for j in range(num_):
                    if mol_bi[i][j][1] == r:
                        mol_bi_QC.append(i)
                        break
            if r == 0:
                for i in mol_bi_QC:
                    formula[i] = len([k for k in mol_bi[i] if k[1]==0])
            else:
                for i in mol_bi_QC:
                    binary[(2048*(r-1))+i] = len([k for k in mol_bi[i] if k[1]==r])
        return formula.reshape(1,2048).astype(np.float32),binary.reshape(1,4096).astype(np.float32)
    
    @staticmethod
    def molecules2fingerprints(input_list: List[Union[str,Mol]]) -> Tuple[np.ndarray, np.ndarray]:
        all_formulas = []
        all_binaries = []

        for input_item in input_list:

            if isinstance(input_item, str):
                try:
                    mol = NPClassifier.get_mol(input_item)
                except ValueError as e:
                    print("An error occurred: {}".format(e))
                    mol = None
            else:
                mol = input_item
            
            if mol is not None:
                try:
                    formula, binary = NPClassifier.calculate_fingerprint(mol)
                except BaseException as e:
                    print("An error occurred: {}".format(e))
                    formula = np.zeros(shape=(1,2048),dtype=np.float32)
                    binary = np.zeros(shape=(1,4096),dtype=np.float32)
            else:
                formula = np.zeros(shape=(1,2048),dtype=np.float32)
                binary = np.zeros(shape=(1,4096),dtype=np.float32)
            
            all_formulas.append(formula)
            all_binaries.append(binary)

        return np.vstack(all_formulas), np.vstack(all_binaries)
    
    @staticmethod
    def get_inputs(session:ort.InferenceSession) -> list:
        return [inp.name for inp in session.get_inputs()]
    
    def batch_predict(
        self, 
        input_list: List[Union[str,Mol]],
        used_model: Union[List[str],Literal["auto"]] = "auto",
    ) -> Dict[str, np.ndarray]:
        
        fps = self.molecules2fingerprints(input_list)
        err_index = np.where(np.sum(fps[0],axis=-1) == 0)[0]

        results = {}
        for model_name, session in self.sessions.items():
            if used_model == "auto" or model_name in used_model:
                inp = dict(zip(self.get_inputs(session),fps))
                result_tensor = session.run(None, inp)[0]
                result_tensor[err_index] = 0
                results[model_name] = result_tensor

        return results
    
    def index2label(
        self,
        index: Union[int, List[int]],
        model_name: str,
    ) -> Union[str, List[str]]:

        mapping_series = self.mapping[model_name.split('_')[0].lower()]

        if isinstance(index, int):
            return mapping_series.iloc[index]
        elif isinstance(index, (list, np.ndarray)):
            return mapping_series.iloc[index].tolist()
        else:
            raise ValueError("index参数类型错误")
        
    def indexs2labels(
        self,
        indexs: np.ndarray,
        model_name: str,
    ) -> np.ndarray:
        shape = indexs.shape
        indexs = indexs.reshape(-1)
        labels = np.array(self.index2label(indexs, model_name))
        return labels.reshape(shape)
    
    @staticmethod
    def get_top_k_index(array: np.ndarray, top_k: int) -> np.ndarray:
        return np.argsort(array)[-top_k:][::-1].astype(int)
    
    @staticmethod
    def get_top_k_prob(array: np.ndarray, top_k: int) -> np.ndarray:
        return np.sort(array)[-top_k:][::-1]
    
    def decode_result(
        self,
        results: np.ndarray, #shape -> (batch_size, num_classes)
        model_name: str,
        top_k: int = 1,
    ) -> pd.DataFrame:
        results_list = []
        for result in results:
            result_dict = {}
            if np.sum(result) > 0:
                i = self.get_top_k_index(result, top_k)
                P = self.get_top_k_prob(result, top_k)
                L = self.index2label(i, model_name)
                for k in range(top_k):
                    result_dict[f"{model_name}_prob_{k+1}"] = P[k]
                    result_dict[f"{model_name}_label_{k+1}"] = L[k]
            results_list.append(result_dict)
            # else:
            #     for k in range(top_k):
            #         result_dict[f"{model_name}_prob_{k+1}"] = None
            #         result_dict[f"{model_name}_label_{k+1}"] = None
        return pd.DataFrame(results_list)
    
    def decode_results(
        self,
        results_dict:Dict[str,np.ndarray],
        top_k:int = 1,
    ) -> pd.DataFrame:
        df_list = []
        for model_name, results in results_dict.items():
            df = self.decode_result(results, model_name, top_k)
            df_list.append(df)
        return pd.concat(df_list, axis=1)
    
    def batch_predict_and_decode(
        self,
        input_list: List[Union[str,Mol]],
        used_model: Union[List[str],Literal["auto"]] = "auto",
        top_k:int = 1,
    ) -> pd.DataFrame:
        results_dict = self.batch_predict(input_list, used_model)
        return self.decode_results(results_dict, top_k)

    @staticmethod
    def load_sessions(
        model_dir_path: str,
        model_set: List[str],
    ) -> Tuple[
        Dict[str, ort.InferenceSession],
        Dict[str, pd.Series]
    ]:
        model_sessions = {}

        for model_file in os.listdir(model_dir_path):
            
            if model_file.endswith('.onnx'):
                model_name = os.path.splitext(model_file)[0]
                
                if any(model_name.startswith(prefix) for prefix in model_set):
                    model_path = os.path.join(model_dir_path, model_file)

                    model_sessions[model_name] = ort.InferenceSession(model_path)
            elif model_file.endswith('.json'):
                model_name = os.path.splitext(model_file)[0]
                
                if model_name.startswith('index'):
                    model_path = os.path.join(model_dir_path, model_file)

                    mapping = NPClassifier.load_mapping(model_path)

        return model_sessions,mapping
    
    @staticmethod
    def load_mapping(mapping_path: str) -> Dict[str, pd.Series]:
        with open(mapping_path, 'r') as f:
            mapping_dict = json.load(f)
        mapping = {}
        for model_type, mapping_dict in mapping_dict.items():
            mapping[model_type.lower()] = pd.Series(
                list(mapping_dict.keys()), 
                index=list(mapping_dict.values())
            )
        return mapping
    
    def predict_from_table_file(
        self,
        data_path: str,
        tag_column: str = 'inchi',
        batch_size: int = 256,
        top_k: int = 1,
    ) -> pd.DataFrame:
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)[tag_column].values
        elif data_path.endswith('.xlsx'):
            data = pd.read_excel(data_path)[tag_column].values
        elif data_path.endswith('.txt') or data_path.endswith('.tsv') or data_path.endswith('.tab'):
            data = pd.read_table(data_path, sep='\t')[tag_column].values
        elif data_path.endswith('.pkl'):
            data = pd.read_pickle(data_path)[tag_column].values
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)[tag_column].values
        elif data_path.endswith('.h5') or data_path.endswith('.hdf5') or data_path.endswith('.hdf'):
            data = pd.read_hdf(data_path, key=tag_column).values
        batched_data = np.array_split(data, len(data)//batch_size+1 if len(data)%batch_size!=0 else len(data)//batch_size)
        results = []
        for batch in tqdm(batched_data):
            results.append(self.batch_predict_and_decode(batch, used_model="auto", top_k=top_k))
        results = pd.concat(results, axis=0)
        results.set_index(data, inplace=True)
        return results
