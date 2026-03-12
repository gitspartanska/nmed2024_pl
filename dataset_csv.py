#%%
import pandas as pd
import tomllib

value_mapping = {
    'his_SEX':          {'female': 0, 'male': 1},
    'his_HISPANIC':     {'no': 0, 'yes': 1},
    'his_NACCNIHR':     {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'mul': 5},
    'his_RACE':         {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'oth': 5},
    'his_RACESEC':      {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'oth': 5},
    'his_RACETER':      {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'oth': 5},
}

label_names = ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']

class CSVDataset:

    def __init__(self, dat_file, cnf_file):
        ''' ... '''
        # load data csv
        df = pd.read_csv(dat_file)

        # value mapping
        # for col, mapping in value_mapping.items():
        #     df[col] = df[col].replace(mapping)

        # load toml file to get feature names
        # with open(cnf_file, 'rb') as file:
        #     feature_names = tomllib.load(file)['feature'].keys()
        
        cnf = pd.read_csv(cnf_file)
        feature_names = [col for col in list(cnf['Name']) if col not in label_names]

        self.df = df
        self.df_features = df[feature_names]
        self.df_labels = df[label_names]

    def __len__(self):
        ''' ... '''
        return len(self.df)

    def __getitem__(self, idx):
        ''' ... '''
        row = self.df_features.iloc[idx]
        clean_row = row.dropna()
        feature_dict = clean_row.to_dict()

        row = self.df_labels.iloc[idx]
        clean_row = row.dropna()
        label_dict = clean_row.to_dict()

        return feature_dict, label_dict

if __name__ == '__main__':
    # load dataset
    dset = CSVDataset(
    dat_file = "./test.csv", 
    cnf_file = "./input_meta_info.csv"
)
    print(dset[1])

# %%
