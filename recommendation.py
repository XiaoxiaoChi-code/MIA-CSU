import torch
import numpy as np
import pandas as pd
from models import mlp
from torch.utils.data import Dataset, DataLoader


class PreprocessDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return len(self.X)


if __name__ == "__main__":

    # preparing model
    PATH = 'model_parameters/mapping_model.pt'
    mf_model = mlp(10, 10)
    mf_model.load_state_dict(torch.load(PATH))

    # preparing data
    # read testing users ID from file
    testing_users_IDs = pd.read_csv('data/mapping/testingUserIDs.csv')
    testing_users_IDs = testing_users_IDs.to_numpy().squeeze().astype(int)
    # print("The first 30 users in testing user set: ")
    # print(testing_users_IDs[:30])

    # read user embedding matrix obtained from trained source model
    user_embedding_src = pd.read_csv('model_parameters/user_embedding_source_model.csv', header=None)
    user_embedding_src_array = user_embedding_src.to_numpy()
    # print("The first 10 user embedding in source domain: ")
    # print(user_embedding_src_array[:10])

    # read user embedding matrix obtained from trained target model
    user_embedding_tgt = pd.read_csv('model_parameters/user_embedding_target_model.csv', header=None)
    user_embedding_tgt_array = user_embedding_tgt.to_numpy()
    # print("The first 10 user embedding in target domain: ")
    # print(user_embedding_tgt_array[:10])

    input_embedding = []
    label_embedding = []
    for user in testing_users_IDs:
        input_embedding.append(user_embedding_src_array[user])
        label_embedding.append(user_embedding_tgt_array[user])

    input_embedding = np.array(input_embedding).astype(float)
    label_embedding = np.array(label_embedding).astype(float)

    # print("this is the first 10 input embedding",input_embedding[:10])
    # print("this is the first 10 output embedding", label_embedding[:10])


    dataset_testing = PreprocessDataset(input_embedding, label_embedding)
    dataloader = DataLoader(dataset_testing, batch_size=128, shuffle=False)

    # 把这个predicted_embedding_tgt 存储起来
    predicted_embedding_tgt = []
    for _,(embedding_src, _) in enumerate(dataloader):
        embedding_src = embedding_src.to(torch.float32)
        predicted_embedding = mf_model(embedding_src)
        print(predicted_embedding)
        predicted_embedding_list = predicted_embedding.detach().cpu().numpy().tolist()
        # print(predicted_embedding_array)
        predicted_embedding_tgt.extend(predicted_embedding_list)
        
    # making recommendations

