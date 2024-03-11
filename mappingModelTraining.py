import numpy as np
import torch
import pandas as pd
from models import mlp
from torch.utils.data import Dataset, DataLoader
from options import get_parameter

class PreprocessDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return len(self.X)


if __name__ == "__main__":
    args = get_parameter()
    # read training users ID from file
    training_users_IDs = pd.read_csv('data/mapping/trainingUserIDs.csv')
    training_users_IDs = training_users_IDs.to_numpy().squeeze().astype(int)
    # read testing users ID from file
    testing_users_IDs = pd.read_csv('data/mapping/testingUserIDs.csv')
    testing_users_IDs = testing_users_IDs.to_numpy().squeeze().astype(int)


    # read user embedding matrix obtained from trained source model
    user_embedding_src = pd.read_csv('model_parameters/user_embedding_source_model.csv', header=None)
    user_embedding_src_array = user_embedding_src.to_numpy()
    # read user embedding matrix obtained from trained target model
    user_embedding_tgt = pd.read_csv('model_parameters/user_embedding_target_model.csv', header=None)
    user_embedding_tgt_array = user_embedding_tgt.to_numpy()

    training_embeddings_input = []
    training_embeddings_label = []
    for user_ID in training_users_IDs:
        training_embeddings_input.append(user_embedding_src_array[user_ID])
        training_embeddings_label.append(user_embedding_tgt_array[user_ID])

    training_embeddings_input = np.array(training_embeddings_input)
    training_embeddings_label = np.array(training_embeddings_label)

    dataset = PreprocessDataset(training_embeddings_input, training_embeddings_label)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    criterion = torch.nn.CrossEntropyLoss()
    # 这个模型如果效果不好，参考RecBole 库是怎么定义这个函数的
    mlp_model = mlp(dim_in=10, dim_hidden1=20, dim_out=10).to(device)
    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.7)
    for e in range(20):
        loss_epoch = []
        for _, (feature, label) in enumerate(dataloader):
            feature, label = feature.to(torch.float32).to(device), label.to(torch.float32).to(device)
            mlp_model.zero_grad()
            predicted_value = mlp_model(feature)

            loss = criterion(predicted_value, label)
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
        print("for epoch {}, the loss is {}".format(e, sum(loss_epoch) / len(loss_epoch)))

