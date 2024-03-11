import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from options import get_parameter
import pandas as pd
from models import MatrixFactorization


class UserItemRatingRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, rating_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.rating_tensor = rating_tensor

    def __getitem__(self, item):
        return self.user_tensor[item], self.item_tensor[item], self.rating_tensor[item]

    def __len__(self):
        return len(self.user_tensor)




if __name__ == "__main__":
    # # source domain is Douban Book data, target domain is Douban Movie data
    # src_ratings, tgt_ratings, src_unique_users, tgt_unique_data = read_data(
    #     src_file_path="data-Douban/bookreviews_cleaned.txt",
    #     tgt_file_path="data-Douban/moviereviews_cleaned.txt")
    #
    # # mapping
    # src, tgt = mapping(src=src_ratings, tgt=tgt_ratings)
    #
    # # splitting
    # train_src, train_tgt, test_tgt = splitting(src, tgt)

    args = get_parameter()

    train_src = pd.read_csv('data/source_trainingData.csv')
    train_tgt = pd.read_csv('data/target_trainingData.csv')
    test_tgt = pd.read_csv('data/target_testingData.csv')


    # training source model and store user embedding matrix and item embedding matrix
    train_src_array = train_src.to_numpy()
    users, items, ratings = train_src_array[:,0].tolist(), train_src_array[:,1].tolist(), train_src_array[:,2].tolist()
    users_tensor, items_tensor, ratings_tensor = torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(ratings)
    dataset = UserItemRatingRatingDataset(user_tensor=users_tensor,
                                          item_tensor=items_tensor,
                                          rating_tensor=ratings_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    num_users = train_src['user_id'].max()
    num_items = len(train_src['book_id'].unique())
    mf_model = MatrixFactorization(num_users+1, num_items, num_emd=10).to(device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(mf_model.parameters(), lr=0.01, momentum=0.9)

    epoch = 50

    print("Start training source domain")
    for t in range(epoch):
        loss_epoch = []
        mf_model.train()
        for _, (user, item, rating) in enumerate(dataloader):
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            predicted_rating = mf_model(user, item)
            loss = loss_func(predicted_rating, rating)
            # print(loss.item())
            loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("for epoch {}, the loss is {}".format(t, sum(loss_epoch)/ len(loss_epoch)))

    # relatively storing item matrix and user matrix for source model
    item_embedding_src = mf_model.item_embedding.weight.data.cpu()
    item_embedding_dataFrame_src = pd.DataFrame(item_embedding_src.numpy())
    item_embedding_dataFrame_src.to_csv('model_parameters/item_embedding_source_model.csv', index=False, header=None)

    user_embedding_src = mf_model.user_embedding.weight.data.cpu()
    user_embedding_dataFrame_src = pd.DataFrame(user_embedding_src.numpy())
    user_embedding_dataFrame_src.to_csv('model_parameters/user_embedding_source_model.csv', index=False, header=None)




    # training target model
    print("Start training target model")
    train_tgt_array = train_tgt.to_numpy()
    users_tgt, items_tgt, ratings_tgt = train_tgt_array[:, 0].tolist(), train_tgt_array[:, 1].tolist(), train_tgt_array[:,2].tolist()
    users_tgt_tensor, items_tgt_tensor, ratings_tgt_tensor = torch.LongTensor(users_tgt), torch.LongTensor(items_tgt), torch.FloatTensor(ratings_tgt)
    dataset_tgt = UserItemRatingRatingDataset(user_tensor=users_tgt_tensor,
                                          item_tensor=items_tgt_tensor,
                                          rating_tensor=ratings_tgt_tensor)
    dataloader_tgt = DataLoader(dataset_tgt, batch_size=128, shuffle=True)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    num_users_tgt = train_tgt['user_id'].max()
    num_items_tgt = train_tgt['movie_id'].max()
    mf_model_tgt = MatrixFactorization(num_users_tgt + 1, num_items_tgt+1, num_emd=10).to(device)
    loss_func_tgt = torch.nn.MSELoss()
    optimizer_tgt = torch.optim.SGD(mf_model_tgt.parameters(), lr=0.01, momentum=0.9)

    epoch = 50

    for t in range(epoch):
        loss_epoch = []
        mf_model_tgt.train()
        for _, (user, item, rating) in enumerate(dataloader_tgt):
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            predicted_rating = mf_model_tgt(user, item)
            loss = loss_func_tgt(predicted_rating, rating)
            # print(loss.item())
            loss_epoch.append(loss.item())
            loss.backward()
            optimizer_tgt.step()
            optimizer_tgt.zero_grad()

        print("for epoch {}, the loss is {}".format(t, sum(loss_epoch) / len(loss_epoch)))


    # relatively storing user embedding matrix and item embedding matrix for target model
    item_embedding_tgt = mf_model_tgt.item_embedding.weight.data.cpu()
    user_embedding_tgt = mf_model_tgt.user_embedding.weight.data.cpu()

    item_embedding_tgt_dataFrame = pd.DataFrame(item_embedding_tgt.numpy())
    item_embedding_tgt_dataFrame.to_csv('model_parameters/item_embedding_target_model.csv', index=False, header=None)

    user_embedding_tgt_dataFrame = pd.DataFrame(user_embedding_tgt.numpy())
    user_embedding_tgt_dataFrame.to_csv('model_parameters/user_embedding_target_model.csv', index=False, header=None)

    

    # print("the max number of user in source domain is {}".format(train_src['user_id'].max()))
    # print("there are total {} users in source domain".format(len(train_src['user_id'].unique())))
    #
    # print("the max number of item in source domain is {}".format(train_src['book_id'].max()))
    # print("there are total {} items in source domain".format(len(train_src['book_id'].unique())))



