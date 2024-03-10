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










    # print("the max number of user in source domain is {}".format(train_src['user_id'].max()))
    # print("there are total {} users in source domain".format(len(train_src['user_id'].unique())))
    #
    # print("the max number of item in source domain is {}".format(train_src['book_id'].max()))
    # print("there are total {} items in source domain".format(len(train_src['book_id'].unique())))



