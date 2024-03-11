import pandas as pd
import numpy as np
import random
import tqdm


def get_history(data, uid_set):
    pos_seq_dict = {}
    for uid in tqdm.tqdm(uid_set):
        pos = data[(data.user_id == uid)&(data.rating) > 3].book_id.values.tolist()
        pos_seq_dict[uid] = pos

    return pos_seq_dict


def read_data(src_file_path, tgt_file_path):
    src_data = pd.read_csv(src_file_path, sep="\t", header=0)
    src_ratings = src_data[['user_id', 'book_id', 'rating', 'time']]
    src_ratings.to_csv('../data/book_rating.csv', index=0)

    tgt_data = pd.read_csv(tgt_file_path, sep="\t", header=0)
    tgt_ratings = tgt_data[['user_id', 'movie_id', 'rating', 'time']]
    tgt_ratings.to_csv('../data/movie_rating.csv', index=0)

    src_unique_users = src_ratings['user_id'].unique()
    tgt_unique_users = tgt_ratings['user_id'].unique()

    return src_ratings, tgt_ratings, src_unique_users, tgt_unique_users


def mapping(src, tgt):

    print("source inters: {}, uid: {}, iid: {}.".format(len(src), len(set(src.user_id)), len(set(src.book_id))))
    print("target inters: {}, uid: {}, iid: {}.".format(len(tgt), len(set(tgt.user_id)), len(set(tgt.movie_id))))

    co_uid = set(src.user_id) & set(tgt.user_id)
    all_uid = set(src.user_id) | set(tgt.user_id)

    print("All uid: {}, Co uid: {}".format(len(all_uid), len(co_uid)))

    uid_dict = dict(zip(all_uid, range(len(all_uid))))
    iid_dict_src = dict(zip(set(src.book_id), range(len(set(src.book_id)))))
    iid_dict_tgt = dict(zip(set(tgt.movie_id), range(len(set(tgt.movie_id)))))

    src.user_id = src.user_id.map(uid_dict)
    src.book_id = src.book_id.map(iid_dict_src)
    tgt.user_id = tgt.user_id.map(uid_dict)
    tgt.movie_id = tgt.movie_id.map(iid_dict_tgt)

    return src, tgt


def splitting(src, tgt):
    random.seed(1200)
    # splitting
    print("All iid: {}.".format(len(set(src.book_id) | set(tgt.movie_id))))
    src_users = set(src.user_id.unique())
    tgt_users = set(tgt.user_id.unique())

    co_users = src_users & tgt_users
    ratio = 0.5
    # 这部分用户作为 target domain 的 cold start users
    test_users = set(random.sample(co_users, round(ratio * len(co_users))))

    overlapping_users_training = co_users - test_users

    train_src = src
    train_tgt = tgt[tgt['user_id'].isin(tgt_users - test_users)]
    test_tgt = tgt[tgt['user_id'].isin(test_users)]


    return train_src, train_tgt, test_tgt, overlapping_users_training, test_users






if __name__ == "__main__":

    # source domain is Douban Book data, target domain is Douban Movie data
    src_ratings, tgt_ratings, src_unique_users, tgt_unique_data = read_data(src_file_path="../data-Douban/bookreviews_cleaned.txt",
                                                  tgt_file_path="../data-Douban/moviereviews_cleaned.txt")

    # mapping
    src, tgt = mapping(src=src_ratings, tgt=tgt_ratings)


    # splitting
    train_src, train_tgt, test_tgt, training_user_IDs, testing_user_IDs = splitting(src, tgt)



    train_src.to_csv('../data/source_trainingData.csv', sep=',', index=0)
    train_tgt.to_csv('../data/target_trainingData.csv', sep=',', index=0)
    test_tgt.to_csv('../data/target_testingData.csv', sep=',', index=0)


    training_user_IDs = np.array(list(training_user_IDs))
    testing_user_IDs = np.array(list(testing_user_IDs))

    print(training_user_IDs)
    print(testing_user_IDs)

    np.savetxt('../data/mapping/trainingUserIDs.csv', training_user_IDs, delimiter=',')
    np.savetxt('../data/mapping/testingUserIDs.csv', testing_user_IDs, delimiter=',')





























