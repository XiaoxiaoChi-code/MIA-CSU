import torch


class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, num_emd):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, num_emd)
        self.item_embedding = torch.nn.Embedding(num_items, num_emd)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        return (user_emb * item_emb).sum(1)


class mlp(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden1):
        super(mlp, self).__init__()
        self.input_layer = torch.nn.Linear(dim_in, dim_hidden1)
        # self.hidden_layer1 = torch.nn.Linear(dim_hidden1, dim_out)

        self.tanh =torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.input_layer(x)
        # x = self.relu(x)
        # x = self.hidden_layer1(x)
       

        return x
    
    



