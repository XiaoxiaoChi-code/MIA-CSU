import torch




class mlp(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, dim_out):
        super(mlp, self).__init__()
        self.input_layer = torch.nn.Linear(dim_in, dim_hidden1)
        self.hidden_layer1 = torch.nn.Linear(dim_hidden1, dim_hidden2)
        self.hidden_layer2 = torch.nn.Linear(dim_hidden2, dim_out)

        self.tanh =torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer1(x)
        x = self.relu(x)
        x = self.hidden_layer2(x)

        return x



