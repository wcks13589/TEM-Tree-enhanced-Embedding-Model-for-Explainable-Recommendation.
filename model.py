import numpy as np
import torch
import torch.nn.functional as F

# TEM model
class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0,*np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):

        c_feature = torch.tensor(gbdt.apply(x.cpu().numpy()[:,2:])[:, :, 0],dtype=torch.float,device=cuda)
        x_data = torch.cat((torch.reshape(x[:,0],(-1,1)),torch.reshape(x[:,1],(-1,1)),c_feature),1)
        x_data = x_data + x_data.new_tensor(self.offsets,device=cuda).unsqueeze(0)
        return self.embedding(x_data.long())
    
class Linear(torch.nn.Module):

    def __init__(self, in_dims):
        super().__init__()
        self.fc = torch.nn.Linear(in_dims, 1)

    def forward(self, x):

        output = self.fc(x)
        return output

class Attention(torch.nn.Module):

    def __init__(self, embed_dim, attn_size):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim*2, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.embed_dim = embed_dim
        
    def forward(self, x):

        a = torch.empty(x.shape[0],1,device=cuda)
        element_product = x[:,0]*x[:,1]
        for i in range(x.shape[1]-2):
            b = torch.cat((element_product,x[:,i+2]),1)
            a = torch.cat((a,b),1)

        a = torch.reshape(a[:,1:],(a.shape[0],num_tree,self.embed_dim*2))
        attn_scores = F.relu(self.attention(a))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        #attn_scores = F.dropout(attn_scores, p=0.5)
        attn_output = torch.sum(attn_scores * x[:,2:],dim = 1)
        return attn_output/num_tree , element_product

class TEM(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, attn_size):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = Linear(x_train.shape[1]-2)
        self.projection1 = torch.nn.Linear(embed_dim, 1)
        self.projection2 = torch.nn.Linear(embed_dim, 1)
        self.afm = Attention(embed_dim, attn_size)

    def forward(self, x):
        att , element = self.afm(self.embedding(x))        
        x = self.linear(x[:,2:]) + self.projection1(element) + self.projection2(att)
        return torch.sigmoid(x.squeeze(1))