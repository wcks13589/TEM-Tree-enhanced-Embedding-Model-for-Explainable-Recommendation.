import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from scipy import sparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

def get_data():
    user_id = sparse.load_npz('user_id.npz')
    item_id = sparse.load_npz('item_id.npz')
    cross_feature = sparse.load_npz('x_data.npz')
    y_data = sparse.load_npz('y_data.npz')
    user_id = sparse.csr_matrix(user_id.toarray().reshape(-1,1))
    item_id = sparse.csr_matrix(item_id.toarray().reshape(-1,1))
    
    x_data = sparse.hstack((user_id, item_id, cross_feature)).tocsr()
    y_data = y_data.A.reshape(-1)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state = 233, test_size=0.98)
    
    n_userid = int(np.max(x_data[:,0]))+1
    n_itemid = int(np.max(x_data[:,1]))+1
    
    return  x_train, y_train, n_userid, n_itemid

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

x_train, x_test, y_train, y_test, n_userid, n_itemid = get_data()

#Train
num_tree = 500
gbdt = GradientBoostingClassifier(n_estimators=num_tree, random_state=3 , max_depth = 6)
gbdt.fit(x_train[:,2:], y_train)
n_cross_feature = np.max(gbdt.apply(x_train[:,2:])[:, :, 0],axis = 0)+1

field_dims = np.array((n_userid, n_itemid, *n_cross_feature),dtype = np.long)

batch_size = 20
x_train_ = torch.tensor(x_train.A)
y_train_ = torch.tensor(y_train)
dataset = Data.TensorDataset(x_train_,y_train_)

train_data_loader = DataLoader(dataset, batch_size=batch_size)

cuda = torch.device('cuda')
model = TEM(field_dims,20,20)
model = model.to(cuda)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adagrad(model.parameters(),lr = 0.01)

epochs = 30

for epoch in range(epochs):
    model.train()
    print('*' * 30)
    print(f'epoch {epoch+1}')
    running_loss = 0.
    running_acc = 0.
    for i , (x,y) in enumerate(train_data_loader, 1):
        x, y = x.cuda(), y.cuda()

        output = model(x.float())
        loss = criterion(output, y.float())
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f'[{epoch+1}/{epochs}] Loss: {running_loss/i:.6f}')

torch.save(model.state_dict(), 'TEM_0422_tree500.pt')

#Test
def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def test(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    for i , (x,y) in enumerate(data_loader, 1):
        x, y = x.cuda(), y.cuda()
        output = model(x.float())        
        targets.extend(y.tolist())        
        predicts.extend(output.tolist())
    a = sorted(predicts, reverse=True).index(predicts[0])
    b = np.zeros(51)
    b[a] = 1
    print(b)
    return log_loss(targets, predicts) ,  ndcg_at_k(b, 5, 1)


x_test = sparse.load_npz('new_test_data.npz')
y_test = np.zeros(51)
y_test[0] = 1

x_test_ = torch.tensor(x_test.A)
y_test_ = torch.tensor(y_test)
testset = Data.TensorDataset(x_test_,y_test_)
test_data_loader = DataLoader(testset, batch_size=batch_size)

log , ndcg =  test(model, test_data_loader)
print('test logloss:', log , 'test NDCG@5', ndcg)