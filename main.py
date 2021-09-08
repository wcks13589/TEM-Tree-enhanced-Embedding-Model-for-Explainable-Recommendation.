import numpy as np
import torch
import torch.utils.data as Data
from scipy import sparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from model import TEM
from utils import *

args = {'num_tree':500,
        'batch_size':20,
        'embed_dim':20, 
        'lr':1e-2,
        'epochs':30,
        'ckpt_path':'TEM_0422_tree500.pt'}

def main(args):
    
    x_train, y_train, n_userid, n_itemid = Load_data()

    # Train
    gbdt = GradientBoostingClassifier(n_estimators=args['num_tree'], random_state=3 , max_depth = 6)
    gbdt.fit(x_train[:,2:], y_train)
    n_cross_feature = np.max(gbdt.apply(x_train[:,2:])[:, :, 0],axis = 0)+1

    field_dims = np.array((n_userid, n_itemid, *n_cross_feature), dtype = np.long)

    x_train_ = torch.Tensor(x_train.A)
    y_train_ = torch.Tensor(y_train)
    dataset = Data.TensorDataset(x_train_,y_train_)

    train_data_loader = DataLoader(dataset, batch_size=args['batch_size'])

    cuda = torch.device('cuda')
    model = TEM(field_dims, args['embed_dim'], args['embed_dim'])
    model = model.to(cuda)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr = args['lr'])

    for epoch in range(args['epochs']):
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

    torch.save(model.state_dict(), args['ckpt_path'])

    x_test = sparse.load_npz('new_test_data.npz')
    y_test = np.zeros(51)
    y_test[0] = 1

    x_test_ = torch.tensor(x_test.A)
    y_test_ = torch.tensor(y_test)
    testset = Data.TensorDataset(x_test_,y_test_)
    test_data_loader = DataLoader(testset, batch_size=args['batch_size'])

    log , ndcg =  test(model, test_data_loader)
    print('test logloss:', log , 'test NDCG@5', ndcg)

                                                   
if __name__ == '__main__':
    main(args)
                                                   