import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import log_loss

from preprocess import create_csv, load_data, create_x_data

# Get data
def Load_data(test_size=0.98):
    
    raw_data_file = 'London_Attractions_Complete_Review.csv'

    raw_data = open(raw_data_file).readlines()
    create_csv(raw_data, 'Data_Lon.csv')

    data = open('Data_Lon.csv').readlines()

    user_data, user_style, item_attribute, item_tag = load_data(data)

    v = DictVectorizer()
    user_data = v.fit_transform(user_data)

    user_data = sparse.hstack((user_data, user_style)).tocsr()
    item_data = sparse.hstack((item_attribute, item_tag)).tocsr()

    x_data, y_data, user_id, item_id = create_x_data()

    sparse.save_npz('x_data.npz', x_data)
    sparse.save_npz('y_data.npz', sparse.csr_matrix(y_data))
    sparse.save_npz('user_id.npz', sparse.csr_matrix(user_id))
    sparse.save_npz('item_id.npz', sparse.csr_matrix(item_id))
    
    user_id = sparse.load_npz('user_id.npz')
    item_id = sparse.load_npz('item_id.npz')
    cross_feature = sparse.load_npz('x_data.npz')
    y_data = sparse.load_npz('y_data.npz')
    user_id = sparse.csr_matrix(user_id.toarray().reshape(-1,1))
    item_id = sparse.csr_matrix(item_id.toarray().reshape(-1,1))
    
    x_data = sparse.hstack((user_id, item_id, cross_feature)).tocsr()
    y_data = y_data.A.reshape(-1)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state = 233, test_size=test_size)
    
    n_userid = int(np.max(x_data[:,0]))+1
    n_itemid = int(np.max(x_data[:,1]))+1
    
    return  x_train, y_train, n_userid, n_itemid

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