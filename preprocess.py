import csv         
import random
import numpy as np
from scipy import sparse


def dis_item_list(data,feature_index):
    items = []
    for index , row in enumerate(data[1:]):
        print('Collecting item now:' , index)
        feature = row.split('\t')
        if feature[feature_index] not in items:
            items.append(feature[feature_index])
    item = [0 for j in range(len(items))]
    for index , line in enumerate(data[1:]):
        print('Sum item vector now:' , index)
        feature = line.split('\t')
        item[items.index(feature[feature_index])] += 1
    count = []
    for index , item in enumerate(item):
        if item < 5 :
            count.append(index)
    dis_item = []
    for i in count:
        dis_item.append(items[i])
    return dis_item

def create_csv(data, path):  #清理不要的資料欄位與重新給予user ID與item ID
    col_names = ["User_id","Age","Gender","Level","Styles","Country","City","Item_id","Attributes","Tag"]
    dis_list = dis_item_list(data,2)
    
    with open(path,'w',newline='') as f:
        csv_write = csv.writer(f, delimiter = '|')
        csv_write.writerow(col_names)
        ord_user_index = []
        ord_item_index = []
        
        for index , line in enumerate(data[1:]):
            print('Saving data now:' , index)
            feature = line.split('\t')
            
            if feature[2] in dis_list:
                continue
                
            if feature[14] not in ord_user_index:
                ord_user_index.append(feature[14])
                feature[14] = str(ord_user_index.index(feature[14]))
            else:
                feature[14] = str(ord_user_index.index(feature[14]))
                
            if feature[2] not in ord_item_index:
                ord_item_index.append(feature[2])
                feature[2] = str(ord_item_index.index(feature[2]))
            else:
                feature[2] = str(ord_item_index.index(feature[2]))
                
            styles = []
            for style in feature[19].strip('[').strip(']').split(sep = ", "):
                styles.append(style.strip("'"))
                
            attributes = []
            for attribute in feature[20].strip('[').strip(']').split(sep = ", "):
                attributes.append(attribute.strip("'"))
                
            tags = []
            for tag in feature[28].strip().strip('[').strip(']').split(sep = ", "):
                tags.append(tag.strip("'"))
            
            row = [feature[14],
                   feature[9],
                   feature[12],
                   feature[15],
                   styles,feature[11],
                   feature[10],
                   feature[2],
                   attributes,tags]
            
            csv_write.writerow(row)

def collect_style_attribute(data):
    styles = []
    attributes = []
    for index , row in enumerate(data[1:]):
        print('Collecting Style and Attribute Now!:' , index)
        feature = row.split('|')
        style = feature[4].strip('[').strip(']').split(', ')
        attribute = feature[8].strip('[').strip(']').split(', ')
        for i in style:
            i = i.strip("'")
            if i not in styles:
                styles.append(i)
        for i in attribute:
            i = i.strip("'")
            if i not in attributes:
                attributes.append(i)
    return styles, attributes

def collect_tag(data):
    tags = []
    for index , row in enumerate(data[1:]):
        print('Collecting Tags Now!:' , index)
        feature = row.split('|')
        tag = feature[9].strip().strip('[').strip(']').split(', ')
        for i in tag:
            i = i.strip("'")
            if i not in tags:
                tags.append(i)
    return tags

def create_vector(item,item_list):
    item_vector = [0 for i in range(len(item_list))]
    for i in item:
        i = i.strip("'")
        try:
            item_vector[item_list.index(i)] = 1
        except:
            continue
    return item_vector

def load_data(data):
    style_list, attribute_list = collect_style_attribute(data)
    style_list.remove('')
    tag_list = collect_tag(data)
    tag_list.remove('')
    user_data = []
    user_style =  sparse.csr_matrix((1,len(style_list)))
    item_attribute = sparse.csr_matrix((1,len(attribute_list)))
    item_tag = sparse.csr_matrix((1,len(tag_list)))
    for index , row in enumerate(data[1:]):
        print('Loading Data Now:', index)
        feature = row.split('|')
        style = feature[4].strip('[').strip(']').split(', ')
        style_vector = create_vector(style,style_list)
        
        user_style = sparse.vstack((user_style, sparse.csr_matrix(style_vector)))
        
        attribute = feature[8].strip('[').strip(']').split(', ')
        attribute_vector = create_vector(attribute,attribute_list)
        
        item_attribute = sparse.vstack((item_attribute, sparse.csr_matrix(attribute_vector)))
        
        tag = feature[9].strip().strip('[').strip(']').split(', ')
        tag_vector = create_vector(tag,tag_list)
        
        item_tag = sparse.vstack((item_tag, sparse.csr_matrix(tag_vector)))
        
        user_data.append({"Age": feature[1], "Gender": feature[2], "Level":feature[3],
                          "Country": feature[5], "City": feature[6] })
    return (user_data, user_style[1:], item_attribute[1:], item_tag[1:])

def interaction_matrix(data):
    max_user_id = -1
    max_item_id = -1
    for row in data[1:]:
        cur_user = int(row.split('|')[0])
        cur_item = int(row.split('|')[7])
        if cur_user > max_user_id:
            max_user_id = cur_user
        if cur_item > max_item_id:
            max_item_id = cur_item
    matrix = np.zeros((max_user_id+1, max_item_id+1),dtype=int)
    for index , row in enumerate(data[1:]):
        print('Building Matrix Now:' , index)
        user_id , item_id = int(row.split('|')[0]) , int(row.split('|')[7])
        matrix[user_id][item_id] = 1
    return matrix

def user_item_vector_dic(data,user_data,item_data):
    user_dic = {}
    item_dic = {}
    for index, row in enumerate(data[1:]):
        feature = row.split('|')
        user_dic[int(feature[0])] = user_data[index]
        item_dic[int(feature[7])] = item_data[index]
        print('Creating User and Item Vector Now:', index)
    return user_dic, item_dic

def create_x_data():
    user_item_matrix = interaction_matrix(data)
    user_dic, item_dic = user_item_vector_dic(data, user_data, item_data)
    x_data = sparse.csr_matrix((1,(user_data.shape[1]+item_data.shape[1])))
    y_data = []
    users = []
    items = []
    for user_id, vector in enumerate(user_item_matrix):
        x_tem = np.zeros((1,user_data.shape[1]+item_data.shape[1]),dtype = np.int)
        neg_index = [i for i in range(len(vector))]
        for item_id, element in enumerate(vector):
            if element == 1:
                neg_index.remove(item_id)
                x = np.append(user_dic[user_id].A, item_dic[item_id].A).reshape(1,-1)
                x_tem = np.append(x_tem, x, axis = 0)
                y_data.append(element)
                users.append(user_id)
                items.append(item_id)
        num_neg = (len(vector)-len(neg_index))*5
        try:
            item_ids = random.sample(neg_index, num_neg)
        except:
            item_ids = []
        for item_id in item_ids:
            x = np.append(user_dic[user_id].A, item_dic[item_id].A).reshape(1,-1)
            x_tem = np.append(x_tem, x, axis = 0)
            y_data.append(vector[item_id])
            users.append(user_id)
            items.append(item_id)
        x_data = sparse.vstack((x_data, sparse.csr_matrix(x_tem[1:])))
        print('Creating X_data Now:', user_id)
    return x_data.tocsr()[1:], np.array(y_data), np.array(users), np.array(items)