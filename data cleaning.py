import os
import csv
import numpy as np

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
    dis_list = dis_item_list(data,2)
    with open(path,'w',newline='') as f:
        csv_write = csv.writer(f, delimiter = '|')
        csv_write.writerow(["User_id","Age","Gender","Level","Styles","Country","City","Item_id","Attributes","Tag"])
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
            
            row = [feature[14],feature[9],feature[12],feature[15],styles,feature[11],feature[10],feature[2],attributes,tags]
            csv_write.writerow(row)
            
os.chdir('C:/Users/wcks1/Desktop/找教授/tree_enhanced_embedding_model-master/Data/Raw/London_Attractions_Complete_Review.csv')
raw_data = open('London_Attractions_Complete_Review.csv').readlines()
create_csv(raw_data, 'Data_Lon.csv')