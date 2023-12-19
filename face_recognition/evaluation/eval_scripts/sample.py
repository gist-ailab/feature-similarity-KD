import pickle

if __name__=='__main__':
    path = "/home/jovyan/SSDb/sung/src/feature-similarity-KD/result/table2/tinyface/qualnet_qualnet_CosFace_5.pkl"
    with open(path, 'rb') as f:
        result = pickle.load(f)
        print(result)