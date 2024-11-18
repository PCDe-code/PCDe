import pickle
import numpy as np
import random
import torch
import os
import sys
from scipy.stats import entropy


CAL = 'CAL'
PHO = 'PHO'
CHA = 'CHA'

DATA_DIR = {
    CHA: './data/CHA',
    PHO: './data/PHO',
    CAL: './data/CAL'
}
TMP_DIR = {
    CHA: './tmp',
    PHO: './tmp',
    CAL: './tmp'
}


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


def save_dataset(dataset, dataset_obj):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)


def load_dataset(dataset):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj


def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))


def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg


def save_fm_sample(dataset, sample_data, mode):
    sample_file = DATA_DIR[dataset] + '/FM_sample_data/sample_fm_data_{}.pkl'.format(mode)
    print('save fm_{}_data in {}'.format(mode, sample_file))
    pickle.dump(sample_data, open(sample_file, 'wb'))


def load_fm_sample(dataset, mode, epoch=0):
    if mode == 'train':
        sample_file = DATA_DIR[dataset] + '/FM_sample_data/sample_fm_data_{}-{}.pkl'.format(mode, epoch)

    if mode == 'valid':
        sample_file = DATA_DIR[dataset] + '/FM_sample_data/sample_fm_data_{}.pkl'.format(mode)
    with open(sample_file, 'rb') as f:
        sample_data = pickle.load(f)
    return sample_data


def load_fm_model(dataset, model, filename, epoch):
    model_file = TMP_DIR[dataset] + '/FM-model-merge/' + filename + '-epoch-{}.pt'.format(epoch)
    model_dict = torch.load(model_file)
    print('Model load at {}'.format(model_file))
    return model_dict


def save_fm_model(dataset, model, filename, epoch):
    model_file = TMP_DIR[dataset] + '/FM-model-merge/' + filename + '-epoch-{}.pt'.format(epoch)
    if not os.path.isdir(TMP_DIR[dataset] + '/FM-model-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/FM-model-merge/')
    torch.save(model.state_dict(), model_file)
    print('Model saved at {}'.format(model_file))


def load_embed(dataset, epoch):
    path = TMP_DIR[dataset] + '/FM-model-embeds/' + 'embeds-epoch-{}.pkl'.format(epoch)
    with open(path, 'rb') as f:
        embeds = pickle.load(f)
        print('FM Epoch:{} Embedding load successfully!'.format(epoch))
        return embeds


def save_embed(dataset, embeds, epoch):
    path = TMP_DIR[dataset] + '/FM-model-embeds/' + 'embeds-epoch-{}.pkl'.format(epoch)
    if not os.path.isdir(TMP_DIR[dataset] + '/FM-model-embeds/'):
        os.makedirs(TMP_DIR[dataset] + '/FM-model-embeds/')
    with open(path, 'wb') as f:
        pickle.dump(embeds, f)
        print('Embedding saved successfully!')


def save_fm_model_log(dataset, filename, epoch, epoch_loss, epoch_loss_2, train_len):
    PATH = TMP_DIR[dataset] + '/FM-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/FM-log-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/FM-log-merge/')
    with open(PATH, 'a') as f:
        f.write('Starting {} epoch\n'.format(epoch))
        f.write('training loss 1: {}\n'.format(epoch_loss / train_len))
        f.write('training loss 2: {}\n'.format(epoch_loss_2 / train_len))
        f.write('=============================\n')
      


def save_fm_sample_log(dataset, log_data, head_name, mode='train'):
    sample_file = TMP_DIR[dataset] + '/sample_fm_{}_log.txt'.format(mode)
    with open(sample_file, mode='w+', encoding='utf-8') as f:
        for name in head_name:
            f.write(name + '\t')
        f.write('\n')
        np.savetxt(f, log_data, fmt='%s', delimiter='\t')


def load_rl_agent(dataset, filename, epoch_user):
    model_file = TMP_DIR[dataset] + '/RL-agent/' + filename + '-epoch-{}.pkl'.format(epoch_user)
    model_dict = torch.load(model_file)
    print('RL policy model load at {}'.format(model_file))
    return model_dict
def load_rl_agent_location(dataset, filename, epoch_user):
    model_file = TMP_DIR[dataset] + '/RL-agent-location/' + filename + '-epoch-{}.pkl'.format(epoch_user)
    model_dict = torch.load(model_file)
    print('RL policy model load at {}'.format(model_file))
    return model_dict


def save_rl_agent(dataset, model, filename, epoch_user):
    model_file = TMP_DIR[dataset] + '/RL-agent/' + filename + '-epoch-{}.pkl'.format(epoch_user)
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-agent/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-agent/')
    torch.save(model.state_dict(), model_file)
    print('RL policy model saved at {}'.format(model_file))
def save_rl_agent_location(dataset, model, filename, epoch_user):
    model_file = TMP_DIR[dataset] + '/RL-agent-location/' + filename + '-epoch-{}.pkl'.format(epoch_user)
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-agent-location/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-agent-location/')
    torch.save(model.state_dict(), model_file)
    print('RL policy model saved at {}'.format(model_file))

def save_rl_mtric(dataset, filename, epoch, SR, spend_time, mode='train'):
    PATH = TMP_DIR[dataset] + '/RL-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-log-merge/')
    if mode == 'train':
        with open(PATH, 'a') as f:
            f.write('===========Train===============\n')
            f.write('Starting {} user epochs\n'.format(epoch))
            f.write('training SR@5: {}\n'.format(SR[0]))
            f.write('training SR@10: {}\n'.format(SR[1]))
            f.write('training SR@15: {}\n'.format(SR[2]))
            f.write('training Avg@T: {}\n'.format(SR[3]))
            f.write('Spending time: {}\n'.format(spend_time))
            f.write('================================\n')
           
        with open(PATH, 'a') as f:
            f.write('===========Test===============\n')
            f.write('Testing {} user tuples\n'.format(epoch))
            f.write('Testing SR@5: {}\n'.format(SR[0]))
            f.write('Testing SR@10: {}\n'.format(SR[1]))
            f.write('Testing SR@15: {}\n'.format(SR[2]))
            f.write('Testing Avg@T: {}\n'.format(SR[3]))
            f.write('Testing time: {}\n'.format(spend_time))
            f.write('================================\n')
           

def save_rl_mtric_location(dataset, filename, epoch, SR, spend_time, mode='train'):
    PATH = TMP_DIR[dataset] + '/RL-log-merge-location/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge-location/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-log-merge-location/')
    if mode == 'train':
        with open(PATH, 'a') as f:
            f.write('===========Train===============\n')
            f.write('Starting {} user epochs\n'.format(epoch))
            f.write('training SR@5: {}\n'.format(SR[0]))
            f.write('training SR@10: {}\n'.format(SR[1]))
            f.write('training SR@15: {}\n'.format(SR[2]))
            f.write('training Avg@T: {}\n'.format(SR[3]))
            f.write('Spending time: {}\n'.format(spend_time))
            f.write('================================\n')
            
    elif mode == 'test':
        with open(PATH, 'a') as f:
            f.write('===========Test===============\n')
            f.write('Testing {} user tuples\n'.format(epoch))
            f.write('Testing SR@5: {}\n'.format(SR[0]))
            f.write('Testing SR@10: {}\n'.format(SR[1]))
            f.write('Testing SR@15: {}\n'.format(SR[2]))
            f.write('Testing Avg@T: {}\n'.format(SR[3]))
            f.write('Testing time: {}\n'.format(spend_time))
            f.write('================================\n')
            


def save_rl_model_log(dataset, filename, epoch, epoch_loss, train_len):
    PATH = TMP_DIR[dataset] + '/RL-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-log-merge/')
    with open(PATH, 'a') as f:
        f.write('Starting {} epoch\n'.format(epoch))
        f.write('training loss : {}\n'.format(epoch_loss / train_len))
        
def save_rl_model_log_location(dataset, filename, epoch, epoch_loss, train_len):
    PATH = TMP_DIR[dataset] + '/RL-log-merge-location/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge-location/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-log-merge-location/')
    with open(PATH, 'a') as f:
        f.write('Starting {} epoch\n'.format(epoch))
        f.write('training loss : {}\n'.format(epoch_loss / train_len))
       

def save_pretrain_data(dataset, data):
    path = TMP_DIR[dataset] + '/Pretrain-data/' + '{}-pretrain-data.pkl'.format(dataset)
    if not os.path.isdir(TMP_DIR[dataset] + '/Pretrain-data/'):
        os.makedirs(TMP_DIR[dataset] + '/Pretrain-data/')
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        print('{} Pretrain_data saved successfully!'.format(dataset))


def load_pretrain_data(dataset):
    path = TMP_DIR[dataset] + '/Pretrain-data/' + '{}-pretrain-data.pkl'.format(dataset)
    with open(path, 'rb') as f:
        data = pickle.load(f)
        print('{} Pretrain_data load successfully!'.format(dataset))
        return data


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def blockPrint():
    sys.stdout = open(os.devnull, 'w')



def enablePrint():
    sys.stdout = sys.__stdout__



def js_divergence(p, q):

    p=np.array(p)
    q=np.array(q)
    m = 0.5 * (p + q)
    
    kl_pm = entropy(p, m, base=2)
    kl_qm = entropy(q, m, base=2)
    
    js = 0.5 * (kl_pm + kl_qm)
    return js
