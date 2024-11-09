
import math
import random
import numpy as np
import os
import sys
# sys.path.append('..')

from collections import namedtuple
import argparse
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *

from env import EnumeratedRecommendEnv
from evaluate import dqn_evaluate
import time
import warnings
warnings.filterwarnings("ignore")
EnvDict = {
    CAL: EnumeratedRecommendEnv,
    CHA: EnumeratedRecommendEnv,
    PHO: EnumeratedRecommendEnv,
    SIN: EnumeratedRecommendEnv
    }
FeatureDict = {
    CHA: 'large_feature',
    PHO: 'large_feature',
    CAL: 'large_feature',
    SIN: 'large_feature'
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_space, hidden_size, action_space):
        super(DQN, self).__init__()
        self.state_space = state_space#command=7时候，state_space=self.cand_len_size + self.max_turn=35
        self.action_space = action_space
        self.fc1 = nn.Linear(self.state_space, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(hidden_size, self.action_space)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class Agent(object):
    def __init__(self, device, memory, state_space, hidden_size, action_space, EPS_START = 0.9, EPS_END = 0.05, EPS_DECAY = 200):
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0
        self.device = device
        self.policy_net = DQN(state_space, hidden_size, action_space).to(device)
        self.target_net = DQN(state_space, hidden_size, action_space).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = memory


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        #eps_threshold=0.9
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
           

    def optimize_model(self, BATCH_SIZE, GAMMA):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        n_states = [s for s in batch.next_state if s is not None]
        #print('main_n_states',n_states)
        non_final_next_states = torch.cat(n_states)
        #print('main_non_final_next_states',non_final_next_states)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()#optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
        #即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
        loss.backward()#反向传播计算得到每个参数的梯度
        for param in self.policy_net.parameters():# clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()#通过梯度下降执行一步参数更新
        return loss.data
    
    def save_model(self, data_name, filename, epoch_user):
        save_rl_agent(dataset=data_name, model=self.policy_net, filename=filename, epoch_user=epoch_user)
    def load_model(self, data_name, filename, epoch_user):
        model_dict = load_rl_agent(dataset=data_name, filename=filename, epoch_user=epoch_user)
        self.policy_net.load_state_dict(model_dict)


def train(popularity_dist,popular_POI_dict,args, kg, dataset, filename):
    env = EnvDict[args.data_name](popularity_dist,popular_POI_dict,args.w_pop,kg, dataset, args.data_name, seed=args.seed, max_turn=args.max_turn,
                       cand_len_size=args.cand_len_size, attr_num=args.attr_num, mode='train', command=args.command, ask_num=args.ask_num, entropy_way=args.entropy_method, fm_epoch=args.fm_epoch,reward_pre=args.reward_pre,remove_PreferAttribute=args.remove_PreferAttribute,remove_graph=args.remove_graph)
    set_random_seed(args.seed)
    state_space = env.state_space
    action_space = env.action_space
    memory = ReplayMemory(args.memory_size) 
    agent = Agent(device=args.device, memory=memory, state_space=state_space, hidden_size=args.hidden, action_space=action_space)
    tt = time.time()

    SR5, SR10, SR15, AvgT = 0, 0, 0, 0
    recall_1_main,recall_5_main,recall_10_main,MRR_1_main,MRR_5_main,MRR_10_main=0,0,0,0,0,0

    rec_success_record=None
    loss = torch.tensor(0, dtype=torch.float, device=args.device)
    start = time.time()
    #agent load policy parameters
    if args.load_rl_epoch != 0 :
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
    for i_episode in range(args.load_rl_epoch+1, args.epochs+1): #args.epochs
        #print('i_episode',i_episode)
        blockPrint()  
        #print('\n================new tuple:{}===================='.format(i_episode))
        state = env.reset()
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(args.device)#变为torch,unsqueeze增加一个维度
        for t in count():   # user  dialog
            action = agent.select_action(state)
            next_state, reward,success, done ,recall_1,recall_5,recall_10,MRR_1,MRR_5,MRR_10= env.step(action.item())
            next_state = torch.tensor([next_state], device=args.device, dtype=torch.float)
            reward = torch.tensor([reward], device=args.device, dtype=torch.float)
            if done:
                next_state = None
            
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            newloss = agent.optimize_model(args.batch_size, args.gamma)
            if newloss is not None:
                loss += newloss
            if done:
                #if reward.item() == 1:  #recommend successfully
                # if reward.item() in [1,0.99]:#ori
                if success == 1:#chen
                    recall_1_main+=recall_1
                    recall_5_main+=recall_5
                    recall_10_main+=recall_10
                    MRR_1_main+=MRR_1
                    MRR_5_main+=MRR_5
                    MRR_10_main+=MRR_10
                    if t < 5:
                        SR5 += 1
                        SR10 += 1
                        SR15 += 1   
                    elif t < 10:
                        SR10 += 1
                        SR15 += 1
                    else:
                        SR15 += 1
                #AvgT += t+1
                #original setting:
                AvgT += t
                break
        if i_episode % args.target_update == 0:#每20次优化policy network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        #enablePrint() # Enable print function为啥用这个
        if i_episode % args.observe_num == 0 and i_episode > 0:
            print('loss : {} in episode {}'.format(loss.item()/args.observe_num, i_episode))
            if i_episode % (args.observe_num * 2) == 0 and i_episode > 0:
                print('save model in episode {}'.format(i_episode))
                save_rl_model_log(dataset=args.data_name, filename=filename, epoch=i_episode, epoch_loss=loss.item()/args.observe_num, train_len=args.observe_num)
                #每400epoch保存loss
                SR = [SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT/args.observe_num]
                # print('main_SR',SR)
                
            if i_episode % (args.observe_num * 4) == 0 and i_episode > 0:#每800次数据保存一次policy network
                agent.save_model(data_name=args.data_name, filename=filename, epoch_user=i_episode) # save RL policy model
                #每800个数据保存一次RL policy model模型
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{} Total epoch_uesr:{}'.format(SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT/args.observe_num, i_episode+1))
            print('Recall1:{}, Recall5:{}, recall10:{}, MRR1:{},MRR5:{},MRR10:{}'.format(recall_1_main/args.observe_num, recall_5_main/args.observe_num, recall_10_main/args.observe_num, MRR_1_main/args.observe_num, MRR_5_main/args.observe_num,MRR_10_main/args.observe_num))
            
            print('spend time: {}'.format(time.time()-start))
            SR5, SR10, SR15, AvgT = 0, 0, 0, 0
            recall_1_main,recall_5_main,recall_10_main,MRR_1_main,MRR_5_main,MRR_10_main = 0,0,0,0,0,0
            loss = torch.tensor(0, dtype=torch.float, device=args.device)
            tt = time.time()

        if i_episode % (args.observe_num * 4) == 0 and i_episode > 0:#每800次数据，评估dqn,original setting
        #if i_episode % 1 == 0 and i_episode > 0:#每800次数据，评估dqn
            #enablePrint() 
            #每800次数据，评估dqn
            print('Evaluating on Test tuples!')
            dqn_evaluate(popularity_dist,popular_POI_dict,args, kg, dataset, agent, filename, i_episode)#i_episode=800,1600,……


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--fm_epoch', type=int, default=0, help='the epoch of FM embedding')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--target_update', type=int, default=20, help='the number of epochs to update policy parameters')
    #optimize policy network with RMSprop optimizer and update the target network every 20 epsiodes.
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--hidden', type=int, default=512, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=50000, help='size of memory ')

    parser.add_argument('--data_name', type=str, default=CAL, choices=[CAL, CHA, PHO, SIN],
                        help='One of {CAL, CHA, PHO, SIN}.')
    parser.add_argument('--entropy_method', type=str, default='weight entropy', help='entropy_method is one of {entropy, weight entropy}')
    # Although the performance of 'weighted entropy' is better, 'entropy' is an alternative method considering the time cost.
    parser.add_argument('--max_turn', type=int, default=10, help='max conversation turn')
    parser.add_argument('--cand_len_size', type=int, default=20, help='binary state size for the length of candidate items')
    parser.add_argument('--attr_num', type=int, help='the number of attributes')
    #因为从图上可以到达的attribute，并不是所有的33个都可以连接，默认20
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')#original seeting is train
    parser.add_argument('--command', type=int, default=7, help='select state vector')
    parser.add_argument('--ask_num', type=int, default=1, help='the number of features asked in a turn')
    parser.add_argument('--observe_num', type=int, default=200, help='the number of epochs to save RL model and metric')#ori=200
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='the epoch of loading RL model')
    parser.add_argument('--reward_pre', type=int, default=0, help='whether to add the pre reward')
    parser.add_argument('--remove_PreferAttribute', type=int, default=0, help='whether to remove prefered attibute for recommandation\'s input')#i.e., 1,2,3,4 vs 2,3,4
    parser.add_argument('--remove_graph', type=int, default=0, help='whether to remove graph')#remove graph
    parser.add_argument('--w_pop', type=float, default=0, help='the weight of popularity bias')#remove graph
    parser.add_argument('--top_k', type=float, default=10, help='the weight of popularity bias')#remove graph
    
    '''
    # conver_his: Conversation_history;   attr_ent: Entropy of attribute ; cand_len: the length of candidate item set 
    # command:1   self.user_embed, self.conver_his, self.attr_ent, self.cand_len
    # command:2   self.attr_ent  
    # command:3   self.conver_his
    # command:4   self.cond_len
    # command:5   self.user_embedding(same to CRIF)
    # command:6   self.conver_his, self.attr_ent, self.cand_len
    # command:7   self.conver_his, self.cand_len
    # command:8   self.conver_his, self.cand_len, self.attribute_seq(same as crm)
    # command:9   self.conver_his, self.cand_len, self.ui_embeds.shape[1] 
    # command:10  self.conver_his, self.cand_len, self.reject_feature_embed_mean, self.reject_item_embed_mean 15+20+64+64=163
    # command:12  self.conver_his, self.cand__item_len, self.cand__att_len
    '''
    import pickle


   # 指定.pkl文件的路径
    popular_POI_dict_path = 'popular_POI_dict.pkl'
    popular_POI_dict_path = 'popular_POI_dict_v2.pkl'#csv交互次数作为流行度
    #popular_POI_dict_path = 'popular_POI_dict_v3.pkl'#多少个用户作为流行度

    # 使用二进制读取模式打开文件，并加载字典
    with open(popular_POI_dict_path, 'rb') as file:
        popular_POI_dict = pickle.load(file)


    # 指定.pkl文件的路径
    popularity_dist_path = 'user_pop_unpop_distribution_dict.pkl'
    popularity_dist_path = 'user_pop_unpop_distribution_dict_v2.pkl'#csv交互次数作为流行度
    #popularity_dist_path = 'user_pop_unpop_distribution_dict_v3.pkl'#多少个用户作为流行度

    # 使用二进制读取模式打开文件，并加载字典
    with open(popularity_dist_path, 'rb') as file:
        popularity_dist = pickle.load(file)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    kg = load_kg(args.data_name)
    #reset attr_num
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    print('dataset:{}, feature_length:{}'.format(args.data_name, feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    print('args.attr_num:', args.attr_num)
    print('args.entropy_method:', args.entropy_method)
    def load_dataset(dataset):
        dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
        dataset_obj = pickle.load(open(dataset_file, 'rb'))
        return dataset_obj

    dataset = load_dataset(args.data_name)
    # f_result= open(f'{args.data_name}_{args.command}_{args.mode}_{args.epochs}_fm_epoch_{args.fm_epoch}_entropy_method_{args.entropy_method}_result831.csv', 'w', encoding='utf-8')
    # f_result.write('Success_Turn,recall_1,recall_5,recall_10,MRR_1,MRR_5,MRR_10' + '\n')
    # f_result.flush()
    # filename = 'train-data-{}-RL-command-{}-ask_method-{}-attr_num-{}-ob-{}-allepoch-{}-fm_epoch-{}'.format(
    #     args.data_name, args.command, args.entropy_method, args.attr_num, args.observe_num,args.epochs,args.fm_epoch)
    filename = 'debiasing-self.w_pop-{}-command-{}-entropy_method-{}-all_epoch-{}-fm_epoch-{}-reward_pre-{}-max_turn-{}-remove_graph-{}_v3'.format(
        args.w_pop,args.command,args.entropy_method,args.epochs,args.fm_epoch,args.reward_pre,args.max_turn,args.remove_graph)
    train(popularity_dist,popular_POI_dict,args, kg, dataset, filename)

if __name__ == '__main__':
    #f_result= open(f'{dataset}-{args.mode}-{args.command}-result.csv', 'w', encoding='utf-8')
    main()