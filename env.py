
import json
import numpy as np
import os
import random
from utils import *
import itertools
from tkinter import _flatten
from collections import Counter
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EnumeratedRecommendEnv(object):
    def __init__(self, popularity_dist,popular_POI_dict,w_pop,kg, dataset, data_name, seed=1, max_turn=15, cand_len_size=20, attr_num=20, mode='train', command=6, ask_num=1, entropy_way='weight entropy', fm_epoch=0,reward_pre=0,remove_PreferAttribute=0,remove_graph=0):
        self.data_name = data_name
        self.command = command
        self.mode = mode
        self.seed = seed
        self.max_turn = max_turn   
        self.reject_feature_embeds_size = 64
        self.reject_item_embeds_size = 64
        self.user_rej_item=[]
        self.attr_state_num = attr_num
        self.cand_len_size = cand_len_size
        self.cand_att_len_size = cand_len_size
        self.user_history_popularity_dis=popularity_dist
        
        self.popular_POI=popular_POI_dict
        self.w_pop=w_pop
        self.kg = kg
        self.dataset = dataset
        self.reward_pre = reward_pre
        self.remove_PreferAttribute = remove_PreferAttribute
        self.remove_graph = remove_graph
        self.feature_length = getattr(self.dataset, 'large_feature').value_len
        self.user_length = getattr(self.dataset, 'user').value_len
        self.item_length = getattr(self.dataset, 'POI').value_len

        self.ask_num = ask_num
        self.rec_num = 10
        self.target_item_position=-1
        self.target_location_position=-1
        self.ent_way = entropy_way
        self.reachable_feature = []   
        self.user_acc_feature = []  
        self.user_rej_feature = []  
        self.acc_samll_fea = [] 
        self.rej_samll_fea = []
        self.cand_items = []  
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0        
        self.cur_node_set = []    
        # state veactor
        self.user_embed = None
        self.user_embed_v2 = None
        self.item_embed = None
        self.reject_feature_embed = None
        self.reject_item_embed = None
        self.reject_item_embed_mean = None
        self.reject_feature_embed_mean = None
        self.conver_his = []   
        self.cand_len = []    
        self.cand_att_len = [] 
        self.attr_ent = []  
        self.attribute_seq=[]
        self.one_hot=[0] * 11
        self.one_hot_encoder_yelp=11
        self.ui_embeds_v2_size=4
        self.ui_dict,self.mydict_location_test,self.poi_all_information,self.user_feature_attention= self.__load_rl_data__(data_name, mode=mode)  # np.array [ u i weight]
        self.user_weight_dict = dict()
        self.user_items_dict = dict()
        self.file_name= 'w_pop-{}-entropy_method-{}-command-{}'.format(self.w_pop,self.ent_way,self.command)

      
        set_random_seed(self.seed) 
        if mode == 'train':
            self.__user_dict_init__() 
        elif mode == 'test':
            self.ui_array = None    
            self.u_location_array = None
            self.__test_tuple_generate__()
            self.test_num = 0
       
        embeds = load_embed(data_name, epoch=fm_epoch)
        self.ui_embeds =embeds['ui_emb']
        self.feature_emb = embeds['feature_emb']
       
        self.action_space = 2

        self.state_space_dict = {
            1: self.max_turn + self.cand_len_size + self.attr_state_num + self.ui_embeds.shape[1],
            2: self.attr_state_num,  
            3: self.max_turn,  
            4: self.cand_len_size,  
            5: self.cand_len_size + self.max_turn + self.ui_embeds_v2_size, 
            6: self.cand_len_size + self.attr_state_num + self.max_turn, 
            7: self.cand_len_size + self.max_turn,
            8: self.cand_len_size + self.max_turn + self.one_hot_encoder_yelp,
            9: self.cand_len_size + self.max_turn + self.ui_embeds.shape[1],
            10: self.cand_len_size + self.max_turn + self.reject_feature_embeds_size+ self.reject_item_embeds_size,
            11: self.cand_len_size + self.max_turn + self.one_hot_encoder_yelp,
            12: self.cand_len_size + self.max_turn+self.cand_att_len_size,
            13: self.max_turn + self.cand_att_len_size,
            14: self.cand_len_size + self.cand_att_len_size
        }
        self.state_space = self.state_space_dict[self.command]
        self.prev_reward = - 0.01
        self.reward_dict_add_reward_pre = {
            'ask_suc': 0.01+self.prev_reward,
            'ask_fail': -0.1+self.prev_reward,
            'rec_suc': 1+self.prev_reward,
            'rec_fail': -0.1+self.prev_reward,
            'until_T': -0.3,      
            'cand_none': -0.1
        }
        self.reward_dict = {
            'ask_suc': 0.01,
            'ask_fail': -0.1,
            'rec_suc': 1,
            'rec_fail': -0.1,
            'until_T': -0.3,      
            'cand_none': -0.1
        }
        self.history_dict = {
            'ask_suc': 1,
            'ask_fail': -1,
            'rec_scu': 2,
            'rec_fail': -2,
            'until_T': 0
        }
        self.attr_count_dict = dict()   

    def __load_rl_data__(self, data_name, mode):
        if mode == 'train':
            with open('UI_Interaction_data/review_dict_valid.json') as f:
                print('train_data: load RL valid data')
                mydict = json.load(f)
        elif mode == 'test':
            with open('UI_Interaction_data/review_dict_test_onlyLast.json') as f:
                mydict = json.load(f)
        with open('UI_Interaction_data/review_dict_test_location.json') as f:
            mydict_location_test=json.load(f)   
        with open('item_dict-original_tag.json') as f:
            poi_all_information = json.load(f)             
        with open('user_category_probability.json') as f:
            user_feature_attention = json.load(f)            
            return mydict,mydict_location_test,poi_all_information,user_feature_attention
        

    def __user_dict_init__(self):   
        ui_nums = 0
        for items in self.ui_dict.values():
            ui_nums += len(items)
        for user_str in self.ui_dict.keys():
            user_id = int(user_str)
            self.user_weight_dict[user_id] = len(self.ui_dict[user_str])/ui_nums

    def __test_tuple_generate__(self):
        ui_list = []
        u_location_list=[]
        for user_str, items in self.ui_dict.items():
            user_id = int(user_str)
            for item_id in items:
                ui_list.append([user_id, item_id])
        for user_str, items in self.mydict_location_test.items():
            user_id = int(user_str)
            for item_id in items:
                u_location_list.append([user_id, item_id])
        self.ui_array = np.array(ui_list)
       
        self.u_location_array = np.array(u_location_list)

    def reset(self):
       
        self.cur_conver_step = 0   
        if self.mode == 'train':
            users = list(self.user_weight_dict.keys())
            self.user_id = np.random.choice(users)
            self.target_item = np.random.choice(self.ui_dict[str(self.user_id)])
        elif self.mode == 'test':
            self.user_id = self.ui_array[self.test_num, 0]
            self.target_item = self.ui_array[self.test_num, 1]
            self.target_location = self.u_location_array[self.test_num, 1]
            self.test_num += 1
      
        self.user_acc_feature = []  
        self.user_rej_feature = []  
        self.acc_samll_fea = []
        self.rej_samll_fea = []
        self.user_rej_item = []
        self.cand_items = list(range(self.item_length))
        self.user_embed = self.ui_embeds[self.user_id].tolist()  
        self.reject_item_embed = []
        self.reject_feature_embed = []
        self.reject_item_embed_mean = [0]*64
        self.reject_feature_embed_mean = [0]*64
        self.conver_his = [0] * self.max_turn  
        self.cand_len = [self.feature_length >>d & 1 for d in range(self.cand_len_size)][::-1]  
        self.cand_att_len = [self.feature_length >>d & 1 for d in range(self.cand_len_size)][::-1] 
        self.attr_ent = [0] * self.attr_state_num  
        self.attribute_seq = [0] * self.attr_state_num
        self.one_hot=[0] * 29

        
        self._updata_reachable_feature(start='user')  
        for i in (set(self.user_acc_feature) | set(self.user_rej_feature)):
            self.get_onehot(i)
        self.reachable_feature = list(set(self.reachable_feature) - set(self.user_acc_feature))
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1

       
        self._update_cand_items(acc_feature=self.cur_node_set, rej_feature=[])
        self._update_feature_entropy()  
        
        reach_fea_score = self._feature_score()
        max_ind_list = []
        for k in range(self.ask_num):
            max_score = max(reach_fea_score)
            max_ind = reach_fea_score.index(max_score)
            reach_fea_score[max_ind] = 0
            max_ind_list.append(max_ind)
        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.pop(v - i) for i, v in enumerate(max_ind_list)]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]

        self.sim_dict = self.feature_similarity()
        self.sim_dict2 = self.sim_dict.copy()
        for f in list(set(self.user_acc_feature) | set(self.user_rej_feature)):
            if self.sim_dict is not None and f in self.sim_dict:
                self.sim_dict[f] = -1

        for f in list(set(self.user_acc_feature) | set(self.user_rej_feature)):
            if self.sim_dict2 is not None and f in self.sim_dict:
                self.sim_dict2[f] = -1
        return self._get_state()

    def _get_state(self):
        if self.command == 1:
            state = [self.user_embed, self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
        elif self.command == 2: 
            state = self.attr_ent
            state = list(_flatten(state))
        elif self.command == 3: 
            state = self.conver_his
            state = list(_flatten(state))
        elif self.command == 4: 
            state = self.cand_len
            state = list(_flatten(state))
        elif self.command == 5:  
            state = [self.conver_his, self.cand_len, self.user_embed_v2]
            state = list(_flatten(state))
        elif self.command == 6: 
            state = [self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
        elif self.command == 7: 
            state = [self.conver_his, self.cand_len]
            state = list(_flatten(state))
        elif self.command == 8: 
            state = [self.conver_his, self.cand_len, self.one_hot]
            state = list(_flatten(state))
        elif self.command == 9: 
            state = [self.conver_his, self.cand_len, self.user_embed]
            state = list(_flatten(state))
        elif self.command == 10:
            state = [self.conver_his, self.cand_len, self.reject_feature_embed_mean,self.reject_item_embed_mean]
            state = list(_flatten(state))
        elif self.command == 11: 
            list4 = [v for k, v in self.sim_dict2.items()]
            state = [self.conver_his, self.cand_len, list4]
            state = list(_flatten(state))
        elif self.command == 12:            
            state = [self.conver_his, self.cand_len, self.cand_att_len]
            state = list(_flatten(state))
        elif self.command == 13:            
            state = [self.conver_his, self.cand_att_len]
            state = list(_flatten(state))
        elif self.command == 14:             
            state = [self.cand_len, self.cand_att_len]
            state = list(_flatten(state))
        return state

    def step(self, action):  
        done = 0
        success = 0
        recall_1,recall_5,recall_10=0.0,0.0,0.0
        MRR_1,MRR_5,MRR_10=0.0,0.0,0.0
       

        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']
            self.conver_his[self.cur_conver_step-1] = self.history_dict['until_T']
           
            done = 1
        elif action == 0:  
           
            reward, done, acc_feature, rej_feature = self._ask_update()  
            self._update_cand_items(acc_feature, rej_feature)  
            if len(acc_feature):  
                self.cur_node_set = acc_feature
                self._updata_reachable_feature(start='large_feature')  
            self.reachable_feature = list(set(self.reachable_feature) - set(self.user_acc_feature))
            self.reachable_feature = list(set(self.reachable_feature) - set(self.user_rej_feature))
            self.cand_att_len = [len(self.reachable_feature) >>d & 1 for d in range(self.cand_att_len_size)][::-1]  # binary
         
            if self.command in [1, 2, 6, 7]:  
                self._update_feature_entropy()
            if len(self.reachable_feature) != 0:  
                reach_fea_score = self._feature_score()  
                max_ind_list = []
                for k in range(self.ask_num):
                    max_score = max(reach_fea_score)
                    max_ind = reach_fea_score.index(max_score)
                    reach_fea_score[max_ind] = 0
                    max_ind_list.append(max_ind)
                max_fea_id = [self.reachable_feature[i] for i in max_ind_list]

                random_ind_list = []
                if self.ent_way == 'random':
                    random_score = random.choice(reach_fea_score)
                    random_ind = reach_fea_score.index(random_score)
                    reach_fea_score[random_ind] = 0
                    random_ind_list.append(random_ind)
                    random_fea_id = [self.reachable_feature[i] for i in random_ind_list]
                    max_fea_id=random_fea_id

                if self.ent_way == 'first_POItype':
                    firstType_fea_id = 9
                    firstType_fea_id = [firstType_fea_id]
                    max_fea_id=firstType_fea_id
                
                if self.ent_way == 'no_POItype':
                    if len(self.reachable_feature) == 1 and (9 in self.reachable_feature):  
                        pass
                    else:
                        reach_fea_score = self._feature_score() 

                        removeType_ind_list = []
                        for k in range(self.ask_num):
                            removeType_score = max(reach_fea_score)
                            
                            removeType_ind = reach_fea_score.index(removeType_score)
                            reach_fea_score[removeType_ind] = 0
                            removeType_ind_list.append(removeType_ind)
                        removeType_fea_id = [self.reachable_feature[i] for i in removeType_ind_list]

                        max_fea_id=removeType_fea_id

                if self.command in [8]:
                    for i in (set(self.user_acc_feature) | set(self.user_rej_feature)):
                        self.get_onehot(i)
                [self.reachable_feature.pop(v - i) for i, v in enumerate(max_ind_list)]
                [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]
        elif action == 1:  
           
            cand_item_score = self._item_score()
            item_score_tuple = list(zip(self.cand_items, cand_item_score))
            sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
            self.cand_items, cand_item_score = zip(*sort_tuple)
            
            reward, success,done, self.target_item_position= self._recommend_updata()
            if len(self.user_rej_item) > 0:
                for item_id in self.user_rej_item:
                    self.reject_item_embed.append(self.ui_embeds[self.user_length + item_id])
                
                self.reject_item_embed_mean=list(np.array(self.reject_item_embed).mean(axis=0))       
            
            if success == 1:
               
                if  self.target_item_position > -1 and  self.target_item_position < 1:
                    recall_1=1.0
                    MRR_1=1/(self.target_item_position+1)
                    recall_5=1.0
                    MRR_5=1/(self.target_item_position+1)
                    recall_10=1.0
                    MRR_10=1/(self.target_item_position+1)
                elif  self.target_item_position > -1 and  self.target_item_position < 5:
                    recall_5=1.0
                    MRR_5=1/(self.target_item_position+1)
                    recall_10=1.0
                    MRR_10=1/(self.target_item_position+1)
                elif  self.target_item_position > -1 and  self.target_item_position < 10:
                    recall_10=1.0
                    MRR_10=1/(self.target_item_position+1)
                else:
                    print('rec fail')
            else:
                if self.command in [1, 2, 6, 7]:  
                    self._update_feature_entropy()
                print('-->Recommend fail !')
        self.cur_conver_step += 1
        if self.command in [11]:
            self.sim_dict = self.feature_similarity()
            self.sim_dict2 = self.sim_dict.copy()
            for f in list(set(self.user_acc_feature) | set(self.user_rej_feature)):
                if self.sim_dict is not None and f in self.sim_dict:
                    self.sim_dict[f] = -1

            for f in list(set(self.user_acc_feature) | set(self.user_rej_feature)):
                if self.sim_dict2 is not None and f in self.sim_dict:
                    self.sim_dict2[f] = -1
    
        return self._get_state(), reward,success, done, recall_1,recall_5,recall_10,MRR_1,MRR_5,MRR_10

    def _updata_reachable_feature(self, start='large_feature'):
        self.reachable_feature = []
        if self.remove_graph == 0:
            if start == 'user':
               
                user_like_random_fea = random.choice(self.kg.G['item'][self.target_item]['belong_to_large'])
                self.user_acc_feature.append(user_like_random_fea)  
                self.cur_node_set = [user_like_random_fea]
               

                next_reachable_feature = []
                for cur_node in self.cur_node_set:
                    fea_belong_items = list(self.kg.G['large_feature'][cur_node]['belong_to_large'])  # A-I

                    cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items))
                   
                    for item_id in cand_fea_belong_items:  
                        next_reachable_feature.append(list(self.kg.G['item'][item_id]['belong_to_large']))
                    next_reachable_feature = list(set(_flatten(next_reachable_feature)))
                self.reachable_feature = next_reachable_feature  
                self.cand_att_len = [len(self.reachable_feature) >>d & 1 for d in range(self.cand_att_len_size)][::-1]  # binary

            elif start == 'large_feature':
               
                next_reachable_feature = []
                for cur_node in self.cur_node_set:
                  
                    fea_belong_items = list(self.kg.G['large_feature'][cur_node]['belong_to_large']) # A-I
                
                    cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items))
                  
                    for item_id in cand_fea_belong_items:  
                        next_reachable_feature.append(list(self.kg.G['item'][item_id]['belong_to_large']))
                    next_reachable_feature = list(set(_flatten(next_reachable_feature)))
                self.reachable_feature = next_reachable_feature
                self.cand_att_len = [len(self.reachable_feature) >>d & 1 for d in range(self.cand_att_len_size)][::-1]  # binary
            
        elif self.remove_graph == 1:
            if start == 'user':
                user_like_random_fea = random.choice(self.kg.G['item'][self.target_item]['belong_to_large'])
                self.user_acc_feature.append(user_like_random_fea)  
            self.reachable_feature = [i for i in range(11)]


    def get_onehot(self,index_):
        self.one_hot[index_]=1

    def _feature_score(self):
        reach_fea_score = []
        for feature_id in self.reachable_feature:
            score = self.attr_ent[feature_id]
            reach_fea_score.append(score)
        return reach_fea_score

    def _item_score(self):
        cand_item_score = []
        for item_id in self.cand_items:
            item_embed = self.ui_embeds[self.user_length + item_id]
            score = 0
            if self.remove_PreferAttribute == 0:
                score += np.inner(np.array(self.user_embed), item_embed)
                prefer_embed = self.feature_emb[self.acc_samll_fea, :]  
                for i in range(len(self.acc_samll_fea)):
                    score += np.inner(prefer_embed[i], item_embed)
              
            elif self.remove_PreferAttribute == 1:
                score += np.inner(np.array(self.user_embed), item_embed)
                
            else:
                print('please set remove_PreferAttribute')
            cand_item_score.append(score)
        return cand_item_score

    def _ask_update(self):
        '''
        :return: reward, acc_feature, rej_feature
        '''
        done = 0

        feature_groundtrue = self.kg.G['item'][self.target_item]['belong_to_large']

        remove_acced_reachable_fea = self.reachable_feature.copy()   

        acc_feature = list(set(remove_acced_reachable_fea[:self.ask_num]) & set(feature_groundtrue))
        rej_feature = list(set(remove_acced_reachable_fea[:self.ask_num]) - set(acc_feature))

       
        self.user_acc_feature.append(acc_feature)
        self.user_acc_feature = list(set(_flatten(self.user_acc_feature)))
        self.user_rej_feature.append(rej_feature)
        self.user_rej_feature = list(set(_flatten(self.user_rej_feature)))

        if len(self.user_rej_feature) > 0 :
           
            for feature_id in self.user_rej_feature:
               
                self.reject_feature_embed.append(self.feature_emb[feature_id])
            
            self.reject_feature_embed_mean=list(np.array(self.reject_feature_embed).mean(axis=0))


        reward = None
        
        rec_list=[]
       
        user_id_his_pop_dis=self.user_history_popularity_dis[str(self.user_id)]
        
        short_num=0
        for i in self.cand_items[: self.rec_num]:
            if i in self.popular_POI:
                short_num+=1
        rec_list=[short_num/len(self.cand_items[: self.rec_num]),1-short_num/len(self.cand_items[: self.rec_num])]
        js_result = js_divergence(user_id_his_pop_dis, rec_list)
        rank_reward=1/(math.log(self.cand_items.index(self.target_item)+1+1))
        pop_item_rank_reward=0
        for i in self.popular_POI:
            if i in self.cand_items:
                pop_item_rank_reward+=1/math.log(1+1+self.cand_items.index(i))
        
        popcorn_pop_bias=pop_item_rank_reward*(short_num/len(self.cand_items[: self.rec_num]))

        if len(acc_feature):
            if self.reward_pre==0:
                reward = self.reward_dict['ask_suc']-self.w_pop*js_result
            elif self.reward_pre==1:
                reward = self.reward_dict_add_reward_pre['ask_suc']-self.w_pop*js_result
            else:
                print('reward error')
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']   
        else:
            if self.reward_pre==0:
                reward = self.reward_dict['ask_fail']-self.w_pop*js_result
            elif self.reward_pre==1:
                reward = self.reward_dict_add_reward_pre['ask_fail']-self.w_pop*js_result
            else:
                print('reward error')
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']  

        if self.cand_items == []:  
            done = 1
            reward = self.reward_dict['cand_none']-self.w_pop*js_result
        return reward, done, acc_feature, rej_feature

    def _update_cand_items(self, acc_feature, rej_feature):

        small_feature_groundtrue = self.kg.G['item'][self.target_item]['belong_to']  # TODO 
        if len(acc_feature):    #accept large_feature
            for feature_id in acc_feature:
                feature_small_ids = self.kg.G['large_feature'][feature_id]['link_to_feature']
                for small_id in feature_small_ids:
                    if small_id in small_feature_groundtrue:  
                        self.acc_samll_fea.append(small_id)
                        feature_items = self.kg.G['feature'][small_id]['belong_to']
                        self.cand_items = set(self.cand_items) & set(feature_items)   
                    else:  
                        self.rej_samll_fea.append(small_id)  

            self.cand_items = list(self.cand_items)

        self.cand_len = [len(self.cand_items) >>d & 1 for d in range(self.cand_len_size)][::-1]  


    def _recommend_updata(self):
       
        recom_items = self.cand_items[: self.rec_num]    
        reward=None
        success=0
       
        rec_list=[]
       
        user_id_his_pop_dis=self.user_history_popularity_dis[str(self.user_id)]
       
        short_num=0
        for i in recom_items:
            if i in self.popular_POI:
                short_num+=1
        rec_list=[short_num/len(recom_items),1-short_num/len(recom_items)]
       
        js_result = js_divergence(user_id_his_pop_dis, rec_list)
        rank_reward=1/(math.log(self.cand_items.index(self.target_item)+1+1))
       
        if self.target_item in recom_items:
           
            if self.mode =='test':
               
                PATH = './tmp/fairness/' + self.file_name + 'top_10_result.txt'
                
                with open(PATH, 'a') as f:
                    f.write(str(self.user_id) + ","+ str(self.target_item) + "," + str(recom_items) + "\n")
                prediction_list=[]
                for i in recom_items:
                    if self.poi_all_information[str(i)]['POI'] == 'Combined':
                    
                        stars = self.poi_all_information[str(i)]["stars"]
                        location_ids = self.poi_all_information[str(i)]["Location_id"]
                        max_star = max(stars)
                       
                        max_star_locations = [loc_id for loc_id, star in zip(location_ids, stars) if star == max_star]
                       
                        if len(max_star_locations)>1:
                            random_element = random.choice(max_star_locations)
                            prediction_list.append(random_element)
                        else:
                            prediction_list.append(max_star_locations[0])
                    else:
                        prediction_list.append(i)
                if self.target_location in prediction_list:
                    success = 1
                    reward = self.reward_dict['rec_suc']-self.w_pop*js_result
                    self.conver_his[self.cur_conver_step] = self.history_dict['rec_scu'] 
                    done = 1
                    self.target_location_position=prediction_list.index(self.target_location)
                    self.target_item_position=self.target_location_position

            success = 1
            if self.reward_pre == 0:

                reward = self.reward_dict['rec_suc']-self.w_pop*js_result
            elif self.reward_pre == 1:
               
                reward = self.reward_dict_add_reward_pre['rec_suc']-self.w_pop*js_result
            else:
                print('reward error')
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_scu'] 
            done = 1
            self.target_item_position=recom_items.index(self.target_item)
            #
        else:
            success = 0
            recom_items=list(recom_items)
           
            self.user_rej_item.append(recom_items)
            self.user_rej_item=list(set(list(_flatten(self.user_rej_item))))
            if self.reward_pre == 0:
                
                reward = self.reward_dict['rec_fail']-self.w_pop*js_result
            elif self.reward_pre == 1:
               
                reward = self.reward_dict_add_reward_pre['rec_fail']-self.w_pop*js_result
                print('reward_fail',reward)
            else:
                print('reward error')
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_fail']  
            self.cand_items = self.cand_items[self.rec_num:] 
            self.cand_len = [len(self.cand_items) >> d & 1 for d in range(self.cand_len_size)][::-1] 
            done = 0
        return reward, success,done,self.target_item_position

    def _update_feature_entropy(self):
        if self.ent_way == 'entropy':
            cand_items_fea_list = []
            
            for item_id in self.cand_items:
                cand_items_fea_list.append(list(self.kg.G['item'][item_id]['belong_to']))
            cand_items_fea_list = list(_flatten(cand_items_fea_list))
            self.attr_count_dict = dict(Counter(cand_items_fea_list))
            self.attr_ent = [0] * self.attr_state_num  
            real_ask_able_large_fea = self.reachable_feature
            for large_fea_id in real_ask_able_large_fea:
                large_ent = 0
                small_feature = list(self.kg.G['large_feature'][large_fea_id]['link_to_feature'])
                small_feature_in_cand = list(set(small_feature) & set(self.attr_count_dict.keys()))

                for fea_id in small_feature_in_cand:
                    p1 = float(self.attr_count_dict[fea_id]) / len(self.cand_items)
                    p2 = 1.0 - p1
                    if p1 == 1:
                        large_ent += 0
                    else:
                        ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                        large_ent += ent
                self.attr_ent[large_fea_id] =large_ent
        elif self.ent_way == 'weight entropy':
            cand_items_fea_list = []
            self.attr_count_dict = {}
            cand_item_score = self._item_score()
            cand_item_score_sig = self.sigmoid(cand_item_score)  

            for score_ind, item_id in enumerate(self.cand_items):
                cand_items_fea_list = list(self.kg.G['item'][item_id]['belong_to'])
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = 0
                    self.attr_count_dict[fea_id] += cand_item_score_sig[score_ind]

            self.attr_ent = [0] * self.attr_state_num  
            real_ask_able_large_fea = self.reachable_feature
            sum_score_sig = sum(cand_item_score_sig)
            for large_fea_id in real_ask_able_large_fea:
                large_ent = 0
                small_feature = list(self.kg.G['large_feature'][large_fea_id]['link_to_feature'])
                small_feature_in_cand = list(set(small_feature) & set(self.attr_count_dict.keys()))

                for fea_id in small_feature_in_cand:
                    p1 = float(self.attr_count_dict[fea_id]) / sum_score_sig
                    p2 = 1.0 - p1
                    if p1 == 1 or p1 <= 0:
                        large_ent += 0
                    else:
                        ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                        large_ent += ent
                self.attr_ent[large_fea_id] = large_ent
        elif self.ent_way == 'weight_entropy_attention':
            cand_items_fea_list = []
            self.attr_count_dict = {}
            cand_item_score = self._item_score()
            cand_item_score_sig = self.sigmoid(cand_item_score)  # sigmoid(score)

            for score_ind, item_id in enumerate(self.cand_items):
                cand_items_fea_list = list(self.kg.G['item'][item_id]['belong_to'])
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = 0
                    self.attr_count_dict[fea_id] += cand_item_score_sig[score_ind]

            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able_large_fea = self.reachable_feature
            sum_score_sig = sum(cand_item_score_sig)
            print('str(self.user_id)',str(self.user_id))
           
            user_lar_feature_attention=dict()
            for large_fea_id in real_ask_able_large_fea:
                large_ent = 0
                large_attention=0
                large_attention_sum=0
                small_feature = list(self.kg.G['large_feature'][large_fea_id]['link_to_feature'])
                small_feature_in_cand = list(set(small_feature) & set(self.attr_count_dict.keys()))

               
                if str(self.user_id) in self.user_feature_attention.keys():
                    user_lar_feature_attention = self.user_feature_attention[str(self.user_id)]
                
                

                for fea_id in small_feature_in_cand:
                                    
                    p1 = float(self.attr_count_dict[fea_id]) / sum_score_sig  * (user_lar_feature_attention.get(str(fea_id), 0)+0.1)# new
                    p2 = 1.0 - p1
                    if p1 == 1 or p1 <= 0:
                        large_ent += 0
                    else:
                        ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                        large_ent += ent
                self.attr_ent[large_fea_id] = large_ent
    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()
    #for variant: RC provide user preference on each attribute to CC

    def feature_similarity(self):
        feature_size=0
        preference_matrix_all = self.ui_embeds[[self.user_id]].tolist()
        if len(self.user_acc_feature) > 0:
            prefer_embed = self.feature_emb[self.user_acc_feature, :]
            preference_matrix_all = np.concatenate((prefer_embed, preference_matrix_all), axis=0)
        result_dict = dict()
        if self.data_name in ['CAL','CHA','SIN','PHO']:
            feature_size=11
        for i in range(feature_size):
            feature_matrix = self.feature_emb[i]
            feature_matrix = feature_matrix.reshape(-1, 64)
            cosine_result = cosine_similarity(feature_matrix, preference_matrix_all)
            cosine_result = np.sum(cosine_result, axis=1)
            normalize_factor = 5.0  
            result_dict[i] = normalize_factor * float(cosine_result[0]) / len(self.user_acc_feature)
        return result_dict
