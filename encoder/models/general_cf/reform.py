import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop
import numpy as np
import random
import pickle
from scipy.sparse import coo_matrix

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class ReFORM(BaseModel):
    def __init__(self, data_handler):
        super(ReFORM, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.adj2 = data_handler.torch_adj2
        self.keep_rate = configs['model']['keep_rate']
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.all_items = t.tensor(range(self.item_num))

        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False
        self.device = configs['device']

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.k = configs['model']['k']

        self.row_indices = self.adj2.indices()[0]  
        self.col_indices = self.adj2.indices()[1] 
        self.max_row = self.adj2.indices()[0].max().item()
        self.max_col = self.adj2.indices()[1].max().item()

        # semantic-embeddings
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda() 
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()

        self.q_u = nn.Linear(self.usrprf_embeds.shape[2], self.embedding_size)
        self.k_i = nn.Linear(self.usrprf_embeds.shape[2], self.embedding_size)
        self.v_u = nn.Linear(self.usrprf_embeds.shape[2], self.embedding_size)

        self.q_i = nn.Linear(self.itmprf_embeds.shape[2], self.embedding_size) 
        self.k_u = nn.Linear(self.itmprf_embeds.shape[2], self.embedding_size)
        self.v_i = nn.Linear(self.itmprf_embeds.shape[2], self.embedding_size)

        self.masking = t.zeros([self.usrprf_embeds.shape[1], self.usrprf_embeds.shape[2]]).cuda() 


    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)
    
    def cross_attention(self, query, key):
        scores = t.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attn_weights = t.softmax(scores, dim=-1)

        return attn_weights
    
    def get_batch_interacted_items(self, batch_user_ids):
        batch_interacted_items = []
        for user_id in batch_user_ids:
            interacted_items = self.col_indices[self.row_indices == user_id].tolist()

            # sampling
            if len(interacted_items) > self.k:
                interacted_items = random.sample(interacted_items, self.k)
            batch_interacted_items.append(interacted_items)

        target_shape = (batch_user_ids.size(0), self.k)
        pad_value = -1  

        batch_interacted_items_tensor = t.full(target_shape, pad_value, dtype=t.long)
        for i, row in enumerate(batch_interacted_items):
            batch_interacted_items_tensor[i, :len(row)] = t.tensor(row)

        return batch_interacted_items_tensor
    
    def get_batch_interacted_users(self, batch_item_ids):
        batch_interacted_users = []
        for item_id in batch_item_ids:
            interacted_users = self.row_indices[self.col_indices == item_id].tolist()

            # sampling
            if len(interacted_users) > self.k:
                interacted_users = random.sample(interacted_users, self.k)
            batch_interacted_users.append(interacted_users)

        target_shape = (batch_item_ids.size(0), self.k)
        pad_value = -1

        batch_interacted_users_tensor = t.full(target_shape, pad_value, dtype=t.long)
        for i, row in enumerate(batch_interacted_users):
            batch_interacted_users_tensor[i, :len(row)] = t.tensor(row)

        return batch_interacted_users_tensor

    def cross_attention_user(self, batch_users):
        user_tensor = self.usrprf_embeds[batch_users]
        user_query = self.q_u(user_tensor)  
        user_value = self.v_u(user_tensor) 

        batch_interacted_items_tensor = self.get_batch_interacted_items(batch_users) 
        item_tensor = self.itmprf_embeds[batch_interacted_items_tensor]

        mask = item_tensor == -1
        mask_positions = t.where(mask)
        
        if not len(mask_positions[0]) == 0:
            item_tensor[mask_positions] = self.masking

        item_keys = self.k_i(item_tensor) 
        item_keys_t = item_keys.transpose(-2, -1)
        attention_scores = t.einsum("bqd,bndk->bnqk", user_query, item_keys_t) / (user_query.size(-1) ** 0.5)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights_mean = attention_weights.max(dim=1).values
        # attention_weights_mean = attention_weights.mean(dim=1)       # mean pooling

        user_weighted_val = t.matmul(attention_weights_mean, user_value)
        user_weighted_val = t.mean(user_weighted_val, dim=1)

        return user_weighted_val
   
    def cross_attention_item(self, batch_items):
        item_tensor = self.itmprf_embeds[batch_items]
        item_query = self.q_i(item_tensor)  
        item_value = self.v_i(item_tensor) 

        batch_interacted_users_tensor = self.get_batch_interacted_users(batch_items) 
        user_tensor = self.usrprf_embeds[batch_interacted_users_tensor]

        mask = user_tensor == -1
        mask_positions = t.where(mask)

        if not len(mask_positions[0]) == 0:
            user_tensor[mask_positions] = self.masking

        user_keys = self.k_u(user_tensor)
        user_keys_t = user_keys.transpose(-2, -1) 
        attention_scores = t.einsum("bqd,bndk->bnqk", item_query, user_keys_t) / (item_query.size(-1) ** 0.5)

        attention_weights = F.softmax(attention_scores, dim=-1)  
        attention_weights_mean = attention_weights.max(dim=1).values 
        # attention_weights_mean = attention_weights.mean(dim=1)       # mean pooling

        item_weighted_val = t.matmul(attention_weights_mean, item_value)
        item_weighted_val = t.mean(item_weighted_val, dim=1)

        return item_weighted_val

    def forward(self, adj=None, keep_rate=1.0):   
            if adj is None:
                adj = self.adj
            if not self.is_training and self.final_embeds is not None:
                return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
            embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
            embeds_list = [embeds]
            if self.is_training:
                adj = self.edge_dropper(adj, keep_rate)
            for i in range(self.layer_num):
                embeds = self._propagate(adj, embeds_list[-1])
                embeds_list.append(embeds)
            embeds = sum(embeds_list)

            user_gcn_embeds = embeds[:self.user_num]
            item_gcn_embeds = embeds[self.user_num:]

            return user_gcn_embeds, item_gcn_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        anc_prf_embeds = self.cross_attention_user(ancs)
        pos_prf_embeds = self.cross_attention_item(poss)
        neg_prf_embeds = self.cross_attention_item(negs)

        anc_cat_embeds = t.concat([anc_embeds, anc_prf_embeds], axis=1)
        pos_cat_embeds = t.concat([pos_embeds, pos_prf_embeds], axis=1)
        neg_cat_embeds = t.concat([neg_embeds, neg_prf_embeds], axis=1)

        bpr_loss = cal_bpr_loss(anc_cat_embeds, pos_cat_embeds, neg_cat_embeds) / anc_cat_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)

        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        
        self.is_training = False
        pck_users, train_mask = batch_data

        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]

        pck_prf_embeds = self.cross_attention_user(pck_users)
        all_itm_embeds = self.cross_attention_item(self.all_items)
        anc_cat_embeds = t.concat([pck_user_embeds, pck_prf_embeds], axis=1)
        pos_cat_embeds = t.concat([item_embeds, all_itm_embeds], axis=1)

        full_preds = anc_cat_embeds @ pos_cat_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
