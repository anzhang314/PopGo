import torch
import torch.nn as nn
import numpy as np


class MF(nn.Module):
    def __init__(self, args, data):
        super(MF, self).__init__()
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.decay = args.regs
        self.device = torch.device(args.cuda)
        self.saveID = args.saveID

        self.train_user_list = data.train_user_list
        self.valid_user_list = data.valid_user_list
        self.population_list = data.population_list

        self.embed_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embed_item = nn.Embedding(self.n_items, self.emb_dim)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

    # Prediction function used when evaluation
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        users = self.embed_user(torch.tensor(users).cuda(self.device))
        items = torch.transpose(self.embed_item(torch.tensor(items).cuda(self.device)), 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()


class LGN(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph()
        self.n_layers = args.n_layers

    def compute(self):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items

    def forward(self, users, pos_items, neg_items, pos_weights):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()

class BPRMF(MF):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores))

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss


class BCEMF(MF):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        mf_loss = torch.mean(torch.negative(torch.log(torch.sigmoid(pos_scores) + 1e-9))
                             + torch.negative(torch.log(1 - torch.sigmoid(neg_scores) + 1e-9)))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss


class IPSMF(MF):
    def __init__(self, args, data):
        super().__init__(args, data)

    # Inputs are tensorflow
    def forward(self, users, pos_items, neg_items, pos_weights):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.mul(torch.log(torch.sigmoid(pos_scores - neg_scores)), pos_weights)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss


class MACR(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.alpha = args.alpha
        self.beta = args.beta
        self.w = nn.Embedding(self.emb_dim, 1)
        self.w_user = nn.Embedding(self.emb_dim, 1)
        nn.init.xavier_normal_(self.w.weight)
        nn.init.xavier_normal_(self.w_user.weight)

        self.pos_item_scores = torch.empty((self.batch_size, 1))
        self.neg_item_scores = torch.empty((self.batch_size, 1))
        self.user_scores = torch.empty((self.batch_size, 1))

        self.rubi_c = args.c * torch.ones([1]).cuda(self.device)

    def forward(self, users, pos_items, neg_items):
        # Original scores
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        # Item module and User module
        self.pos_item_scores = torch.matmul(pos_items, self.w.weight)
        self.neg_item_scores = torch.matmul(neg_items, self.w.weight)
        self.user_scores = torch.matmul(users, self.w_user.weight)

        # fusion
        pos_scores = pos_scores * torch.sigmoid(self.pos_item_scores) * torch.sigmoid(self.user_scores)
        neg_scores = neg_scores * torch.sigmoid(self.neg_item_scores) * torch.sigmoid(self.user_scores)

        # loss
        mf_loss_ori = torch.mean(torch.negative(torch.log(torch.sigmoid(pos_scores) + 1e-10)) + torch.negative(
            torch.log(1 - torch.sigmoid(neg_scores) + 1e-10)))

        mf_loss_item = torch.mean(
            torch.negative(torch.log(torch.sigmoid(self.pos_item_scores) + 1e-10)) + torch.negative(
                torch.log(1 - torch.sigmoid(self.neg_item_scores) + 1e-10)))

        mf_loss_user = torch.mean(torch.negative(torch.log(torch.sigmoid(self.user_scores) + 1e-10)) + torch.negative(
            torch.log(1 - torch.sigmoid(self.user_scores) + 1e-10)))

        mf_loss = mf_loss_ori + self.alpha * mf_loss_item + self.beta * mf_loss_user

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        users = self.embed_user(torch.tensor(users).cuda(self.device))
        items = torch.transpose(self.embed_item(torch.tensor(items).cuda(self.device)), 0, 1)
        rate_batch = torch.matmul(users, items)

        item_scores = torch.matmul(torch.transpose(items,0,1), self.w.weight)
        user_scores = torch.matmul(users, self.w_user.weight)

        rubi_rating_both = (rate_batch - self.rubi_c) * torch.transpose(torch.sigmoid(item_scores),0,1) * torch.sigmoid(user_scores)

        return rubi_rating_both.cpu().detach().numpy()

class CausE(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.cf_pen = args.cf_pen
        self.embed_item_ctrl = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.xavier_normal_(self.embed_item_ctrl.weight)
    
    
    def forward(self, users, pos_items, neg_items, item_embed, control_embed):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)   #users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer/self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores)+1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        # counterfactual loss
        #cf_loss = torch.mean(tf.reduce_sum(tf.abs(tf.subtract(item_embed, control_embed)), axis=1))
        cf_loss = torch.sqrt(torch.sum(torch.square(torch.subtract(0.5 * torch.norm(item_embed) ** 2, 0.5 * torch.norm(control_embed) ** 2))))
        cf_loss = cf_loss * self.cf_pen #/ self.batch_size

        return mf_loss, reg_loss, cf_loss



class PopGO(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.n_users_pop=data.n_user_pop
        self.n_items_pop=data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)
    
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        all_users, all_items = self.compute()

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]
        users_pop = self.embed_user_pop(users_pop)
        pos_items_pop = self.embed_item_pop(pos_items_pop)
        neg_items_pop = self.embed_item_pop(neg_items_pop)

        tiled_usr=torch.tile(users,[1,self.neg_sample]).reshape([-1,self.emb_dim])
        tiled_usr_pop=torch.tile(users_pop,[1,self.neg_sample]).reshape([-1,self.emb_dim])
        user_n2=torch.norm(users,p=2,dim=1)
        user_pop_n2=torch.norm(users_pop,p=2,dim=1)
        tiled_usr_n2=torch.norm(tiled_usr,p=2,dim=1)
        tiled_usr_pop_n2=torch.norm(tiled_usr_pop,p=2,dim=1)
        pos_item_n2=torch.norm(pos_items,p=2,dim=1)
        neg_item_n2=torch.norm(neg_items,p=2,dim=1)
        neg_item_pop_n2=torch.norm(neg_items_pop,p=2,dim=1)
        pos_item_pop_n2=torch.norm(pos_items_pop,p=2,dim=1)

        pos_item_pop_prod=torch.sum(torch.mul(users_pop,pos_items_pop),dim=1)
        neg_item_pop_prod=torch.sum(torch.mul(tiled_usr_pop,neg_items_pop),dim=1)
        pos_item_prod=torch.sum(torch.mul(users,pos_items),dim=1)
        neg_item_prod=torch.sum(torch.mul(tiled_usr,neg_items),dim=1)

        # option 1: sigmoid dot-product
        # pos_item_score=tf.sigmoid(pos_item_prod)
        # neg_item_score=tf.sigmoid(neg_item_prod)
        # pos_item_pop_score=tf.sigmoid(pos_item_pop_prod)/self.tau2
        # neg_item_pop_score=tf.sigmoid(neg_item_pop_prod)/self.tau2


        # option 2: cosine similarity
        pos_item_score=pos_item_prod/user_n2/pos_item_n2
        neg_item_score=neg_item_prod/tiled_usr_n2/neg_item_n2
        pos_item_pop_score=pos_item_pop_prod/user_pop_n2/pos_item_pop_n2/self.tau2
        neg_item_pop_score=neg_item_pop_prod/tiled_usr_pop_n2/neg_item_pop_n2/self.tau2

        # pure infonce loss
        #pos_item_score_mf_exp=torch.exp(pos_item_score/self.tau1)
        #neg_item_score_mf_exp=torch.sum(torch.exp(torch.reshape(neg_item_score/self.tau,[-1,self.neg_sample])),dim=1)
        #loss_mf=torch.mean(torch.negative(torch.log(pos_item_score_mf_exp/(pos_item_score_mf_exp+neg_item_score_mf_exp))))


        neg_item_pop_score_exp=torch.sum(torch.exp(torch.reshape(neg_item_pop_score,[-1,self.neg_sample])),dim=1)
        pos_item_pop_score_exp=torch.exp(pos_item_pop_score)
        loss2=self.w_lambda*torch.mean(torch.negative(torch.log(pos_item_pop_score_exp/(pos_item_pop_score_exp+neg_item_pop_score_exp))))

        weighted_pos_item_score=torch.mul(pos_item_score,torch.sigmoid(pos_item_pop_prod))/self.tau1
        weighted_neg_item_score=torch.mul(neg_item_score,torch.sigmoid(neg_item_pop_prod))/self.tau1

        neg_item_score_exp=torch.sum(torch.exp(torch.reshape(weighted_neg_item_score,[-1,self.neg_sample])),dim=1)
        pos_item_score_exp=torch.exp(weighted_pos_item_score)
        loss1=(1-self.w_lambda)*torch.mean(torch.negative(torch.log(pos_item_score_exp/(pos_item_score_exp+neg_item_score_exp))))

        regularizer1 = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer1 = regularizer1/self.batch_size

        regularizer2= 0.5 * torch.norm(users_pop) ** 2 + 0.5 * torch.norm(pos_items_pop) ** 2 + 0.5 * torch.norm(neg_items_pop) ** 2 
        regularizer2  = regularizer2/self.batch_size
        reg_loss = self.decay * (regularizer1+regularizer2)

        reg_loss_freeze=self.decay * (regularizer2)
        reg_loss_norm=self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm
    
    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)


    
    





class IPSLGN(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items, pos_weights):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.mul(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10), pos_weights)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss
