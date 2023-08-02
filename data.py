import random as rd
import collections
from types import new_class
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from parse import parse_args
import time
import torch
from copy import deepcopy
from tqdm import tqdm


# Helper function used when loading data from files
def helper_load(filename):
    user_dict_list = {}
    item_dict = set()

    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            item_dict.update(items)
            if len(items) == 0:
                continue
            user_dict_list[user] = items
            """
            for item in items:
                if item in item_dict_list.keys():
                    item_dict_list[item].append(user)
                else:
                    item_dict_list[item] = [user]
            """
    return user_dict_list, item_dict,


def helper_load_train(filename):
    user_dict_list = {}
    item_dict = set()
    item_dict_list = {}
    trainUser, trainItem = [], []

    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            item_dict.update(items)
            # LGN
            trainUser.extend([user] * len(items))
            trainItem.extend(items)
            if len(items) == 0:
                continue
            user_dict_list[user] = items

            for item in items:
                if item in item_dict_list.keys():
                    item_dict_list[item].append(user)
                else:
                    item_dict_list[item] = [user]

    return user_dict_list, item_dict, item_dict_list, trainUser, trainItem


class Data:
    # Initialize elements stored in the Data class
    def __init__(self, args):
        # Path to the dataset files
        self.path = args.data_path + args.dataset + '/'
        self.train_file = self.path + 'train.txt'
        self.valid_file = self.path + 'valid.txt'
        self.test_ood_file = self.path + 'test_ood.txt'
        self.test_id_file = self.path + 'test_id.txt'
        self.batch_size = args.batch_size
        self.neg_sample = args.neg_sample
        self.IPStype = args.IPStype
        self.device = torch.device(args.cuda)
        self.modeltype = args.modeltype

        # Batch size during training
        # self.batch_size = args.batch_size
        # Organized Data used for evaluation
        # self.evalsets = {}

        # Number of total users and items
        self.n_users, self.n_items, self.n_observations = 0, 0, 0
        self.users = []
        self.items = []
        self.population_list = []
        self.weights = []
        # Set of all the users and items
        # self.items = set()
        # self.n_train = 0

        # Set of users and items in the validation dataset
        # self.valid_users = set()
        # self.valid_items = set()
        # Number of total observations in training  dataset

        # List of dictionaries of users and its observed items in corresponding dataset
        # {user1: [item1, item2, item3...], user2: [item1, item3, item4],...}
        # {item1: [user1, user2], item2: [user1, user3], ...}
        self.train_user_list = collections.defaultdict(list)
        self.valid_user_list = collections.defaultdict(list)
        self.test_ood_user_list = collections.defaultdict(list)
        self.test_id_user_list = collections.defaultdict(list)

        # Used to track early stopping point
        self.best_valid_recall = -np.inf
        self.best_valid_epoch, self.patience = 0, 0

        self.train_item_list = collections.defaultdict(list)
        self.Graph = None
        self.trainUser, self.trainItem, self.UserItemNet = [], [], []
        self.n_interations = 0
        # self.valid_item_list = collections.defaultdict(list)
        # self.test_ood_item_list = collections.defaultdict(list)
        # self.test_id_item_list = collections.defaultdict(list)

    # Load data from file into Data class structure
    def load_data(self):

        # Load data into structures
        self.train_user_list, train_item, self.train_item_list, self.trainUser, self.trainItem = helper_load_train(
            self.train_file)
        self.valid_user_list, valid_item = helper_load(self.valid_file)
        self.test_ood_user_list, test_ood_item = helper_load(self.test_ood_file)
        self.test_id_user_list, test_id_item = helper_load(self.test_id_file)
        self.pop_dict_list = []

        # Check if test dataset has item not appeared in train and validation

        # self.evalsets['valid'] = self.valid_user_list
        # self.evalsets['test_ood'] = self.test_ood_user_list
        # self.evalsets['test_id'] = self.test_id_user_list
        # Obtain meta data
        # self.valid_users = set(self.valid_user_list.keys())
        # self.valid_items = set(self.valid_item_list.keys())

        temp_lst = [train_item, valid_item, test_ood_item, test_id_item]

        self.users = list(set(self.train_user_list.keys()))
        self.items = list(set().union(*temp_lst))
        self.n_users = len(self.users)
        self.n_items = len(self.items)

        for i in range(self.n_users):
            self.n_observations += len(self.train_user_list[i])
            self.n_interations += len(self.train_user_list[i])
            if i in self.valid_user_list.keys():
                self.n_interations += len(self.valid_user_list[i])
            if i in self.test_id_user_list.keys():
                self.n_interations += len(self.test_id_user_list[i])
            if i in self.test_ood_user_list.keys():
                self.n_interations += len(self.test_ood_user_list[i])

        # Population matrix
        pop_dict = {}
        for item, users in self.train_item_list.items():
            pop_dict[item] = len(users) + 1
        for item in range(0, self.n_items):
            if item not in pop_dict.keys():
                pop_dict[item] = 1

            self.population_list.append(pop_dict[item])

        pop_user = {key: len(value) for key, value in self.train_user_list.items()}
        pop_item = {key: len(value) for key, value in self.train_item_list.items()}
        self.pop_item = pop_item
        sorted_pop_user = list(set(list(pop_user.values())))
        sorted_pop_item = list(set(list(pop_item.values())))
        sorted_pop_user.sort()
        sorted_pop_item.sort()
        self.n_user_pop = len(sorted_pop_user)
        self.n_item_pop = len(sorted_pop_item)
        user_idx = {}
        item_idx = {}
        for i, item in enumerate(sorted_pop_user):
            user_idx[item] = i
        for i, item in enumerate(sorted_pop_item):
            item_idx[item] = i
        self.user_pop_idx = np.zeros(self.n_users, dtype=int)
        self.item_pop_idx = np.zeros(self.n_items, dtype=int)
        for key, value in pop_user.items():
            self.user_pop_idx[key] = user_idx[value]
        for key, value in pop_item.items():
            self.item_pop_idx[key] = item_idx[value]

        self.weights = self.get_weight()

        # Item pop
        

        # Initilize negative sampling buffer
        if self.modeltype == 'PopGO':
            print("Setting up negative sampling buffer...")
            self.sample_items = np.array(self.items, dtype=int)
            self.neg_items = {}
            for key, value in tqdm(self.train_user_list.items()):
                my_data = {}
                mask = np.ones(self.n_items, dtype=bool)
                mask[np.array(value)] = 0
                my_data['data'] = self.sample_items[mask]
                np.random.shuffle(my_data['data'])
                my_data['pos'] = rd.randint(0, len(my_data['data']) - 1)
                self.neg_items[key] = my_data

            # Initilize user sampling buffer
            self.sample_users = np.array(self.users, dtype=int)
            np.random.shuffle(self.sample_users)
            self.user_pos = 0

    def get_neg_sample(self, user):
        pos = self.neg_items[user]['pos']
        data = self.neg_items[user]['data']
        l = len(data)
        if pos + self.neg_sample <= l:
            res = self.neg_items[user]['data'][pos:pos + self.neg_sample]
        else:
            new_data = np.random.permutation(self.neg_items[user]['data'])
            res = np.concatenate([data[pos:], new_data[:self.neg_sample - (l - pos)]])
            self.neg_items[user]['data'] = new_data

        self.neg_items[user]['pos'] = (pos + self.neg_sample) % l

        if self.neg_items[user]['pos'] == 0:
            np.random.shuffle(self.neg_items[user]['data'])

        return res

    def get_users(self):
        pos = self.user_pos
        if pos + self.batch_size <= self.n_users:
            res = self.sample_users[pos:pos + self.batch_size]
        else:
            new_data = np.random.permutation(self.sample_users)
            res = np.concatenate([self.sample_users[pos:], new_data[:self.batch_size - (len(self.sample_users) - pos)]])
            self.sample_users = new_data
        self.user_pos = (pos + self.batch_size) % self.n_users

        if self.user_pos == 0:
            np.random.shuffle(self.sample_users)
        return res

    def get_weight(self):

        pop = self.population_list
        pop = np.clip(pop, 1, max(pop))
        pop = pop / np.linalg.norm(pop, ord=np.inf)
        pop = 1 / pop

        if 'c' in self.IPStype:
            pop = np.clip(pop, 1, np.median(pop))
        if 'n' in self.IPStype:
            pop = pop / np.linalg.norm(pop, ord=np.inf)

        return pop

    def get_pop(self):
        pop_10_list = collections.defaultdict(list)
        pop_20_list = collections.defaultdict(list)
        pop_30_list = collections.defaultdict(list)
        pop_40_list = collections.defaultdict(list)
        pop_50_list = collections.defaultdict(list)

        pop_list = [pop_10_list, pop_20_list, pop_30_list, pop_40_list, pop_50_list]

        K_pop = np.array(self.population_list)

        # index of sorted array
        K_pop = list(np.argsort(-K_pop))

        for user in range(self.n_users):

            train = self.train_user_list[user]

            if user in self.valid_user_list.keys():
                valid = self.valid_user_list[user]
                train.extend(valid)

            my_rank = []
            num = 0
            ptr = 0

            while num < 50:
                if K_pop[ptr] not in train:
                    my_rank.append(K_pop[ptr])
                    num += 1
                ptr += 1

            for K in range(5):
                pop_list[K][user] = my_rank[:(K + 1) * 10]

        return pop_list

    # Get adjacency matrix in LightGCN

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):

        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                self.trainItem = np.array(self.trainItem)
                self.trainUser = np.array(self.trainUser)
                self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                              shape=(self.n_users, self.n_items))
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().cuda(self.device)

        return self.Graph

    # Sampling batch_size number of users from the training users
    # Each one with one positive observation and one negative observation
    def sample(self):

        if self.batch_size <= self.n_users:
            users = rd.sample(self.users, self.batch_size)
        else:
            users = [rd.choice(self.users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []

        for user in users:

            if not self.train_user_list[user]:
                pos_items.append(0)
            else:
                index = rd.randint(0, len(self.train_user_list[user]) - 1)
                pos_items.append(self.train_user_list[user][index])

            while True:

                neg_item = self.items[rd.randint(0, len(self.items) - 1)]

                if neg_item not in self.train_user_list[user]:
                    neg_items.append(neg_item)
                    break

        pos_weights = [self.weights[i] for i in pos_items]

        return users, pos_items, neg_items, pos_weights

    def sample_infonce(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.users, self.batch_size)
        else:
            users = [rd.choice(self.users) for _ in range(self.batch_size)]
        users = np.array(users, dtype=int)
        # users=self.get_users()
        users_pop = []

        pos_items, neg_items = [], []
        for user in users:
            if self.train_user_list[user] == []:
                pos_items.append(0)
            else:
                pos_items.append(rd.choice(self.train_user_list[user]))

            neg_item = self.get_neg_sample(user)
            neg_items.append(neg_item)
        neg_items = np.concatenate(neg_items)
        users_pop = self.user_pop_idx[users]
        pos_items_pop = self.item_pop_idx[pos_items]
        neg_items_pop = self.item_pop_idx[neg_items]

        return users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop


if __name__ == '__main__':
    print("here!")
    args = parse_args()
    data = Data(args)
    data.load_data()
    device = torch.device(args.cuda)
    n_batch = data.n_observations // args.batch_size + 1
    for epochs in range(50):
        for idx in tqdm(range(n_batch)):
            # Sample batch-sized data from training dataset
            users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop = data.sample_infonce()
            print(neg_items_pop.shape)
            input()
