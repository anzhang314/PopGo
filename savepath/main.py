import random
import re
random.seed(101)
import itertools
import torch
import time
import numpy as np
from tqdm import tqdm
from evaluator import ProxyEvaluator
import collections
import os
from data import Data
from parse import parse_args
from model import BPRMF, BCEMF, IPSMF, IPSLGN, LGN, MACR, PopGO


def merge_user_list(user_lists):
    out = collections.defaultdict(list)
    for user_list in user_lists:
        for key, item in user_list.items():
            out[key] = out[key] + item
    return out


def save_checkpoint(model, epoch, checkpoint_dir, buffer, max_to_keep=10):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)
    buffer.append(filename)
    if len(buffer)>max_to_keep:
        os.remove(buffer[0])
        del(buffer[0])

    return buffer


def restore_checkpoint(model, checkpoint_dir, force=False, pretrain=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model and the current epoch.
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    if not cp_files:
        print('No saved model parameters found')
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0,

    epoch_list = []

    regex = re.compile(r'\d+')

    for cp in cp_files:
        epoch_list.append([int(x) for x in regex.findall(cp)][0])

    epoch = max(epoch_list)

   
    if not force:
        print("Which epoch to load from? Choose in range [0, {})."
              .format(epoch), "Enter 0 to train from scratch.")
        print(">> ", end='')
        inp_epoch = int(input())
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0,
    else:
        print("Which epoch to load from? Choose in range [0, {}).".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(0, epoch):
            raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir,
                            'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

    print("Loading from checkpoint {}?".format(filename))

    checkpoint = torch.load(filename)

    try:
        if pretrain:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
              .format(checkpoint['epoch']))
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch


def restore_best_checkpoint(epoch, model, checkpoint_dir):
    """
    Restore the best performance checkpoint
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    filename = os.path.join(checkpoint_dir,
                            'epoch={}.checkpoint.pth.tar'.format(epoch))

    print("Loading from checkpoint {}?".format(filename))

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['state_dict'])
    print("=> Successfully restored checkpoint (trained for {} epochs)"
          .format(checkpoint['epoch']))

    return model


def clear_checkpoint(checkpoint_dir):
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def evaluation(args, data, model, epoch, base_path, evaluator, name="valid"):
    # Evaluate with given evaluator

    ret, _ = evaluator.evaluate(model)

    n_ret = {"recall": ret[1], "hit_ratio": ret[5], "precision": ret[0], "ndcg": ret[3], "mrr":ret[4], "map":ret[2]}

    perf_str = name+':{}'.format(n_ret)
    print(perf_str)
    with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(perf_str + "\n")
    # Check if need to early stop (on validation)
    is_best=False
    early_stop=False
    if name=="valid":
        if ret[1] > data.best_valid_recall:
            data.best_valid_epoch = epoch
            data.best_valid_recall = ret[1]
            data.patience = 0
            is_best=True
        else:
            data.patience += 1
            if data.patience >= args.patience:
                print_str = "The best performance epoch is % d " % data.best_valid_epoch
                print(print_str)
                early_stop=True

    return is_best, early_stop


def Item_pop(args, data, model):

    for K in range(5):

        eval_pop = ProxyEvaluator(data, data.train_user_list, data.pop_dict_list[K], top_k=[(K+1)*10],
                                   dump_dict=merge_user_list([data.train_user_list, data.valid_user_list]))

        ret, _ = eval_pop.evaluate(model)

        print_str = "Overlap for K = % d is % f" % ( (K+1)*10, ret[1] )

        print(print_str)

        with open('stats_{}.txt'.format(args.saveID), 'a') as f:
            f.write(print_str + "\n")


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)



if __name__ == '__main__':

    start = time.time()

    args = parse_args()
    data = Data(args)
    data.load_data()
    device="cuda:"+str(args.cuda)
    device = torch.device(args.cuda)
    base_path = './weights/{}/{}/{}/'.format(args.dataset, args.modeltype, args.saveID)
    checkpoint_buffer=[]
    freeze_epoch=args.freeze_epoch if args.modeltype=="PopGO" else 0
    ensureDir(base_path)

    eval_valid = ProxyEvaluator(data, data.train_user_list, data.valid_user_list, top_k=[20])
    eval_test_ood = ProxyEvaluator(data, data.train_user_list, data.test_ood_user_list, top_k=[20],
                                   dump_dict=merge_user_list(
                                       [data.train_user_list, data.valid_user_list, data.test_id_user_list]))
    eval_test_id = ProxyEvaluator(data, data.train_user_list, data.test_id_user_list, top_k=[20],
                                dump_dict=merge_user_list(
                                    [data.train_user_list, data.valid_user_list, data.test_ood_user_list]))
    
    evaluators=[eval_test_id, eval_test_ood, eval_valid]
    eval_names=["test_id", "test_ood", "valid"]

    if args.modeltype == 'BPRMF':
        model = BPRMF(args, data)
    if args.modeltype == 'BCEMF':
        model = BCEMF(args, data)
    if args.modeltype == 'IPSMF':
        model = IPSMF(args, data)
    if args.modeltype == 'MACRMF':
        model = MACR(args, data)
    if args.modeltype == 'IPSLGN':
        model = IPSLGN(args, data)
    if args.modeltype == 'LGN':
        model = LGN(args, data)
    if args.modeltype == 'PopGO':
        model = PopGO(args, data)

    model.cuda(device)

    model, start_epoch = restore_checkpoint(model, base_path)

    n_batch = data.n_observations // args.batch_size + 1

    flag = False

    # Training
    for epoch in range(start_epoch, args.epoch):

        # If the early stopping has been reached, restore to the best performance model
        if flag:
            break

        optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0
        running_loss1, running_loss2=0,0
        running_cf_loss=0
        t1=time.time()
        # Running through several batches of data
        for idx in tqdm(range(n_batch)):
            
            # Sample batch-sized data from training dataset
            if args.modeltype == 'PopGO':
                users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop=data.sample_infonce()
            else:
                users, pos_items, neg_items, pos_weights = data.sample()
                pos_weights = torch.tensor(pos_weights).cuda(device)

            if args.modeltype == 'IPSLGN' or args.modeltype =='LGN':
                users = torch.tensor(users).cuda(device)
                pos_items = torch.tensor(pos_items).cuda(device)
                neg_items = torch.tensor(neg_items).cuda(device)
                mf_loss, reg_loss = model(users, pos_items, neg_items, pos_weights)
            elif args.modeltype == 'PopGO':
                users = torch.tensor(users).cuda(device)
                pos_items = torch.tensor(pos_items).cuda(device)
                neg_items = torch.tensor(neg_items).cuda(device)
                users_pop = torch.tensor(users_pop).cuda(device)
                pos_items_pop = torch.tensor(pos_items_pop).cuda(device)
                neg_items_pop = torch.tensor(neg_items_pop).cuda(device)
                loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm = model(users,pos_items,neg_items,users_pop,pos_items_pop,neg_items_pop)
            else:
                # Get the slice of embedded data and convert to GPU
                users = model.embed_user(torch.tensor(users).cuda(device))
                pos_items = model.embed_item(torch.tensor(pos_items).cuda(device))
                neg_items = model.embed_item(torch.tensor(neg_items).cuda(device))
                
                if args.modeltype == 'IPSMF':
                    mf_loss, reg_loss = model(users, pos_items, neg_items, pos_weights)
                else:
                    mf_loss, reg_loss = model(users, pos_items, neg_items)
            
            if args.modeltype == "PopGO":
                if epoch<args.freeze_epoch:
                    loss =  loss2 + reg_loss_freeze
                else:
                    model.freeze_pop()
                    loss = loss1 + reg_loss_norm
            else:
                loss = mf_loss + reg_loss
            
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            
            
            running_reg_loss += reg_loss.detach().item()

            if args.modeltype=="PopGO":
                running_loss1 += loss1.detach().item()
                running_loss2 += loss2.detach().item()
                running_loss +=  loss1.detach().item()+loss2.detach().item()+reg_loss.detach().item()
            else:
                running_loss += loss.detach().item()
                running_mf_loss += mf_loss.detach().item()

            num_batches += 1

        
        t2=time.time()

        # Training data for one epoch
        if args.modeltype=="PopGO":
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_loss1 / num_batches, running_loss2 / num_batches, running_reg_loss / num_batches)
        else:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_mf_loss / num_batches, running_reg_loss / num_batches)

        print(perf_str)

        with open(base_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
            f.write(perf_str+"\n")

        # Evaluate the trained model
        if (epoch + 1) % args.verbose == 0 and epoch >= freeze_epoch:
            model.eval() 
            for i,evaluator in enumerate(evaluators):
                is_best, flag = evaluation(args, data, model, epoch, base_path, evaluator,eval_names[i])
            
            if is_best:
                checkpoint_buffer=save_checkpoint(model, epoch, base_path, checkpoint_buffer, args.max2keep)
            model.train()
            

    # Get result
    model = restore_best_checkpoint(data.best_valid_epoch, model, base_path)
    print_str = "The best epoch is % d" % data.best_valid_epoch
    for i,evaluator in enumerate(evaluators[:2]):
        evaluation(args, data, model, epoch, base_path, evaluator, eval_names[i])
    with open('stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(print_str + "\n")

    # Get overlap
    #model.Calculate_pop()




