import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from metrics import mrr_mr_hitk
from utils import batch_by_size
import logging
import os
import time
from torch.optim import Adam, SGD, Adagrad
from models import TransDModule, TransEModule, TransHModule, DistMultModule, ComplExModule, SimplEModule, RotatEModule

class BaseModel(object):
    def __init__(self, n_ent, n_rel, args):
        if args.model == 'TransE':
            self.model = TransEModule(n_ent, n_rel, args)
        elif args.model == 'TransD':
            self.model = TransDModule(n_ent, n_rel, args)
        elif args.model == 'TransH':
            self.model = TransHModule(n_ent, n_rel, args)
        elif args.model == 'DistMult':
            self.model = DistMultModule(n_ent, n_rel, args)
        elif args.model == 'ComplEx':
            self.model = ComplExModule(n_ent, n_rel, args)
        elif args.model == 'SimplE':
            self.model = SimplEModule(n_ent, n_rel, args)
        elif args.model == 'RotatE':
            self.model = RotatEModule(n_ent, n_rel, args)
        else:
            raise NotImplementedError

        self.model.cuda()

        self.n_ent = n_ent
        self.weight_decay = args.lamb * args.n_batch / args.n_train
        self.time_tot = 0
        self.args = args
        self.cache_score = np.random.randn(args.n_train,)   # used for sampling positive triplets


    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage.cuda()))


    def remove_positive(self, remove=True):
        ''' this function removes false negative triplets in cache. '''
        length_h = len(self.head_pos)
        length_t = len(self.tail_pos)
        length = length_h + length_t
        self.count_pos = 0

        # use random variable to replace the false negative in cache
        def head_remove(arr):
            idx = arr[0]
            mark = np.isin(self.head_cache[idx], self.head_pos[idx])
            if remove == True:
                rand = np.random.choice(self.n_ent, size=(self.args.N_1,), replace=False)
                self.head_cache[idx][mark] = rand[mark]
            self.count_pos += np.sum(mark)

        def tail_remove(arr):
            idx = arr[0]
            mark = np.isin(self.tail_cache[idx], self.tail_pos[idx])
            if remove == True:
                rand = np.random.choice(self.n_ent, size=(self.args.N_1,), replace=False)
                self.tail_cache[idx][mark] = rand[mark]
            self.count_pos += np.sum(mark)

        head_idx = np.expand_dims(np.array(range(length_h), dtype='int'), 1)
        tail_idx = np.expand_dims(np.array(range(length_t), dtype='int'), 1)

        np.apply_along_axis(head_remove, 1, head_idx)
        np.apply_along_axis(tail_remove, 1, tail_idx)

        print("number of positives:", self.count_pos, self.count_pos/length)
        return self.count_pos / length
    

    def update_cache(self, head, tail, rela, idx, head_idx, tail_idx):
        ''' update the cache with different schemes '''
        batch_size = len(head_idx)

        # get candidate for updating the cache
        h_cache = self.head_cache[head_idx]
        t_cache = self.tail_cache[tail_idx]
        rand1 = np.random.choice(self.n_ent, (batch_size, self.args.N_2))
        rand2 = np.random.choice(self.n_ent, (batch_size, self.args.N_2))
        h_cand = np.concatenate([h_cache, rand1], 1)
        t_cand = np.concatenate([t_cache, rand2], 1)
        h_cand = torch.from_numpy(h_cand).type(torch.LongTensor).cuda()
        t_cand = torch.from_numpy(t_cand).type(torch.LongTensor).cuda()

        # expand for computing scores/probs
        head = head.unsqueeze(1).expand(-1, self.args.N_1 + self.args.N_2)
        tail = tail.unsqueeze(1).expand(-1, self.args.N_1 + self.args.N_2)
        rela = rela.unsqueeze(1).expand(-1, self.args.N_1 + self.args.N_2)

        # for negative triplets, larger loss approximates larger gradient
        h_logits = self.model.point_loss(head, tail, rela, 1) + self.model.point_loss(h_cand, tail, rela, -1)
        t_logits = self.model.point_loss(head, tail, rela, 1) + self.model.point_loss(head, t_cand, rela, -1)

        # normarlize the logits
        h_logits_np = h_logits.data.cpu().numpy()
        quant_lo = np.repeat(np.quantile(h_logits_np, 0.1, axis=1, keepdims=True), h_logits_np.shape[1], axis=1)
        quant_hi = np.repeat(np.quantile(h_logits_np, 0.9, axis=1, keepdims=True), h_logits_np.shape[1], axis=1)
        ind_lo = h_logits_np<quant_lo
        ind_hi = h_logits_np>quant_hi
        h_logits_np[ind_lo] = quant_lo[ind_lo]
        h_logits_np[ind_hi] = quant_hi[ind_hi]
        quant = np.maximum(1e-4, quant_hi - quant_lo)
        h_logits_np = (h_logits_np-quant_lo)/quant
        h_logits_np = torch.FloatTensor(h_logits_np).cuda()

        t_logits_np = t_logits.data.cpu().numpy()
        quant_lo = np.repeat(np.quantile(t_logits_np, 0.1, axis=1, keepdims=True), t_logits_np.shape[1], axis=1)
        quant_hi = np.repeat(np.quantile(t_logits_np, 0.9, axis=1, keepdims=True), t_logits_np.shape[1], axis=1)
        ind_lo = t_logits_np<quant_lo
        ind_hi = t_logits_np>quant_hi
        t_logits_np[ind_lo] = quant_lo[ind_lo]
        t_logits_np[ind_hi] = quant_hi[ind_hi]
        quant = np.maximum(1e-4, quant_hi - quant_lo)
        t_logits_np = (t_logits_np-quant_lo)/quant
        t_logits_np = torch.FloatTensor(t_logits_np).cuda()

        # sampling and update the cache
        h_probs = F.softmax(h_logits_np * self.args.alpha_3, dim=-1)
        t_probs = F.softmax(t_logits_np * self.args.alpha_3, dim=-1)

        h_new = torch.multinomial(h_probs, self.args.N_1, replacement=False)
        t_new = torch.multinomial(t_probs, self.args.N_1, replacement=False)

        row_idx = torch.arange(0, batch_size).type(torch.LongTensor).unsqueeze(1).expand(-1, self.args.N_1)
        h_rep = h_cand[row_idx, h_new]
        t_rep = t_cand[row_idx, t_new]

        self.head_cache[head_idx] = h_rep.cpu().numpy()
        self.tail_cache[tail_idx] = t_rep.cpu().numpy()
        self.cache_score[idx] = torch.sum(h_logits[row_idx, h_new] + t_logits[row_idx, t_new], 1).data.cpu().numpy()
        self.head_score[head_idx] = h_logits[row_idx, h_new].data.cpu().numpy()
        self.tail_score[tail_idx] = t_logits[row_idx, t_new].data.cpu().numpy()


    def neg_sample(self, head, tail, rela, head_idx, tail_idx, sample='basic', loss='pair'):
        if sample == 'bern':    # Bernoulli sampling
            n = head_idx.shape[0]
            h_idx = np.random.randint(low=0, high=self.n_ent, size=(n, self.args.n_sample))
            t_idx = np.random.randint(low=0, high=self.n_ent, size=(n, self.args.n_sample))
            h_rand = torch.LongTensor(h_idx).cuda()
            t_rand = torch.LongTensor(t_idx).cuda()
        else:
            '''
            negative sampling scheme based on alpha
            '''
            h_logits = self.head_score[head_idx]
            quant_lo = np.repeat(np.quantile(h_logits, 0.1, axis=1, keepdims=True), h_logits.shape[1], axis=1)
            quant_hi = np.repeat(np.quantile(h_logits, 0.9, axis=1, keepdims=True), h_logits.shape[1], axis=1)
            ind_lo = h_logits<quant_lo
            ind_hi = h_logits>quant_hi
            h_logits[ind_lo] = quant_lo[ind_lo]
            h_logits[ind_hi] = quant_hi[ind_hi]
            quant = np.maximum(1e-4, quant_hi - quant_lo)
            h_logits = (h_logits-quant_lo)/quant

            t_logits = self.tail_score[tail_idx]
            quant_lo = np.repeat(np.quantile(t_logits, 0.1, axis=1, keepdims=True), t_logits.shape[1], axis=1)
            quant_hi = np.repeat(np.quantile(t_logits, 0.9, axis=1, keepdims=True), t_logits.shape[1], axis=1)
            ind_lo = t_logits<quant_lo
            ind_hi = t_logits>quant_hi
            t_logits[ind_lo] = quant_lo[ind_lo]
            t_logits[ind_hi] = quant_hi[ind_hi]
            quant = np.maximum(1e-4, quant_hi - quant_lo)
            t_logits = (t_logits-quant_lo)/quant

            #h_logits = self.head_score[head_idx]
            #t_logits = self.tail_score[tail_idx]
            head_probs = F.softmax(torch.FloatTensor(h_logits).cuda() * self.args.alpha_2, dim=-1)
            tail_probs = F.softmax(torch.FloatTensor(t_logits).cuda() * self.args.alpha_2, dim=-1)
            head_new = torch.multinomial(head_probs, 1).squeeze().cpu().numpy()
            tail_new = torch.multinomial(tail_probs, 1).squeeze().cpu().numpy()
            h_rand = torch.LongTensor(self.head_cache[head_idx, head_new]).cuda()
            t_rand = torch.LongTensor(self.tail_cache[tail_idx, tail_new]).cuda()
        return h_rand, t_rand


    def train(self, train_data, caches, corrupter, tester_val, tester_tst):
        heads, tails, relas = train_data
        # useful information related to cache
        head_idxs, tail_idxs, self.head_cache, self.tail_cache, self.head_pos, self.tail_pos = caches
        self.head_score = np.random.randn(len(self.head_cache), self.args.N_1)
        self.tail_score = np.random.randn(len(self.tail_cache), self.args.N_1)
        n_train = len(heads)

        if self.args.optim=='adam' or self.args.optim=='Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        elif self.args.optim=='adagrad' or self.args.optim=='Adagrad':
            self.optimizer = Adagrad(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)

        n_epoch = self.args.n_epoch
        n_batch = self.args.n_batch
        best_mrr = 0

        losses = []
        for epoch in range(n_epoch):
            start = time.time()
            self.epoch = epoch

            # positive sampling
            logits = self.cache_score
            quant_lo = np.quantile(logits, 0.2)
            quant_hi  = np.quantile(logits, 0.8)
            logits[logits<quant_lo] = quant_lo
            logits[logits>quant_hi] = quant_hi
            logits = (logits-quant_lo)/(quant_hi - quant_lo)
            logits = logits * self.args.alpha_1
            probb = np.exp(logits) / np.exp(logits).sum()
            if epoch == 0:      # use uniform sampling for the first epoch
                probb = np.ones((n_train,)) / n_train
            
            indices = np.random.choice(n_train, n_train, replace=False, p=probb)
            rand_idx = torch.LongTensor(indices)
            head = heads[rand_idx].cuda()
            tail = tails[rand_idx].cuda()
            rela = relas[rand_idx].cuda()
            head_idx = head_idxs[indices]
            tail_idx = tail_idxs[indices]

            epoch_loss = 0

            if self.args.save and epoch==self.args.s_epoch:
                self.save(os.path.join(self.args.task_dir, self.args.model + '.mdl'))

            iters = 0
            for h, t, r, h_idx, t_idx, idx, in batch_by_size(n_batch, head, tail, rela, head_idx, tail_idx, indices, n_sample=n_train):
                self.model.zero_grad()

                h_rand, t_rand = self.neg_sample(h, t, r, h_idx, t_idx, self.args.sample, self.args.loss)
              
                # Bernoulli sampling to select (h', r, t) and (h, r, t')
                prob = corrupter.bern_prob[r]
                selection = torch.bernoulli(prob).type(torch.ByteTensor).cuda()
                n_h = torch.LongTensor(h.cpu().numpy()).cuda()
                n_t = torch.LongTensor(t.cpu().numpy()).cuda()
                n_r = torch.LongTensor(r.cpu().numpy()).cuda()
                if n_h.size() != h_rand.size():
                    n_h = n_h.unsqueeze(1).expand_as(h_rand)
                    n_t = n_t.unsqueeze(1).expand_as(h_rand)
                    n_r = n_r.unsqueeze(1).expand_as(h_rand)
                    h = h.unsqueeze(1)
                    r = r.unsqueeze(1)
                    t = t.unsqueeze(1)
                    
                n_h[selection] = h_rand[selection]
                n_t[~selection] = t_rand[~selection]
                
                if not (self.args.sample=='bern') and iters % self.args.lazy==0:
                    self.update_cache(h, t, r, idx, h_idx, t_idx)

                if self.args.loss == 'point':
                    p_loss = torch.sum(self.model.point_loss(h, t, r, 1))
                    n_loss = torch.sum(self.model.point_loss(n_h, n_t, n_r, -1))
                    loss = p_loss + n_loss
                else:
                    loss = self.model.pair_loss(h, t, r, n_h, n_t)
                
                loss.backward()
                self.optimizer.step()
                self.remove_nan()
                epoch_loss += loss.data.cpu().numpy()
                iters += 1
            # get the time of each epoch
            self.time_tot += time.time() - start
            losses.append(round(epoch_loss/n_train, 4))
           
               
            if (epoch+1) % self.args.epoch_per_test == 0:
                # output performance 
                valid_mrr, valid_mr, valid_1, valid_3, valid_10 = tester_val()
                test_mrr,  test_mr,  test_1,  test_3,  test_10 =  tester_tst()
                out_str = '%d\t%.2f\t%.4f %.1f %.4f %.4f %.4f\t%.4f %.1f %.4f %.4f %.4f\n' % (epoch, self.time_tot, \
                        valid_mrr, valid_mr, valid_1, valid_3, valid_10, \
                        test_mrr, test_mr, test_1, test_3, test_10)
                with open(self.args.perf_file, 'a') as f:
                    f.write(out_str)

                # remove false negative 
                if self.args.remove:
                    self.remove_positive(self.args.remove)

                # output the best performance info
                if valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    best_str = out_str
        return best_mrr, best_str

    def remove_nan(self,):
        # avoid nan parameters
        for p in self.model.parameters():
            X = p.data.clone()
            flag = X != X
            X[flag] = 0
            p.data.copy_(X)


    def test_link(self, test_data, n_ent, heads, tails, filt=True):
        mrr_tot = 0.
        mr_tot = 0
        #hit10_tot = 0
        hit_tot = np.zeros((3,))
        count = 0
        for batch_h, batch_t, batch_r in batch_by_size(self.args.test_batch_size, *test_data):
            batch_size = batch_h.size(0)
            head_val = Variable(batch_h.unsqueeze(1).expand(batch_size, n_ent).cuda())
            tail_val = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent).cuda())
            rela_val = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent).cuda())
            all_val = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent).type(torch.LongTensor).cuda())
            batch_head_scores = self.model.score(all_val, tail_val, rela_val).data
            batch_tail_scores = self.model.score(head_val, all_val, rela_val).data
            # for each positive, compute its head scores and tail scores
            for h, t, r, head_score, tail_score in zip(batch_h, batch_t, batch_r, batch_head_scores, batch_tail_scores):
                h_idx = int(h.data.cpu().numpy())
                t_idx = int(t.data.cpu().numpy())
                r_idx = int(r.data.cpu().numpy())
                if filt:            # filtered setting
                    if tails[(h_idx,r_idx)]._nnz() > 1:
                        tmp = tail_score[t_idx].data.cpu().numpy()
                        idx = tails[(h_idx, r_idx)]._indices()
                        tail_score[idx] = 1e20
                        tail_score[t_idx] = torch.from_numpy(tmp).cuda()
                    if heads[(t_idx, r_idx)]._nnz() > 1:
                        tmp = head_score[h_idx].data.cpu().numpy()
                        idx = heads[(t_idx, r_idx)]._indices()
                        head_score[idx] = 1e20
                        head_score[h_idx] = torch.from_numpy(tmp).cuda()
                mrr, mr, hit = mrr_mr_hitk(tail_score, t_idx)
                mrr_tot += mrr
                mr_tot += mr
                hit_tot += hit
                mrr, mr, hit = mrr_mr_hitk(head_score, h_idx)
                mrr_tot += mrr
                mr_tot += mr
                hit_tot += hit
                count += 2
        logging.info('Test_MRR=%f, Test_MR=%f, Test_H=%f %f %f, Count=%d', float(mrr_tot)/count, float(mr_tot)/count, hit_tot[0]/count, hit_tot[1]/count, hit_tot[2]/count, count)
        return float(mrr_tot)/count, mr_tot/count, hit_tot[0]/count, hit_tot[1]/count, hit_tot[2]/count





