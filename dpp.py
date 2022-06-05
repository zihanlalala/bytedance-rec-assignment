from math import sqrt
import numpy as np
from numpy.random import default_rng
from copy import deepcopy
import time

def log_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        # logger.info(f"{func.__name__} using time {end - start}")
        return res, (end - start)
    return wrapper


class DPP:
    def __init__(self, item_num, emb_size, max_iter, theta=0.5) -> None:
        self.item_num = item_num
        self.emb_size = emb_size
        self.theta = theta
        self.max_iter = max_iter - 1
        pass
    
    def init_kernel(self, item_attrib):
        ## item_attrib should be of shape [item_num, 1]
        ## ru should be of shape [item_num]
        ss = np.dot(item_attrib, item_attrib.T) ## similarity matrix S is of shape [item_num, item_num]
        ss = (ss + 1) / 2 ## similarity value shoule be in [0, 1]

        # diag_ru = np.diag(ru)
        # self.L = np.dot(np.dot(diag_ru, self.S), diag_ru) ## kernel L is of shape [item_num, item_num]
        self.L = ss

    @log_time
    def inference(self, item_attrib, ru):
        self.init_kernel(item_attrib)

        ci = np.zeros([self.item_num, self.max_iter])
        di2 = deepcopy(np.diagonal(self.L))
        weight = self.theta * ru + (1 - self.theta)*di2
        j = np.argmax(weight)
        Yg = []
        Yg.append(j)
        probs = weight[j]
        # ### old version
        # for iter_num in range(self.max_iter):
        #     max_dj = 0
        #     max_dj_idx = 0
        #     for ii in range(self.item_num):
        #         if ii not in Yg:
        #             if iter_num == 0:
        #                 ei = self.L[j, ii] / sqrt(di2[j])
        #             else:
        #                 ei = (self.L[j, ii] - np.dot(ci[j, :iter_num], ci[ii, :iter_num])) / sqrt(di2[j])
        #             di2[ii] = di2[ii] - ei*ei
        #             ci[ii, iter_num] = ei
        #             if max_dj < di2[ii]:
        #                 ## theta controls trade-off between relevance and diversity
        #                 max_dj = self.theta * ru[ii] + (1 - self.theta) * di2[ii]
        #                 max_dj_idx = ii
            
        #     j = max_dj_idx
        #     Yg.append(j)
        #     if di2[j] < 0.01:
        #         break

        ## new version refering to ther auther's version
        for iter_num in range(self.max_iter):
            # print("cj shape ", ci[j, :iter_num].shape, ci[:, :iter_num].shape)
            ei = (self.L[j, :] - np.dot(ci[j, :iter_num], ci[:, :iter_num].T)) / sqrt(di2[j])
            di2 = di2 - np.square(ei)
            ci[:, iter_num] = ei
            
            ## theta controls trade-off between relevance and diversity
            weight = self.theta * ru + (1 - self.theta) * di2
            di2[j] = -np.inf

            j = np.argmax(weight)
            probs += weight[j]
            Yg.append(j)
            if di2[j] < 0.01:
                break
        return (Yg, probs)


    @log_time
    def beam_search_inference(self, item_attrib, ru, beam_size=1):
        self.init_kernel(item_attrib)

        max_iter = self.max_iter
        ci = np.zeros([max_iter, beam_size, self.item_num])
        di2 = deepcopy(np.diagonal(self.L))
        weight = self.theta * ru + (1 - self.theta)*di2
        sorted_indices = np.flip(np.argsort(weight,kind="stable"))
        js = sorted_indices[:beam_size] ## selected item j
        Yg = np.zeros([max_iter+1, beam_size])
        Yg[0, :] = js
        di2 = np.broadcast_to(di2, shape=[beam_size, di2.shape[0]])

        probs = np.zeros([beam_size])
        probs += weight[js]

        for iter_num in range(max_iter):
            beam_ci_opt = ci[:iter_num, :, :] ## size(iter_num, beam_size, item_num)
            beam_ci_opt = np.take_along_axis(beam_ci_opt, np.expand_dims(np.expand_dims(js, 0), 2), axis=2) ## shape [iter_num, beam_size, 1]
            beam_ci_opt = beam_ci_opt.transpose([1,2,0]) ## shape [beam_size, iter_num*1]
            ci_all = ci[:iter_num, :, :].transpose([1,0,2])
            tmp = np.matmul(beam_ci_opt, ci_all)

            tmp = (self.L[:, js] - tmp.reshape(self.item_num, beam_size))
            js_for_take = np.expand_dims(js, axis=1)
            di2js = np.take_along_axis(di2, js_for_take,axis=1)
            ei = (tmp / np.sqrt(np.squeeze(di2js))).T
            di2 = di2 - np.square(ei)
            ci[iter_num, :] = ei
            
            ## mask selected item 
            np.put_along_axis(di2, js_for_take, -np.inf, axis=1)
            ## theta controls trade-off between relevance and diversity
            weight = self.theta * np.expand_dims(ru, axis=0) + (1 - self.theta) * di2
            # print("weight shape ", weight.shape)
            # print("new di2 shape", di2)
            repeated_probs = np.repeat(probs, self.item_num)
            # print("repeated probs ",repeated_probs.shape)
            newprobs = repeated_probs + weight.reshape(-1)
            ## get beam idx
            js = np.flip(np.argsort(newprobs))
            js = js[:beam_size]
            beam_idx, item_idx = np.divmod(js, self.item_num)
            ## update beam
            probs = newprobs[js]
            di2 = di2[beam_idx, :]
            ci = ci[:, beam_idx, :]
            js = item_idx
            Yg[iter_num+1, :] = js

            
            # if di2[js] < 0.01:
            #     break
        return (Yg.transpose(1,0).astype(np.int), probs)




    def auther_ver(self, item_attrib, ru):
        self.init_kernel(item_attrib=item_attrib)
        item_size = self.L.shape[0]
        cis = np.zeros((self.max_iter, item_size))
        di2s = np.copy(np.diag(self.L))
        selected_items = list()
        selected_item = np.argmax(di2s)
        selected_items.append(selected_item)
        while len(selected_items) < self.max_iter+1:
            k = len(selected_items) - 1
            ci_optimal = cis[:k, selected_item]
            di_optimal = sqrt(di2s[selected_item])
            elements = self.L[selected_item, :]
            # print("cj shape ", ci_optimal.shape, cis[:k, :].shape)

            eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
            cis[k, :] = eis
            di2s -= np.square(eis)
            di2s[selected_item] = -np.inf
            selected_item = np.argmax(di2s)
            if di2s[selected_item] < 0.01:
                break
            selected_items.append(selected_item)
        return selected_items



def get_features(item_num, emb_size):
    ## after norm, the diag are all 1., so the first item is always random or fixed? 
    rng = default_rng(1234)
    item_attrib = rng.random([item_num, emb_size])
    attrib_norm = np.linalg.norm(item_attrib, ord=2, axis=1, keepdims=True)
    normed_item_attrib = np.divide(item_attrib, attrib_norm)
    ru = rng.random([item_num])
    ru = np.exp(ru * 0.01 + 0.2)
    ru.sort()
    ru = np.flip(ru)
    return normed_item_attrib, ru


def cal_MRR(Yg):
    has_taget = np.equal(Yg, 0)
    if np.any(has_taget):
        idx = np.argmax(has_taget)
        return 1.0 / (idx+1)
    else:
        return 0

def cal_ILAD(Yg, S):
    sim = 1 - S[Yg][:, Yg]
    res = np.sum(sim) / len(Yg)*(len(Yg)-1)
    return res

def main():
    item_num = 100
    emb_size = 64
    max_iter = 8

    for theta in range(0, 1):
        theta = 5
        print("theta is ", theta*0.1)
        dpp_model = DPP(item_num=item_num, emb_size=emb_size, max_iter=max_iter, theta=theta*0.1)

        infer_time = 0
        infer_mrr = []
        infer_ilad = []

        beam_infer_time = [0 for _ in range(9)]
        beam_infer_mrr = [[] for _ in range(9)]
        beam_infer_ilad = [[] for _ in range(9)]
        prob_ratio = [[] for _ in range(9)]
        for i in range(1):
            item_attrib, ru = get_features(item_num=item_num, emb_size=emb_size)

            (Yg, probs), tt = dpp_model.inference(item_attrib=item_attrib, ru=ru)
            infer_time += tt
            # print(list(Yg))

            mrr = cal_MRR(Yg)
            ilad = cal_ILAD(Yg, dpp_model.L)

            infer_mrr.append(mrr)
            infer_ilad.append(ilad)
            for beam in range(1, 9):
                # print("beam is ", beam)

                (Yg, beam_probs), tt = dpp_model.beam_search_inference(item_attrib=item_attrib, ru=ru, beam_size=beam)
                print(list(Yg))
                beam_infer_time[beam-1] += tt

                mrr = cal_MRR(Yg[0])
                ilad = cal_ILAD(Yg[0], dpp_model.L)
                beam_infer_mrr[beam-1].append(mrr)
                beam_infer_ilad[beam-1].append(ilad)
                prob_ratio[beam-1].append( beam_probs[0] / probs)
            # auther_res = dpp_model.auther_ver(item_attrib=item_attrib, ru=ru)
            # print("auther res ", auther_res)

        for ii in range(8):
            print(f"infer mrr {np.mean(infer_mrr)}, ilad {np.mean(infer_ilad)}, using time {infer_time}")
            print(f"beam {ii+1} infer mrr {np.mean(beam_infer_mrr[ii])}, ilad {np.mean(beam_infer_ilad[ii])}, using time {beam_infer_time[ii]}")
            print(f"probs ratio {np.mean(prob_ratio[ii])}")

    pass


if __name__ == "__main__":

    main()
