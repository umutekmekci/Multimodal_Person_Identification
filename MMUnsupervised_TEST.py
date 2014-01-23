# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:40:24 2014

@author: daredavil
"""

from __future__ import division
import copy
import numpy as np
from parseTextToJson import TargetRel, ParseTextToJson
from MMSupervisedCluster import get_mapping_dict,extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import cPickle
from sklearn.preprocessing import StandardScaler

def extract_unique_names_probs(person_dict):
    pers_names = person_dict.keys()
    dur_times = np.array(person_dict.values())
    names = []
    probs = []
    for name in pers_names:
        probability = float(name[name.index('<')+1:-1])
        real_name = name[:name.index('<')]
        names.append(real_name)
        probs.append(probability)
    probs = np.array(probs)
    names = np.array(names)
    unique_names = []
    unique_probs = []
    unique_times = []
    for name in np.unique(names):
        unique_names.append(name)
        unique_probs.append(probs[names == name].mean())
        unique_times.append(dur_times[names==name].sum())
    return unique_names, unique_probs, unique_times

def parse_time_interval(time_int):
    """
    for each time in time_interval
    if time == 234.0 then translate it to 234
    if time == 234.450 then translate it to 234.45
    """
    temp = []
    for t in time_int:
        if t[0] == '-':
            print "aha buldum ", t
        tl = t.split('.')
        if len(tl) == 1:
            temp.append(tl[0])
            continue
        if len(tl[1])>2:
            tl[1] = tl[1][:2]
        if len(tl[1]) == 1 and tl[1] == '0':
            gen = tl[0]
        else:
            gen = '.'.join(tl)
        temp.append(gen)
    return temp

class UnknownName(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class MMUnsCluster:
    def __init__(self, list_of_shows):
        #self.cls_parse = ParseTextToJson(file_names_dict = {}, list_file_name = list_of_shows)
        #self.programs_list = self.cls_parse._get_programs_list()
        self.list_of_shows = list_of_shows        
        self.cls_names = TargetRel(None)
        self.name_map = self.cls_names.name_map_all()
        self.id_map = self.cls_names.target_id_map(self.name_map)
        
    def initialize_dict(self):
        speaker_cluster = {}
        with open(self.list_of_shows, 'r') as f_:
            for line in f_:
                pr_name1, pr_name2, program_date, program_idx = line.strip().split('_')
                program_name = pr_name1+'_'+pr_name2
                if program_name in speaker_cluster:
                    if program_idx in speaker_cluster[program_name]:
                        speaker_cluster[program_name][program_idx][program_date] = {'interval':[], 'cluster_names':[]}
                    else:
                        speaker_cluster[program_name][program_idx] = {program_date:  {'interval':[], 'cluster_names':[]}}
                else:
                    speaker_cluster[program_name] = {program_idx: {program_date: {'interval':[], 'cluster_names':[]}}}
        return speaker_cluster
    
    def parse_diarization_file(self, file_name):
        """
        prog_name:
            prog_id:
                prog_date:
                    'interval':[ [],[],... ]
                    'cluster_names':[ ... ]
                    'overlap_list': [{name:time, name:time,...},{name:time},...] sonradan ekleniyor
                    'assigned_names': [name, name,...]  sonradan ekleniyor
                    'sealed': [True, False,...]  sonradan ekleniyor
        """
        assigned_dict = {}
        N = 0
        speaker_cluster = self.initialize_dict()
        with open(file_name,'r') as f_:
            prev_beg_time = -1
            for line in f_:
                N+=1
                line_split = line.strip().split()
                program_detail = line_split[0]
                person_name = line_split[-1]
                time_interval = map(float, parse_time_interval(line_split[2:4]))
                pr_name1, pr_name2, program_date, program_idx = program_detail.split('_')
                program_name = pr_name1+'_'+pr_name2
                if not program_name in speaker_cluster or not program_idx in speaker_cluster[program_name] or not program_date in speaker_cluster[program_name][program_idx]:
                    continue
                speaker_cluster[program_name][program_idx][program_date]['interval'].append([time_interval[0],sum(time_interval)])
                speaker_cluster[program_name][program_idx][program_date]['cluster_names'].append(person_name)
                if speaker_cluster[program_name][program_idx][program_date]['interval'][-1][0] == prev_beg_time:
                    print 'buldum speaker'
                prev_beg_time = time_interval[0]
                if not person_name in assigned_dict:
                    assigned_dict[person_name] = []
        return speaker_cluster,N,assigned_dict
        
    def parse_etf_file(self, file_name):
        speaker_cluster = self.initialize_dict()
        with open(file_name,'r') as f_:
            prev_beg_time = -1
            for line in f_:
                line_split = line.strip().split()
                if line_split[-1] != "top_score":
                    continue
                program_detail = line_split[0]
                probability = line_split[-2]
                person_name = line_split[-3] + '<' + probability + '>'
                time_interval = map(float, parse_time_interval(line_split[2:4]))
                pr_name1, pr_name2, program_date, program_idx = program_detail.split('_')
                program_name = pr_name1+'_'+pr_name2
                if not program_name in speaker_cluster or not program_idx in speaker_cluster[program_name] or not program_date in speaker_cluster[program_name][program_idx]:
                    continue
                speaker_cluster[program_name][program_idx][program_date]['interval'].append([time_interval[0],sum(time_interval)])
                speaker_cluster[program_name][program_idx][program_date]['cluster_names'].append(person_name)
                if speaker_cluster[program_name][program_idx][program_date]['interval'][-1][0] == prev_beg_time:
                    print 'buldum speaker'
                prev_beg_time = time_interval[0]
        return speaker_cluster
    
    def parse_written_name_file(self, file_name):
        written_names = self.initialize_dict()
        with open(file_name,'r') as f_:
            prev_beg_time = -1
            for line in f_:
                line_split = line.strip().split()
                program_detail = line_split[0]
                person_name = line_split[-1]
                time_interval = map(float, parse_time_interval(line_split[1:3]))
                pr_name1, pr_name2, program_date, program_idx = program_detail.split('_')
                program_name = pr_name1+'_'+pr_name2
                if not program_name in written_names or not program_idx in written_names[program_name] or not program_date in written_names[program_name][program_idx]:
                    continue
                written_names[program_name][program_idx][program_date]['interval'].append(time_interval)
                written_names[program_name][program_idx][program_date]['cluster_names'].append(person_name)
                if written_names[program_name][program_idx][program_date]['interval'][-1][0] == prev_beg_time:
                    print 'buldum written'
                prev_beg_time = time_interval[0]
        return written_names
        
    def _get_name_cand_list(self, beg_time,end_time,wr_intervals, wr_names):
        name_list = []
        for i, (bb, ee) in enumerate(wr_intervals):
            if bb < beg_time and ee < beg_time:
                continue
            if bb>end_time and ee > end_time:
                continue
                #break
            name_list.append(wr_names[i])
        return name_list
        
    def test_for_labels(self, speaker_cluster, cluster_general_intervals, name_cluster,spoken_cluster,cluster_frequencies):
        name_cand_dict = {}
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    cluster_names = np.array(cluster_general_intervals[program_name][program_idx][program_date].keys())
                    cl_intervals = np.array(cluster_general_intervals[program_name][program_idx][program_date].values())
                    inds = cl_intervals[:,0].argsort()
                    cluster_names = cluster_names[inds]
                    cl_intervals = cl_intervals[inds]
                    time_intervals = speaker_cluster[program_name][program_idx][program_date]['interval']
                    wr_intervals = name_cluster[program_name][program_idx][program_date]['interval']
                    wr_names = name_cluster[program_name][program_idx][program_date]['cluster_names']
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if not cluster_name in name_cand_dict or not name_cand_dict[cluster_name]:
                            end_time = time_intervals[i][-1]
                            ind = np.nonzero(cluster_names == cluster_name)[0][0]
                            end_time2 = cl_intervals[ind,1]
                            beg_time = cl_intervals[0,0] if ind in [0,1] else cl_intervals[ind-2,0]
                            name_cand_dict[cluster_name] = self._get_name_cand_list(beg_time,end_time2,wr_intervals, wr_names)
                            #ii = ind if ind in [0,] else ind-1
                            #name_cand_dict[cluster_name].extend(name_cand_dict[cluster_names[ii]])
        print "there are ",np.mean([len(set(x)) for x in name_cand_dict.values()]), " average name candidates"
        buldum = 0
        toplam = 0
        wr,sp = 0,0
        total2,tot_true,tot_false = 0,0,0
        tot_true2,tot_false2 = 0,0
        sup_equal,sup_true,kotu,iyi,sim,sim2 = 0,0,0,0,0,0
        true_tot3,false_tot3, total3,true_tot33,false_tot33 = 0,0,0,0,0
        true_names,false_names,true_f_l,false_f_l = [],[],[],[]
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    labels = speaker_cluster[program_name][program_idx][program_date]['labels']
                    wr_names = name_cluster[program_name][program_idx][program_date]['cluster_names']
                    GMM_labels = speaker_cluster[program_name][program_idx][program_date]['GMM']
                    SVM_labels = speaker_cluster[program_name][program_idx][program_date]['SVM']
                    ivector_labels = speaker_cluster[program_name][program_idx][program_date]['ivector']
                    spoken_labels = spoken_cluster[program_name][program_idx][program_date]['cluster_names']
                    cll_names = speaker_cluster[program_name][program_idx][program_date]['cluster_names']
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        #if cluster_name == "MS12_LCP_CaVousRegarde_2011-12-20_204600":
                        #    pass
                        if labels[i]:
                            toplam += 1
                            name_cands = name_cand_dict[cluster_name]
                            label_name = labels[i].keys()[np.argmax(labels[i].values())]
                            gmm_label = GMM_labels[i].keys()[np.argmax(GMM_labels[i].values())]
                            svm_label = SVM_labels[i].keys()[np.argmax(SVM_labels[i].values())]
                            if ivector_labels[i]:
                                ivector_label = ivector_labels[i].keys()[np.argmax(ivector_labels[i].values())]
                            else:
                                ivector_label = svm_label
                            gmm_prob,svm_prob = float(gmm_label[gmm_label.find('<')+1:-1]),float(svm_label[svm_label.find('<')+1:-1])
                            ivector_prob = float(ivector_label[ivector_label.find('<')+1:-1])
                            gmm_label,svm_label = gmm_label[:gmm_label.find('<')],svm_label[:svm_label.find('<')]
                            ivector_label = ivector_label[:ivector_label.find('<')]
                            frequency = cluster_frequencies[program_name][program_idx][program_date][cluster_name]
                            if label_name in wr_names:
                                wr += 1
                            if label_name in spoken_labels:
                                sp+= 1
                            if label_name in name_cands:
                                buldum += 1
                            if svm_label != gmm_label:
                                total3 += 1
                                if ivector_label == label_name:
                                    true_tot3 +=1
                                    if ivector_label in name_cands:
                                        true_tot33 += 1
                                else:
                                    if ivector_label in name_cands:
                                        false_tot33 += 1
                                    false_tot3 += 1
                            
                            if gmm_label == svm_label and not gmm_label in name_cands:
                                total2 += 1
                                if gmm_label == label_name:
                                    tot_true+=1
                                    sim_list = []
                                    for k,cl_nn in enumerate(cll_names[:i]):
                                        gmm_ = GMM_labels[k].keys()[np.argmax(GMM_labels[k].values())]
                                        svm_ = SVM_labels[k].keys()[np.argmax(SVM_labels[k].values())]
                                        gmm_,svm_ = gmm_[:gmm_.find('<')],svm_[:svm_.find('<')]
                                        if gmm_ == gmm_label or svm_ == svm_label:
                                            sim_list.append(cl_nn)
                                    if sim_list:
                                        sim+= 1
                                    true_f_l.append(frequency)
                                    if gmm_label in spoken_labels:
                                        tot_true2 += 1
                                    true_names.append(gmm_label)
                                else:
                                    tot_false+=1
                                    sim_list = []
                                    for k,cl_nn in enumerate(cll_names[:i]):
                                        gmm_ = GMM_labels[k].keys()[np.argmax(GMM_labels[k].values())]
                                        svm_ = SVM_labels[k].keys()[np.argmax(SVM_labels[k].values())]
                                        gmm_,svm_ = gmm_[:gmm_.find('<')],svm_[:svm_.find('<')]
                                        if gmm_ == gmm_label or svm_ == svm_label:
                                            sim_list.append(cl_nn)
                                    if sim_list:
                                        sim2+= 1
                                    false_f_l.append(frequency)
                                    if gmm_label in spoken_labels:
                                        tot_false2 += 1
                                    false_names.append((gmm_label,label_name))
                            if gmm_label == svm_label and gmm_prob>0 and svm_prob>0:
                                sup_equal += 1
                                if label_name == gmm_label:
                                    sup_true+=1
                                    if gmm_label in name_cands:
                                        iyi += 1
                                else:
                                    if gmm_label in name_cands:
                                        kotu += 1
        print "bulunan ",buldum, "  in ",toplam
        print "writtten_names: ",wr, " spoken_names: ",sp
        print "dogru: ",sup_true, " in ",sup_equal, " in ",toplam
        print "kotu: ", kotu, " iyi: ",iyi
        print "dogru: ",tot_true, " yanlis: ",tot_false, " in ", total2
        print "sim: ", sim, " sim2: ",sim2
        print "tot_true2: ",tot_true2, " tot_false2: ", tot_false2
        print
        print "total: ",total3, " dogru: ",true_tot3, " yanlis: ",false_tot3
        print "true_tot33: ",true_tot33, " false_tot33 ",false_tot33
       # print "true frequencies"
       # print true_f_l
       # print
       # print "false_frequencies"
       # print false_f_l
        #print "true_names:"
        #print true_names
        #print
        #print "false_names:"
        #for x in false_names:
        #    print x
        return name_cand_dict
        
                    
        
    def _interval_of_clusters(self, cl_names, interval_sp,i):
        cl_name = cl_names[i]
        beg_time,end_time = interval_sp[i]
        for name, k_interval in zip(cl_names[i+1:],interval_sp[i+1:]):
            if name == cl_name:
                end_time = k_interval[-1]
        return (beg_time,end_time)
        
    def interval_of_clusters(self, speaker_cluster):
        cluster_general_intervals = {}
        cluster_frequencies = {}
        for program_name in speaker_cluster:
            if not program_name in cluster_general_intervals:
                cluster_general_intervals[program_name] = {}
                cluster_frequencies[program_name] = {}
            for program_idx in speaker_cluster[program_name]:
                if not program_idx in cluster_general_intervals[program_name]:
                    cluster_general_intervals[program_name][program_idx] = {}
                    cluster_frequencies[program_name][program_idx] = {}
                for program_date in speaker_cluster[program_name][program_idx]:
                    if not program_date in cluster_general_intervals[program_name][program_idx]:
                        cluster_general_intervals[program_name][program_idx][program_date] = {}
                        cluster_frequencies[program_name][program_idx][program_date] = {}
                    interval_sp = speaker_cluster[program_name][program_idx][program_date]['interval']
                    cl_names = speaker_cluster[program_name][program_idx][program_date]['cluster_names']
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if not cluster_name in cluster_general_intervals[program_name][program_idx][program_date]:
                            cluster_general_intervals[program_name][program_idx][program_date][cluster_name] = self._interval_of_clusters(cl_names, interval_sp,i)
                            cluster_frequencies[program_name][program_idx][program_date][cluster_name]  = 1
                        else:
                            cluster_frequencies[program_name][program_idx][program_date][cluster_name]+=1
        return cluster_general_intervals,cluster_frequencies
        
    def _check_for_coc(self, cl_dict, beg_time, end_time, assign_name):
        if not cl_dict:
            return False
        merge_clusters = []
        for cl_name in cl_dict:
            cl_beg, cl_end = cl_dict[cl_name]['interval']
            if beg_time<cl_beg and end_time>cl_beg and end_time<=cl_end:
                merge_clusters.append((cl_name,beg_time,cl_end))
            elif beg_time<cl_beg and end_time>cl_beg and end_time>=cl_end:
                merge_clusters.append((cl_name,beg_time,end_time))
            elif beg_time>=cl_beg and beg_time<cl_end and end_time<=cl_end:
                merge_clusters.append((cl_name,cl_beg,cl_end))
            elif beg_time>=cl_beg and beg_time<cl_end and end_time>cl_end:
                merge_clusters.append((cl_name,cl_beg,end_time))
        if not merge_clusters:
            return False
        if len(merge_clusters) == 1:
            cl_dict[merge_clusters[0][0]]['interval'] = merge_clusters[0][1:]
            cl_dict[merge_clusters[0][0]]['clusters'].append(assign_name)
            return True
        else:
            new_cl = {'interval':[], 'clusters':[]}
            new_beg,new_end = merge_clusters[0][1:]
            new_name = merge_clusters[0][0]
            new_cl['clusters'].extend(cl_dict[new_name]['clusters'])      
            cl_dict.pop(new_name)
            for item_ in merge_clusters[1:]:
                new_cl['clusters'].extend(cl_dict[item_[0]]['clusters'])
                name, beg_time,end_time = item_
                if beg_time < new_beg:
                    new_beg = beg_time
                if end_time > new_end:
                    new_end = end_time
                if name < new_name:
                    new_name = name
                cl_dict.pop(name)
            new_cl['interval'] = [new_beg, new_end]
            cl_dict[new_name] = new_cl
            return True
            
        
    def cluster_of_clusters_general(self, speaker_cluster, cluster_general_intervals):
        coc_dict = {}
        current_cluster = -1
        for program_name in cluster_general_intervals:
            if not program_name in coc_dict:
                coc_dict[program_name] = {}
            for program_idx in cluster_general_intervals[program_name]:
                if not program_idx in coc_dict[program_name]:
                    coc_dict[program_name][program_idx] = {}
                for program_date in cluster_general_intervals[program_name][program_idx]:
                    if not program_date in coc_dict[program_name][program_idx]:
                        coc_dict[program_name][program_idx][program_date] = {}
                    for i, cluster_name in enumerate(cluster_general_intervals[program_name][program_idx][program_date]):
                        beg_time, end_time = cluster_general_intervals[program_name][program_idx][program_date][cluster_name]
                        isok = self._check_for_coc(coc_dict[program_name][program_idx][program_date], beg_time,end_time,cluster_name)
                        if not isok:
                            current_cluster += 1
                            coc_dict[program_name][program_idx][program_date][current_cluster] = {'interval':[beg_time,end_time], 'clusters':[cluster_name,]}
        self._assign_clusters_and_names_coc(coc_dict,speaker_cluster)
        return coc_dict
        
    def _assign_clusters_and_names_coc(self, coc_dict,speaker_cluster):
        pass
                            
        
    """
    def parse_speaker_identification_files(self, file_name):
        speaker_identities = self.initialize_dict()
        with open(file_name,'r') as f_:
            prev_beg_time = -1
            for line in f_:
                line_split = line.strip().split()
                program_detail = line_split[0]
                person_name = line_split[-1]
                probability = float(person_name[person_name.index('<')+1:-1])
                person_name = person_name[:person_name.index('<')]
                time_interval = map(float, parse_time_interval(line_split[1:3]))
                pr_name1, pr_name2, program_date, program_idx = program_detail.split('_')
                program_name = pr_name1+'_'+pr_name2
                speaker_identities[program_name][program_idx][program_date]['interval'].append(time_interval)
                speaker_identities[program_name][program_idx][program_date]['cluster_names'].append(person_name)
                if speaker_identities[program_name][program_idx][program_date]['interval'][-1][0] == prev_beg_time:
                    print 'buldum written'
                prev_beg_time = time_interval[0]
        return speaker_identities
    """    
        
    def _overlap_modes(self,interval_sp, interval_wr, wr_names):
        overlap_list = []        
        for _ in xrange(len(interval_sp)):
            overlap_list.append({})
        interval_sp = np.array(interval_sp)
        interval_wr = np.array(interval_wr)
        for i, inter_sp in enumerate(interval_sp):
            beg_sp,end_sp = inter_sp[0],inter_sp[1]
            if beg_sp>2646 and beg_sp<2648 and end_sp>2652 and end_sp<2656:
                pass
            for inter_wr, name in zip(interval_wr,wr_names):
                beg_wr,end_wr = inter_wr[0],inter_wr[1]
                if beg_wr>2646 and beg_wr<2648 and end_wr>2652 and end_wr<2656:
                    pass
                if beg_wr < beg_sp and end_wr <= beg_sp:
                    continue
                if beg_wr >= end_sp and end_wr > end_sp:
                    #break
                    continue
                if beg_wr <= beg_sp and end_wr <= end_sp:
                    overlap_list[i][name] = overlap_list[i].get(name,0) + end_wr - beg_sp
                elif beg_wr >= beg_sp and end_wr <= end_sp:
                    overlap_list[i][name] = overlap_list[i].get(name,0) + end_wr - beg_wr
                elif beg_wr >= beg_sp and end_wr >= end_sp:
                    overlap_list[i][name] = overlap_list[i].get(name,0) + end_sp - beg_wr
                elif beg_wr <= beg_sp and end_wr >= end_sp:
                    overlap_list[i][name] = overlap_list[i].get(name,0) + end_sp - beg_sp
        return overlap_list
        
    def overlap_modes(self, speaker_cluster, written_names, list_name = 'overlap_list'):
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    if program_name == "LCP_CaVousRegarde" and program_date == "2011-12-20" and program_idx == "204600" and list_name == 'ivector':
                        pass
                    interval_sp = speaker_cluster[program_name][program_idx][program_date]['interval']
                    interval_wr = written_names[program_name][program_idx][program_date]['interval']
                    names_wr = written_names[program_name][program_idx][program_date]['cluster_names']
                    overlap_list = self._overlap_modes(interval_sp, interval_wr, names_wr)
                    speaker_cluster[program_name][program_idx][program_date][list_name] = overlap_list
        return speaker_cluster
        
    def assign_raw_supervised2(self, speaker_cluster,spoken_cluster,name_cand_dict,assigned_dict):
        N_assigned = 0
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    sealed = speaker_cluster[program_name][program_idx][program_date]['sealed']
                    assigned_names = speaker_cluster[program_name][program_idx][program_date]['assigned_names']
                    candidate_names_GMM = speaker_cluster[program_name][program_idx][program_date]["GMM"]
                    candidate_names_SVM = speaker_cluster[program_name][program_idx][program_date]["SVM"]
                    candidate_names_ivector = speaker_cluster[program_name][program_idx][program_date]["ivector"]
                    spoken_labels = spoken_cluster[program_name][program_idx][program_date]["cluster_names"]
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if not sealed[i]:
                            gmm_label = candidate_names_GMM[i].keys()[np.argmax(candidate_names_GMM[i].values())]
                            svm_label = candidate_names_SVM[i].keys()[np.argmax(candidate_names_SVM[i].values())]
                            if not candidate_names_ivector[i]:
                                ivector_label = svm_label
                            else:
                                ivector_label = candidate_names_ivector[i].keys()[np.argmax(candidate_names_ivector[i].values())]
                            gmm_prob = float(gmm_label[gmm_label.index('<')+1:-1])
                            svm_prob = float(svm_label[svm_label.index('<')+1:-1])
                            ivector_prob = float(ivector_label[ivector_label.index('<')+1:-1])
                            gmm_label = gmm_label[:gmm_label.index('<')]
                            svm_label = svm_label[:svm_label.index('<')]
                            ivector_label = ivector_label[:ivector_label.index('<')]
                            name_cands = name_cand_dict[cluster_name]
                            if gmm_label == svm_label == ivector_label: # and (gmm_prob > 0) and (svm_prob > 0): # and (gmm_label in name_cands):
                                sealed[i] = True
                                assigned_names[i] = gmm_label
                                N_assigned += 1
                                assigned_dict[cluster_name].append((gmm_label,'SUP1'))
                            #elif gmm_label == svm_label and (gmm_label in spoken_labels) and (not gmm_label in name_cands):
                            #    sealed[i] = True 
                            #    assigned_names[i] = gmm_label
                            #    N_assigned += 1
                            #    assigned_dict[cluster_name].append((gmm_label,'SUP'))
                            elif gmm_label !=svm_label and (ivector_label in name_cands):
                                sealed[i] = True 
                                assigned_names[i] = ivector_label
                                N_assigned += 1
                                assigned_dict[cluster_name].append((ivector_label,'SUP2'))
        print N_assigned, "cluster is assigned in raw2"
        return speaker_cluster
        
    def assign_with_similarity(self, speaker_cluster,name_cand_dict, spoken_cluster,assigned_dict,mode_choose = 'SUP2'):
        N_assigned = 0
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    sealed = speaker_cluster[program_name][program_idx][program_date]['sealed']
                    assigned_names = speaker_cluster[program_name][program_idx][program_date]['assigned_names']
                    candidate_names_GMM = speaker_cluster[program_name][program_idx][program_date]["GMM"]
                    candidate_names_SVM = speaker_cluster[program_name][program_idx][program_date]["SVM"]
                    candidate_names_ivector = speaker_cluster[program_name][program_idx][program_date]["ivector"]
                    spoken_labels = spoken_cluster[program_name][program_idx][program_date]["cluster_names"]
                    cl_names = speaker_cluster[program_name][program_idx][program_date]["cluster_names"]
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if not sealed[i] and i != 0:
                            gmm_label = candidate_names_GMM[i].keys()
                            svm_label = candidate_names_SVM[i].keys()
                            if candidate_names_ivector[i]:
                                ivector_label = candidate_names_ivector[i].keys()
                            else:
                                ivector_label = svm_label
                            gmm_prob = [float(x[x.index('<')+1:-1]) for x in gmm_label]
                            svm_prob = [float(x[x.index('<')+1:-1]) for x in svm_label]
                            ivector_prob = [float(x[x.index('<')+1:-1]) for x in ivector_label]
                            gmm_label_ = [x[:x.index('<')] for x in gmm_label]
                            svm_label_ = [x[:x.index('<')] for x in svm_label]
                            ivector_label_ = [x[:x.index('<')] for x in ivector_label]
                            name_cands = name_cand_dict[cluster_name]
                            sim_list = []
                            for k, cl_name in enumerate(cl_names[:i]):
                                gmm_sim = candidate_names_GMM[k].keys()
                                svm_sim = candidate_names_SVM[k].keys()
                                if candidate_names_ivector[k]:
                                    ivector_sim = candidate_names_ivector[k].keys()
                                else:
                                    ivector_sim = svm_sim
                                gmm_sim_prob = [float(x[x.index('<')+1:-1]) for x in gmm_sim]
                                svm_sim_prob = [float(x[x.index('<')+1:-1]) for x in svm_sim]
                                ivector_sim_prob = [float(x[x.index('<')+1:-1]) for x in ivector_sim]
                                gmm_sim_ = [x[:x.index('<')] for x in gmm_sim]
                                svm_sim_ = [x[:x.index('<')] for x in svm_sim]
                                ivector_sim_ = [x[:x.index('<')] for x in ivector_sim]
                                conn_weight = 0
                                for nn in gmm_label_:
                                    if nn in gmm_sim_:
                                        conn_weight += 0.5
                                for nn in svm_label_:
                                    if nn in svm_sim_:
                                        conn_weight += 0.8
                                for nn in ivector_label_:
                                    if nn in ivector_sim_:
                                        conn_weight += 1
                                if conn_weight != 0 and cl_name != cluster_name:
                                    sim_list.append((conn_weight,cl_name,assigned_names[k]))
                            sim_list.sort(reverse = True)
                            if not sim_list: continue
                            for cw_,cl_name_,name_ in (sim_list[0],):
                                if name_ in name_cands:
                                    sealed[i] = True
                                    assigned_names[i] = name_
                                    N_assigned += 1
                                    assigned_dict[cluster_name].append((name_,'sim1'))
                                    #break
                                elif assigned_dict[cl_name] and assigned_dict[cl_name][0][1] == mode_choose:
                                    sealed[i] = True
                                    assigned_names[i] = name_
                                    N_assigned += 1
                                    assigned_dict[cluster_name].append((name_,'sim2'))
                                    
                            
        print N_assigned," assigned in sim "
        return speaker_cluster
                                
    
    def assign_with_propoagate(self, speaker_cluster, name_cand_dict, assigned_dict):
        N_assigned = 0
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    sealed = speaker_cluster[program_name][program_idx][program_date]['sealed']
                    assigned_names = speaker_cluster[program_name][program_idx][program_date]['assigned_names']
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if not sealed[i] and assigned_dict[cluster_name]:
                            pp = np.zeros(len(assigned_dict[cluster_name]))
                            for k,(name,method) in enumerate(assigned_dict[cluster_name]):
                                if method == 'SUP1' and name in name_cand_dict[cluster_name]:
                                    pp[k] = 1
                                elif method == 'M2' and name in name_cand_dict[cluster_name]:
                                    pp[k] = 0.8
                                elif method == 'M3' and name in name_cand_dict[cluster_name]:
                                    pp[k] = 0.7
                                #elif method == 'sim1' and name in name_cand_dict[cluster_name]:
                                #    pp[k] = 0.6
                                #elif method == 'sim2' and name in name_cand_dict[cluster_name]:
                                #    pp[k] = 0.5
                                #elif method == 'M2' and name in name_cand_dict[cluster_name]:
                                #    pp[k] = 0.4
                            if pp.sum()>0:
                                sealed[i] = True
                                ind = pp.argmax()
                                assigned_names[i] = assigned_dict[cluster_name][ind][0]
                                N_assigned += 1
                            #elif len(name_cand_dict[cluster_name]) == 1:
                            #    sealed[i] = True
                            #    assigned_names[i] = name_cand_dict[cluster_name][0]
                            #    N_assigned += 1
        print N_assigned, " assigned in propagate"
        return speaker_cluster
                            
        
    def assign_raw_supervised(self, speaker_cluster, mode_name = "GMM", all_assign = False):
        N_assigned = 0
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    sealed = speaker_cluster[program_name][program_idx][program_date]['sealed']
                    assigned_names = speaker_cluster[program_name][program_idx][program_date]['assigned_names']
                    candidate_names_GMM = speaker_cluster[program_name][program_idx][program_date][mode_name]
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if not sealed[i] and candidate_names_GMM[i]:
                            cand_names = candidate_names_GMM[i].keys()
                            cand_name = cand_names[np.argmax(candidate_names_GMM[i].values())]
                            probability = float(cand_name[cand_name.index('<')+1:-1])
                            pers_name = cand_name[:cand_name.index('<')]
                            if probability > 0.5 or all_assign:
                                sealed[i] = True 
                                assigned_names[i] = pers_name
                                N_assigned += 1
        print N_assigned, "cluster is assigned in ", mode_name
        return speaker_cluster
        
    def assign_raw_supervised_cluster_based(self, speaker_cluster, supervised_dict):
        N_assigned = 0
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    sealed = speaker_cluster[program_name][program_idx][program_date]['sealed']
                    assigned_names = speaker_cluster[program_name][program_idx][program_date]['assigned_names']
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if not sealed[i]:
                            cand_names_GMM,cand_probs_GMM,cand_times_GMM = extract_unique_names_probs(supervised_dict[cluster_name]['GMM'])
                            cand_names_SVM,cand_probs_SVM,cand_times_SVM = extract_unique_names_probs(supervised_dict[cluster_name]['SVM'])
                            i_GMM,i_SVM = np.argmax(cand_times_GMM),np.argmax(cand_times_SVM)
                            n_GMM,n_SVM = cand_names_GMM[i_GMM],cand_names_SVM[i_SVM]
                            p_GMM,p_SVM = cand_probs_GMM[i_GMM],cand_probs_SVM[i_SVM]
                            pers_name = n_GMM if p_GMM > p_SVM else n_SVM
                            pp = np.max((p_GMM,p_SVM))
                            if pp > 0:
                                sealed[i] = True 
                                assigned_names[i] = pers_name
                                N_assigned += 1
        print N_assigned, "cluster is assigned in GMM"
        return speaker_cluster
        
    def get_dict_of_supervised(self, speaker_cluster):
        supervised_dict = {}
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    candidate_names_GMM = speaker_cluster[program_name][program_idx][program_date]['GMM']
                    candidate_names_SVM = speaker_cluster[program_name][program_idx][program_date]['SVM']
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if supervised_dict.get(cluster_name,None) is None:
                            supervised_dict[cluster_name] = {'GMM':{},'SVM':{}}
                        for pers_name in candidate_names_GMM[i]:
                            supervised_dict[cluster_name]['GMM'][pers_name] = supervised_dict[cluster_name]['GMM'].get(pers_name,0) + candidate_names_GMM[i][pers_name]
                        for pers_name in candidate_names_SVM[i]:
                            supervised_dict[cluster_name]['SVM'][pers_name] = supervised_dict[cluster_name]['SVM'].get(pers_name,0) + candidate_names_SVM[i][pers_name]
        return supervised_dict 

    
    def temporal_count(self, speaker_cluster):
        temporal_dict = {}
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    previous_cluster_name = -1
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if temporal_dict.get(cluster_name,None) is None:
                            temporal_dict[cluster_name] = {}
                        if i == 0:
                            previous_cluster_name = cluster_name
                            continue
                        temporal_dict[previous_cluster_name][cluster_name] = temporal_dict[previous_cluster_name].get(cluster_name,0) + 1
                        previous_cluster_name = cluster_name
        return temporal_dict
                        
         
    def temporal_assignment(self, speaker_cluster, temporal_dict, result_dict):
        N_assigned = 0
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    sealed = speaker_cluster[program_name][program_idx][program_date]['sealed']
                    assigned_names = speaker_cluster[program_name][program_idx][program_date]['assigned_names']                    
                    previous_cluster_name = -1
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if not sealed[i] and i != 0 and temporal_dict.get(previous_cluster_name,False):
                            cl_names = temporal_dict[previous_cluster_name].keys()
                            cl_freqs = temporal_dict[previous_cluster_name].values()
                            cl_name = cl_names[np.argmax(cl_freqs)]
                            if result_dict[cl_name]: #and len(result_dict[cl_name]) == 1:
                                sealed[i] = True
                                names = result_dict[cl_name].keys()
                                freqs = result_dict[cl_name].values()
                                assigned_names[i] = names[np.argmax(freqs)]
                                #assigned_names[i] = result_dict[cl_name].keys()[0]
                                N_assigned += 1
                        previous_cluster_name = cluster_name
        print N_assigned, "cluster is assigned in temporal assignment"
        return speaker_cluster
                        
    def get_results_untill_now(self, speaker_cluster, with_labels = False):
        result_dict = {}
        correct = 0
        labels = None
        over_all = 0
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    sealed = speaker_cluster[program_name][program_idx][program_date]['sealed']
                    assigned_names = speaker_cluster[program_name][program_idx][program_date]['assigned_names']
                    if with_labels:
                        labels = speaker_cluster[program_name][program_idx][program_date]['labels']
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if result_dict.get(cluster_name,None) is None:
                            result_dict[cluster_name] = {}
                        if sealed[i]:
                            result_dict[cluster_name][assigned_names[i]] = result_dict[cluster_name].get(assigned_names[i],0) + 1
                            if labels and labels[i]:
                                over_all += 1
                                target = labels[i].keys()[np.argmax(labels[i].values())]
                                correct = correct + 1 if target == assigned_names[i] else correct
        print "there are ", correct, " in ", over_all 
        return result_dict
        
    def M2_assignment(self, speaker_cluster,assigned_dict):
        one_to_one = 0
        all_sayac = 0
        M2_dict = {}
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    assigned_names = [None]*len(speaker_cluster[program_name][program_idx][program_date]['cluster_names'])
                    sealed = [False]*len(speaker_cluster[program_name][program_idx][program_date]['cluster_names'])
                    speaker_cluster[program_name][program_idx][program_date]['assigned_names'] = assigned_names
                    speaker_cluster[program_name][program_idx][program_date]['sealed'] = sealed
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        overlap_map = speaker_cluster[program_name][program_idx][program_date]['overlap_list'][i]
                        all_sayac += 1
                        if not cluster_name in M2_dict:
                            M2_dict[cluster_name] = []
                        if len(overlap_map) == 1:
                            one_to_one += 1
                            assigned_names[i] = overlap_map.keys()[0]
                            sealed[i] = True
                            assigned_dict[cluster_name].append((assigned_names[i],'M2'))
                            M2_dict[cluster_name].append(assigned_names[i])
        print one_to_one, ' is assigned in M2', all_sayac
        return speaker_cluster, M2_dict
        
    def propagate_M2(self, speaker_cluster, M2_dict,assigned_dict):
        N_assigned = 0
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    sealed = speaker_cluster[program_name][program_idx][program_date]['sealed']
                    assigned_names = speaker_cluster[program_name][program_idx][program_date]['assigned_names']
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if not sealed[i] and M2_dict[cluster_name] and len(M2_dict[cluster_name]) == 1:
                            sealed[i] = True 
                            assigned_names[i] = M2_dict[cluster_name][0]
                            N_assigned += 1
                            assigned_dict[cluster_name].append((assigned_names[i],'M2_prop'))
        print N_assigned, "cluster is assigned in M2_propagate"
        return speaker_cluster
                    
    def getSpeakerDoc_NameDoc(self, speaker_cluster):
        speaker_document = {}
        name_document = {}
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        overlap_map = speaker_cluster[program_name][program_idx][program_date]['overlap_list'][i]
                        speaker_document[cluster_name] = speaker_document.get(cluster_name, {})
                        for wr_name in overlap_map:
                            speaker_document[cluster_name][wr_name] = speaker_document[cluster_name].get(wr_name, 0) + overlap_map[wr_name]
                            name_document[wr_name] = name_document.get(wr_name, {})
                            name_document[wr_name][cluster_name] = name_document[wr_name].get(cluster_name,0) + overlap_map[wr_name]
        return (speaker_document, name_document)
        
    def stableMatching(self, speaker_document, name_document, name_pers = True):
        # name based matching
        speaker_pref = {}
        speaker_matched = {}
        total_N_cluster = len(speaker_document)
        for cluster_name in speaker_document:
            speaker_matched[cluster_name] = None
            if speaker_document[cluster_name]:
                k_list = np.array(speaker_document[cluster_name].keys())
                v_list = np.array(speaker_document[cluster_name].values())
                for i,person_name in enumerate(k_list):
                    idf = np.sum(name_document[person_name].values())/total_N_cluster
                    v_list[i] = 0 if idf == 0 else v_list[i]/idf
                ind = v_list.argsort()[::-1]
                k_list = k_list[ind]
                speaker_pref[cluster_name] = copy.copy(list(k_list))
            else:
                speaker_pref[cluster_name] = []
                
        name_pref = {}
        name_matched = {}
        total_N_name = len(name_document)
        for person_name in name_document:
            name_matched[person_name] = None
            if name_document[person_name]:
                k_list = np.array(name_document[person_name].keys())
                v_list = np.array(name_document[person_name].values())
                for i,cluster_name in enumerate(k_list):
                    idf = np.sum(speaker_document[cluster_name].values())/total_N_name
                    v_list[i] = 0 if idf == 0 else v_list[i]/idf
                ind = v_list.argsort()[::-1]
                k_list = k_list[ind]
                name_pref[person_name] = copy.copy(list(k_list))
            else:
                name_pref[person_name] = []
                
        if name_pers:
            ischange = True           
            while ischange:
                ischange = False
                for person_name in name_pref:
                    if name_matched[person_name]:
                        continue
                    for candidate in name_pref[person_name]:
                        if speaker_matched[candidate]:
                            if person_name in speaker_pref[candidate]:
                                enemy = speaker_matched[candidate]
                                if speaker_pref[candidate].index(person_name) < speaker_pref[candidate].index(enemy):
                                    speaker_matched[candidate] = person_name
                                    name_matched[person_name] = candidate
                                    ischange = True
                                    break
                        else:
                            speaker_matched[candidate] = person_name
                            name_matched[person_name] = candidate
                            ischange = True
                            break
        else:
            ischange = True           
            while ischange:
                ischange = False
                for cluster_name in speaker_pref:
                    if speaker_matched[cluster_name]:
                        continue
                    for candidate in speaker_pref[cluster_name]:
                        if name_matched[candidate]:
                            if cluster_name in name_pref[candidate]:
                                enemy = name_matched[candidate]
                                if name_pref[candidate].index(cluster_name) < name_pref[candidate].index(enemy):
                                    name_matched[candidate] = cluster_name
                                    speaker_matched[cluster_name] = candidate
                                    ischange = True
                                    break
                        else:
                            name_matched[candidate] = cluster_name
                            speaker_matched[cluster_name] = candidate
                            ischange = True
                            break
        return speaker_matched
        
    def method_M1(self, speaker_cluster, name_pers = True):
        speaker_cluster,_ = self.M2_assignment(speaker_cluster)
        speaker_cluster = self.propagate_M2(speaker_cluster, M2_dict)
        sp_doc, nm_doc = self.getSpeakerDoc_NameDoc(speaker_cluster)
        speaker_matched = self.stableMatching(sp_doc, nm_doc, name_pers = name_pers)
        N_assigned = 0
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    sealed = speaker_cluster[program_name][program_idx][program_date]['sealed']
                    assigned_names = speaker_cluster[program_name][program_idx][program_date]['assigned_names']
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if not sealed[i] and speaker_matched.get(cluster_name,None) is not None:
                            sealed[i] = True
                            assigned_names[i] = speaker_matched[cluster_name]
                            N_assigned += 1
        print N_assigned, "cluster is assigned in method M1"
        return speaker_cluster
        
    def method_M3(self, speaker_cluster, speaker_document, name_document, assigned_dict,name_pers = True):
        ## name perspective
        speaker_matched = {}
        if name_pers:
            total_N_name = len(name_document)
            for person_name in name_document:
                tf_idf = np.zeros(len(name_document[person_name]))
                cluster_names = name_document[person_name].keys()
                if not cluster_names:
                    continue
                for i, cluster_name in enumerate(cluster_names):
                    tf = speaker_document[cluster_name][person_name]
                    idf = np.sum(speaker_document[cluster_name].values())/total_N_name
                    tf_idf[i] = 0 if idf == 0 else tf/idf
                speaker_matched[cluster_names[tf_idf.argmax()]] = person_name
        else:
            total_N_cluster = len(speaker_document)
            for cluster_name in speaker_document:
                #if cluster_name == 'MS41_BFMTV_BFMStory_2012-01-10_175800':
                #    pass
                tf_idf = np.zeros(len(speaker_document[cluster_name]))
                person_names = speaker_document[cluster_name].keys()
                if not person_names:
                    continue
                for i, person_name in enumerate(person_names):
                    tf = name_document[person_name][cluster_name]
                    idf = np.sum(name_document[person_name].values())/total_N_cluster
                    tf_idf[i] = 0 if idf == 0 else tf/idf
                speaker_matched[cluster_name] = person_names[tf_idf.argmax()]
            
        N_assigned = 0
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    sealed = speaker_cluster[program_name][program_idx][program_date]['sealed']
                    assigned_names = speaker_cluster[program_name][program_idx][program_date]['assigned_names']
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        #if cluster_name == 'MS41_BFMTV_BFMStory_2012-01-10_175800':
                        #    pass
                        if not sealed[i] and speaker_matched.get(cluster_name,None) is not None:
                            sealed[i] = True
                            assigned_names[i] = speaker_matched[cluster_name]
                            N_assigned += 1
                            assigned_dict[cluster_name].append((assigned_names[i],'M3'))
        print N_assigned, "cluster is assigned in method M3"
        return speaker_cluster
            
                            
    def write_as_hyp(self, speaker_cluster, file_name, mapping_dict):
         with open(file_name,'w') as file_:
            for program_name in speaker_cluster:
                for program_idx in speaker_cluster[program_name]:
                    for program_date in speaker_cluster[program_name][program_idx]:
                        assigned_names = speaker_cluster[program_name][program_idx][program_date]['assigned_names']
                        for i,cl_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                            pred_name = 'Inconnu_' + cl_name if assigned_names[i] is None else assigned_names[i]
                            pred_name = mapping_dict[pred_name] if pred_name in mapping_dict else pred_name
                            beg,last = speaker_cluster[program_name][program_idx][program_date]['interval'][i]
                            string_ = "%s_%s_%s %s %s %s %s\n"%(program_name,program_date,program_idx,beg,
                                                                last, 'speaker', pred_name)
                            file_.write(string_)
                            
                    
    def assignM1(self, speaker_cluster, file_name):
        speaker_document = {}
        name_document = {}
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        overlap_map = speaker_cluster[program_name][program_idx][program_date]['overlap_list'][i]
                        speaker_document[cluster_name] = speaker_document.get(cluster_name, {})
                        for wr_name in overlap_map:
                            speaker_document[cluster_name][wr_name] = speaker_document[cluster_name].get(wr_name, 0) + overlap_map[wr_name]
                            name_document[wr_name] = name_document.get(wr_name, {})
                            name_document[wr_name][cluster_name] = name_document[wr_name].get(cluster_name,0) + overlap_map[wr_name]
                            
        for wr_name in name_document:
            assign_cluster = None
            largest = 0
            for cluster_name in name_document[wr_name]:
                if name_document[wr_name][cluster_name] > largest:
                    largest = name_document[wr_name][cluster_name]
                    assign_cluster = cluster_name
            if assign_cluster is not None:
                if speaker_document[assign_cluster].get('command',0) == -1:
                    continue
                if len(name_document[wr_name]) == 1:
                    speaker_document[assign_cluster]['command'] = -1
                    speaker_document[assign_cluster]['assigned_name'] = wr_name
                    continue
                if speaker_document[assign_cluster].get('command',0) < largest:
                    speaker_document[assign_cluster]['assigned_name'] = wr_name
                    speaker_document[assign_cluster]['command'] = largest
                     
                
                
        with open(file_name,'w') as file_:
            for program_name in speaker_cluster:
                for program_idx in speaker_cluster[program_name]:
                    for program_date in speaker_cluster[program_name][program_idx]:
                        for i,cl_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                            pred_name = speaker_document[cl_name].get('assigned_name','unknownUmit')
                            beg,last = speaker_cluster[program_name][program_idx][program_date]['interval'][i]
                            string_ = "%s_%s_%s %s %s %s %s\n"%(program_name,program_date,program_idx,beg,
                                                                last, 'speaker', pred_name)
                            file_.write(string_)
        return (speaker_document,name_document)

if __name__ == '__main__':
     list_of_shows = r'C:\Users\daredavil\Documents\hbredin-repere\phase2\lists\uri.test2.lst'
     diarization_file_name = r'C:\Users\daredavil\Documents\hbredin-repere\phase2\auto\speaker\diarization\cross_show_full.test2.mdtm'
     written_file_name = r'C:\Users\daredavil\Documents\hbredin-repere\phase2\auto\written\named_entity_detection\Overlaid_names_aligned.repere'
     spoken_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase2\auto\spoken\named_entity_detection\spoken.4.1.processed.repere'
     GMM_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase2\auto\speaker\identification\trnTRNm_tstTEST2_genind_GMMUBM_cross_show_full.etf0'
     SVM_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase2\auto\speaker\identification\trnTRNm_tstTEST2_genind_GSVSVM_cross_show_full.etf0'
     ivector_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase2\auto\speaker\identification\trnTRNm_tstTEST2_genind_IVECTOR_PCA150_mono_show_full.seg_wise.etf0'
     labels_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase2\groundtruth\speaker.mdtm'
     obj = MMUnsCluster(list_of_shows)
     speaker_cluster,N_input,assigned_dict = obj.parse_diarization_file(diarization_file_name)
     name_cluster = obj.parse_written_name_file(written_file_name)
     speaker_cluster = obj.overlap_modes(speaker_cluster,name_cluster)
     spoken_cluster = obj.parse_written_name_file(spoken_file)
     speaker_cluster = obj.overlap_modes(speaker_cluster, spoken_cluster, 'spoken')
     
     cluster_general_intervals,cluster_frequencies = obj.interval_of_clusters(speaker_cluster)
     labels_dict,_,_ = obj.parse_diarization_file(labels_file)
     speaker_cluster = obj.overlap_modes(speaker_cluster, labels_dict, 'labels')  
     GMM_cluster = obj.parse_etf_file(GMM_file)
     SVM_cluster = obj.parse_etf_file(SVM_file)
     ivector_cluster = obj.parse_etf_file(ivector_file)
     speaker_cluster = obj.overlap_modes(speaker_cluster, GMM_cluster,list_name = 'GMM')
     speaker_cluster = obj.overlap_modes(speaker_cluster, SVM_cluster,list_name = 'SVM')
     speaker_cluster = obj.overlap_modes(speaker_cluster, ivector_cluster,list_name = 'ivector')
     name_cand_dict = obj.test_for_labels(speaker_cluster, cluster_general_intervals,name_cluster,spoken_cluster,cluster_frequencies)     
     #raise
     #coc_dict = obj.cluster_of_clusters_general(speaker_cluster, cluster_general_intervals)     
     
     #GMM_cluster = obj.parse_etf_file(GMM_file)
     #SVM_cluster = obj.parse_etf_file(SVM_file)
     #speaker_cluster = obj.overlap_modes(speaker_cluster, GMM_cluster,list_name = 'GMM')
     #speaker_cluster = obj.overlap_modes(speaker_cluster, SVM_cluster,list_name = 'SVM')
     
     ## isimler hala ali<-0.12> şeklinde
     #speaker_trnA_GMM = obj.parse_written_name_file(speaker_ident_trnA_GMM)
     #speaker_cluster = obj.overlap_modes(speaker_cluster,speaker_trnA_GMM,list_name = 'GMM')
     #speaker_trnA_SVM = obj.parse_written_name_file(speaker_ident_trnA_SVM)
     #speaker_cluster = obj.overlap_modes(speaker_cluster,speaker_trnA_SVM,list_name = 'SVM')
     #labels_dict,_ = obj.parse_diarization_file(labels_file)
     #speaker_cluster = obj.overlap_modes(speaker_cluster, labels_dict, 'labels')
     speaker_document,name_document = obj.getSpeakerDoc_NameDoc(speaker_cluster)
     #speaker_cluster = obj.method_M1(speaker_cluster, name_pers = False)
     speaker_cluster,M2_dict = obj.M2_assignment(speaker_cluster,assigned_dict)
     speaker_cluster = obj.assign_raw_supervised2(speaker_cluster, spoken_cluster, name_cand_dict,assigned_dict)
     speaker_cluster = obj.method_M3(speaker_cluster, speaker_document, name_document, assigned_dict,name_pers = False)
     speaker_cluster = obj.assign_with_similarity(speaker_cluster,name_cand_dict, spoken_cluster,assigned_dict,mode_choose = 'SUP2')
     
     #speaker_cluster = obj.propagate_M2(speaker_cluster,M2_dict,assigned_dict)
     #speaker_cluster = obj.method_M3(speaker_cluster, speaker_document, name_document, assigned_dict,name_pers = False)
     
     speaker_cluster = obj.assign_with_propoagate(speaker_cluster,name_cand_dict,assigned_dict)     
     speaker_cluster = obj.assign_with_similarity(speaker_cluster,name_cand_dict, spoken_cluster,assigned_dict,mode_choose = 'SUP1')
     
     #temporal_dict = obj.temporal_count(speaker_cluster)
     #result_dict = obj.get_results_untill_now(speaker_cluster,with_labels = False)
     #speaker_cluster = obj.temporal_assignment(speaker_cluster, temporal_dict, result_dict)
     #result_dict = obj.get_results_untill_now(speaker_cluster,with_labels = True)
     
     #supervised_dict = obj.get_dict_of_supervised(speaker_cluster)
     """     
     mapping_file1 = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\groundtruth\mapping.txt'
     mapping_file2 = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\dev\groundtruth\mapping.txt'
     mapping_dict = get_mapping_dict([mapping_file1, mapping_file2])
     X_test, y_test,GMM_perf,SVM_perf = extract_features(speaker_cluster, supervised_dict, obj.name_map,mapping_dict,N_input,train_mode = True)     
     with open("sup_train_info.pkl",'rb') as f_:
        sup_train_dict = cPickle.load(f_)
        X_train,y_train = sup_train_dict['X_train'], sup_train_dict['y_train']
        scs = StandardScaler()
        X_train = scs.fit_transform(X_train)
     X_test = scs.transform(X_test)
     rf = RandomForestClassifier(n_estimators = 30, random_state = 500)
     print "training starts..."
     rf.fit(X_train, y_train)
     y_pred = rf.predict(X_test)
     print "GMM: ",GMM_perf
     print "SVM: ",SVM_perf
     print "rf: ", np.sum(y_pred == y_test)
     lr = LogisticRegression(C = 1, random_state = 500)
     lr.fit(X_train, y_train)
     y_pred = lr.predict(X_test)
     print "lr: ", np.sum(y_pred == y_test)
     svc = svm.SVC(kernel = 'rbf',C=1,gamma=0.1, random_state = 500)
     svc.fit(X_train, y_train)
     y_pred = svc.predict(X_test)
     print "svc: ", np.sum(y_pred == y_test)
     """
  #   speaker_cluster = obj.assign_raw_supervised2(speaker_cluster, spoken_cluster, name_cand_dict)
  #   speaker_cluster = obj.assign_with_similarity(speaker_cluster,name_cand_dict, spoken_cluster)
     
     #speaker_cluster = obj.assign_raw_supervised(speaker_cluster,mode_name = "SVM", all_assign = True)
     #result_dict = obj.get_results_untill_now(speaker_cluster,with_labels = True)
     ##speaker_cluster = obj.assign_raw_supervised_cluster_based(speaker_cluster, supervised_dict)
     mapping_file1 = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\groundtruth\mapping.txt'
     mapping_file2 = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\dev\groundtruth\mapping.txt'
     mapping_dict = get_mapping_dict([mapping_file1, mapping_file2])
     fuse_file_name = r'TEST2_SUPERVISED_M3_ITU.hyp'
     obj.write_as_hyp(speaker_cluster, fuse_file_name, mapping_dict)