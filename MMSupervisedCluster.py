# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:23:58 2014

@author: daredavil
"""

from __future__ import division
import copy
import numpy as np
from parseTextToJson import TargetRel
from sklearn.ensemble import RandomForestClassifier
import cPickle

def get_mapping_dict(file_names):
    mapping_dict = {}
    for file_name in file_names:
        with open(file_name,'r') as f_:
            for line in f_:
                line = line.strip().split()
                key_,value_ = line[0], line[1]
                if key_ in mapping_dict:
                    continue
                mapping_dict[key_] = value_
    return mapping_dict            

def extract_features(speaker_cluster, supervised_dict, name_dict,mapping_dict, N_input, train_mode = True):
    features_dict = {}
    for i,cluster_name in enumerate(supervised_dict):
        features_dict[cluster_name] = np.zeros(len(name_dict))
        for pers_name in supervised_dict[cluster_name]['GMM']:
            if not pers_name in name_dict:
                id_ = name_dict[mapping_dict[pers_name]]
            else:
                id_ = name_dict[pers_name]
            features_dict[cluster_name][id_] = supervised_dict[cluster_name]['GMM'][pers_name]
        for pers_name in supervised_dict[cluster_name]['SVM']:
            if features_dict[cluster_name][id_] == 0:
                features_dict[cluster_name][id_] = supervised_dict[cluster_name]['SVM'][pers_name]
    
    GMM_perf,SVM_perf = 0,0
    labels = None
    X_train = np.zeros((N_input,len(name_dict)))
    y_train = np.zeros(len(name_dict))
    inst_num = 0
    for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    mode_GMM = speaker_cluster[program_name][program_idx][program_date]['GMM']
                    mode_SVM = speaker_cluster[program_name][program_idx][program_date]['SVM']
                    if speaker_cluster[program_name][program_idx][program_date].get("labels",-1) != -1:
                        labels = speaker_cluster[program_name][program_idx][program_date]['labels']
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if train_mode and not labels[i]:
                            continue
                        if labels and labels[i]:
                            target = labels[i].keys()[np.argmax(labels[i].values())]
                        GMM_pred = mode_GMM[i].keys()[np.argmax(mode_GMM[i].values())] if (mode_GMM[i] and labels and labels[i]) else "Olivier_TRUCHOT_"
                        SVM_pred = mode_SVM[i].keys()[np.argmax(mode_SVM[i].values())] if (mode_SVM[i] and labels and labels[i]) else "Olivier_TRUCHOT_"
                        if not target in name_dict and not target in mapping_dict:
                            continue
                        if GMM_pred == target:
                            GMM_perf += 1
                        if SVM_pred == target:
                            SVM_perf += 1
                        X_train[inst_num] = features_dict[cluster_name]
                        #if train_mode:
                        y_train[inst_num] = name_dict[target] if target in name_dict else name_dict[mapping_dict[target]]
                        inst_num += 1
    X_train = X_train[:inst_num]
    y_train = y_train[:inst_num]
    return X_train, y_train,GMM_perf,SVM_perf
                
        

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

class MMSupCluster:
    def __init__(self, list_of_shows):
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
                        speaker_cluster[program_name][program_idx][program_date] = {'interval':[], 'cluster_names':[],'labels':[],'GMM':[],'SVM':[]}
                    else:
                        speaker_cluster[program_name][program_idx] = {program_date:  {'interval':[], 'cluster_names':[],'labels':[],'GMM':[],'SVM':[]}}
                else:
                    speaker_cluster[program_name] = {program_idx: {program_date: {'interval':[], 'cluster_names':[],'labels':[],'GMM':[],'SVM':[]}}}
        return speaker_cluster
        
    def parse_diarization_file(self, file_name):
        """
        prog_name:
            prog_id:
                prog_date:
                    'interval':[ [],[],... ]
                    'cluster_names':[ ... ]
                    'GMM': [{'Ali':4,'Veli':2},{...}]
                    'SVM': ...
                    'labels': [[123,333], 567]
        """
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
        return speaker_cluster,N
        
    def parse_mode_file(self, file_name):
        mode_dict = self.initialize_dict()
        with open(file_name,'r') as f_:
            prev_beg_time = -1
            for line in f_:
                line_split = line.strip().split()
                program_detail = line_split[0]
                person_name = line_split[-1]
                time_interval = map(float, parse_time_interval(line_split[1:3]))
                pr_name1, pr_name2, program_date, program_idx = program_detail.split('_')
                program_name = pr_name1+'_'+pr_name2
                mode_dict[program_name][program_idx][program_date]['interval'].append(time_interval)
                mode_dict[program_name][program_idx][program_date]['cluster_names'].append(person_name)
                if mode_dict[program_name][program_idx][program_date]['interval'][-1][0] == prev_beg_time:
                    print 'buldum written'
                prev_beg_time = time_interval[0]
        return mode_dict
        
    def _overlap_modes(self,interval_sp, interval_wr, wr_names):
        overlap_list = []        
        for _ in xrange(len(interval_sp)):
            overlap_list.append({})
        interval_sp = np.array(interval_sp)
        interval_wr = np.array(interval_wr)
        for i, inter_sp in enumerate(interval_sp):
            beg_sp,end_sp = inter_sp[0],inter_sp[1]
            for inter_wr, name in zip(interval_wr,wr_names):
                beg_wr,end_wr = inter_wr[0],inter_wr[1]
                if beg_wr < beg_sp and end_wr <= beg_sp:
                    continue
                if beg_wr >= end_sp and end_wr > end_sp:
                    break
                if beg_wr <= beg_sp and end_wr <= end_sp:
                    overlap_list[i][name] = overlap_list[i].get(name,0) + end_wr - beg_sp
                elif beg_wr >= beg_sp and end_wr <= end_sp:
                    overlap_list[i][name] = overlap_list[i].get(name,0) + end_wr - beg_wr
                elif beg_wr >= beg_sp and end_wr >= end_sp:
                    overlap_list[i][name] = overlap_list[i].get(name,0) + end_sp - beg_wr
                elif beg_wr <= beg_sp and end_wr >= end_sp:
                    overlap_list[i][name] = overlap_list[i].get(name,0) + end_sp - beg_sp
        return overlap_list
        
    def overlap_modes(self, speaker_cluster, mode_dict, mode_name = 'GMM'):
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    interval_sp = speaker_cluster[program_name][program_idx][program_date]['interval']
                    interval_wr = mode_dict[program_name][program_idx][program_date]['interval']
                    names_wr = mode_dict[program_name][program_idx][program_date]['cluster_names']
                    overlap_list = self._overlap_modes(interval_sp, interval_wr, names_wr)
                    speaker_cluster[program_name][program_idx][program_date][mode_name] = overlap_list
        return speaker_cluster
        
    def merge_clusters(self, speaker_cluster):
        supervised_dict = {}
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    mode_GMM = speaker_cluster[program_name][program_idx][program_date]['GMM']
                    mode_SVM = speaker_cluster[program_name][program_idx][program_date]['SVM']
                    for i, cluster_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                        if supervised_dict.get(cluster_name,None) is None:
                            supervised_dict[cluster_name] = {'GMM':{},'SVM':{}}
                        for pers_name in mode_GMM[i]:
                            supervised_dict[cluster_name]['GMM'][pers_name] = supervised_dict[cluster_name]['GMM'].get(pers_name,0) + mode_GMM[i][pers_name]
                        for pers_name in mode_SVM[i]:
                            supervised_dict[cluster_name]['SVM'][pers_name] = supervised_dict[cluster_name]['SVM'].get(pers_name,0) + mode_SVM[i][pers_name]
        return supervised_dict 


if __name__ == '__main__':
    GMM_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\speaker\identification\trnA_tstB_GMMUBM1.repere'
    SVM_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\speaker\identification\trnA_tstB_GSVSVM1.repere'
    diarization_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\speaker\diarization\cross_show_full.B.mdtm'
    labels_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\groundtruth\speaker.mdtm'
    list_of_shows = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\lists\uri.B.lst'
    obj = MMSupCluster(list_of_shows)
    speaker_cluster,N_input = obj.parse_diarization_file(diarization_file)
    print "there are ", N_input, " instances"
    GMM_dict = obj.parse_mode_file(GMM_file)
    SVM_dict = obj.parse_mode_file(SVM_file)
    labels_dict,_ = obj.parse_diarization_file(labels_file)
    speaker_cluster = obj.overlap_modes(speaker_cluster, GMM_dict, 'GMM')
    speaker_cluster = obj.overlap_modes(speaker_cluster, SVM_dict, 'SVM')
    speaker_cluster = obj.overlap_modes(speaker_cluster, labels_dict,'labels')
    supervised_dict = obj.merge_clusters(speaker_cluster)
    mapping_file1 = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\groundtruth\mapping.txt'
    mapping_file2 = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\dev\groundtruth\mapping.txt'
    mapping_dict = get_mapping_dict([mapping_file1, mapping_file2])
    X_train, y_train,GMM_perf,SVM_perf = extract_features(speaker_cluster, supervised_dict, obj.name_map,mapping_dict,N_input,train_mode = True)
    rf = RandomForestClassifier(n_estimators = 20)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    print "true: ",np.sum(y_pred == y_train), " in ", y_train.shape[0]
    sup_train_dict = {'X_train':X_train, 'y_train':y_train}
    with open("sup_train_info.pkl",'wb') as f_:
        cPickle.dump(sup_train_dict,f_)