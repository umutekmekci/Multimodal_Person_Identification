# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:14:31 2013

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
                speaker_cluster[program_name][program_idx][program_date]['interval'].append([time_interval[0],sum(time_interval)])
                speaker_cluster[program_name][program_idx][program_date]['cluster_names'].append(person_name)
                if speaker_cluster[program_name][program_idx][program_date]['interval'][-1][0] == prev_beg_time:
                    print 'buldum speaker'
                prev_beg_time = time_interval[0]
        return speaker_cluster,N
    
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
                written_names[program_name][program_idx][program_date]['interval'].append(time_interval)
                written_names[program_name][program_idx][program_date]['cluster_names'].append(person_name)
                if written_names[program_name][program_idx][program_date]['interval'][-1][0] == prev_beg_time:
                    print 'buldum written'
                prev_beg_time = time_interval[0]
        return written_names
        
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
        
    def overlap_modes(self, speaker_cluster, written_names, list_name = 'overlap_list'):
        for program_name in speaker_cluster:
            for program_idx in speaker_cluster[program_name]:
                for program_date in speaker_cluster[program_name][program_idx]:
                    #if program_name == "LCP_CaVousRegarde" and program_date == "2011-12-20" and program_idx == "204600":
                    #    pass
                    interval_sp = speaker_cluster[program_name][program_idx][program_date]['interval']
                    interval_wr = written_names[program_name][program_idx][program_date]['interval']
                    names_wr = written_names[program_name][program_idx][program_date]['cluster_names']
                    overlap_list = self._overlap_modes(interval_sp, interval_wr, names_wr)
                    speaker_cluster[program_name][program_idx][program_date][list_name] = overlap_list
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
        
    def M2_assignment(self, speaker_cluster):
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
                            M2_dict[cluster_name].append(assigned_names[i])
        print one_to_one, ' is assigned in M2', all_sayac
        return speaker_cluster, M2_dict
        
    def propagate_M2(self, speaker_cluster, M2_dict):
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
        #speaker_cluster = self.propagate_M2(speaker_cluster, M2_dict)
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
        
    def method_M3(self, speaker_cluster, speaker_document, name_document, name_pers = True):
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
        print N_assigned, "cluster is assigned in method M3"
        return speaker_cluster
            
                            
    def write_as_hyp(self, speaker_cluster, file_name):
         with open(file_name,'w') as file_:
            for program_name in speaker_cluster:
                for program_idx in speaker_cluster[program_name]:
                    for program_date in speaker_cluster[program_name][program_idx]:
                        assigned_names = speaker_cluster[program_name][program_idx][program_date]['assigned_names']
                        for i,cl_name in enumerate(speaker_cluster[program_name][program_idx][program_date]['cluster_names']):
                            pred_name = 'Inconnu_' + cl_name if assigned_names[i] is None else assigned_names[i]
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
     list_of_shows = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\lists\uri.lst'
     diarization_file_name = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\auto\speaker\diarization\cross_show_full.mdtm'
     written_file_name = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\auto\written\named_entity_detection\Overlaid_names_aligned.repere'
     speaker_ident_trnA_GMM = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\auto\speaker\identification\trnA_tstTEST_GMMUBM1_umit.repere'
     speaker_ident_trnA_SVM = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\auto\speaker\identification\trnA_tstTEST_GSVSVM1_umit.repere'
     labels_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\groundtruth\speaker.mdtm'
     obj = MMUnsCluster(list_of_shows)
     speaker_cluster,N_input = obj.parse_diarization_file(diarization_file_name)
     name_cluster = obj.parse_written_name_file(written_file_name)
     speaker_cluster = obj.overlap_modes(speaker_cluster,name_cluster)
     ## isimler hala ali<-0.12> şeklinde
     speaker_trnA_GMM = obj.parse_written_name_file(speaker_ident_trnA_GMM)
     speaker_cluster = obj.overlap_modes(speaker_cluster,speaker_trnA_GMM,list_name = 'GMM')
     speaker_trnA_SVM = obj.parse_written_name_file(speaker_ident_trnA_SVM)
     speaker_cluster = obj.overlap_modes(speaker_cluster,speaker_trnA_SVM,list_name = 'SVM')
     labels_dict,_ = obj.parse_diarization_file(labels_file)
     speaker_cluster = obj.overlap_modes(speaker_cluster, labels_dict, 'labels')
     speaker_document,name_document = obj.getSpeakerDoc_NameDoc(speaker_cluster)
     #speaker_cluster = obj.method_M1(speaker_cluster, name_pers = False)
     speaker_cluster,M2_dict = obj.M2_assignment(speaker_cluster)
     speaker_cluster = obj.propagate_M2(speaker_cluster,M2_dict)
     speaker_cluster = obj.method_M3(speaker_cluster, speaker_document, name_document, name_pers = False)
     #temporal_dict = obj.temporal_count(speaker_cluster)
     #result_dict = obj.get_results_untill_now(speaker_cluster,with_labels = True)
     #speaker_cluster = obj.temporal_assignment(speaker_cluster, temporal_dict, result_dict)
     result_dict = obj.get_results_untill_now(speaker_cluster,with_labels = True)
     
     supervised_dict = obj.get_dict_of_supervised(speaker_cluster)
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
     speaker_cluster = obj.assign_raw_supervised(speaker_cluster,mode_name = "SVM", all_assign = True)
     result_dict = obj.get_results_untill_now(speaker_cluster,with_labels = True)
     #speaker_cluster = obj.assign_raw_supervised_cluster_based(speaker_cluster, supervised_dict)
     fuse_file_name = r'Unsupervised_speaker_M3__M2_propagate_only_SVM.hyp'
     obj.write_as_hyp(speaker_cluster, fuse_file_name)