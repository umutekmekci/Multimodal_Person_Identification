# -*- coding: utf-8 -*-
"""
Created on Wed Dec 04 17:08:27 2013

@author: daredavil
"""

from __future__ import division
import json
import copy
import numpy as np
from xmltodict import parse as parse_xml2json
import os
import pickle
from sklearn.linear_model import SGDClassifier

def extract_y_pred(X_test):
    L = X_test.shape[1]
    hknn = np.arange(0,L,10)
    hlin = np.arange(1,L,10)
    hsvm = np.arange(2,L,10)
    sgmm = np.arange(3,L,10)
    ssvm = np.arange(5,L,10)
    wr = np.arange(7,L,10)
    nbl = np.arange(8,L,10)
    sp = np.arange(9,L,10)
    
    sgmm_freq = X_test[:,sgmm]
    r_sgmm,c_sgmm = np.nonzero(sgmm_freq)
    ssvm_freq = X_test[:,ssvm]
    r_ssvm,c_ssvm = np.nonzero(ssvm_freq)
    wr_freq = X_test[:,wr]
    wr_ind = np.zeros(X_test.shape[0])
    r_wr,c_wr = np.nonzero(wr_freq)
    tt = 0
    for i, x in enumerate(xrange(X_test.shape[0])):
        ind = list(set(c_wr[x == r_wr]) | set(c_ssvm[x == r_ssvm]) | set(c_sgmm[x == r_sgmm]))
        #ind = (x == r_wr)
        tt += len(ind)
        wr_ind[i] = np.random.choice(c_wr[ind]) if np.any(ind) else 0 
    #ssvm_freq = X_test[:,ssvm]
    #ssvm_ind = np.nonzero(ssvm_freq)[0]
    #ind = set(wr_ind) & set(ssvm_ind)
    #ind = ind if ind else [ssvm_freq.argmax(axis = 1)]
    #return np.random.choice(ind)
    tt /= X_test.shape[0]
    print "ortalama: ",tt
    return wr_ind
    

class UnknownName(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class ParseTextToJson:
    """
    takes the file names to be combined as a dictionary
    takes list_file list A, list B, listAB or list Test or list Dev
    returns json file
    """
    def __init__(self, file_names_dict, head_nbl_file_name=None, list_file_name=None, mode_names = None):
        self.head_svm_list = file_names_dict.get('head_svm',None)
        self.head_lin_list = file_names_dict.get('head_lin',None)
        self.head_knn_list = file_names_dict.get('head_knn',None)
        self.speaker_gmm_list = file_names_dict.get('speaker_gmm',None) 
        self.speaker_svm_list = file_names_dict.get('speaker_svm',None) 
        #if not self.speaker_gmm.endswith('umit.repere') or not self.speaker_svm.endswith('umit.repere'):
        #    raise UnknownName('speaker files must be converted to include probs. Use hbredin_merge2files.py')
        self.spoken_mrf_list = file_names_dict.get('spoken_mrf', None) 
        self.written_unknown_list = file_names_dict.get('written_unknown',None)
        self.head_nbl_list = head_nbl_file_name
        if list_file_name is None:
            raise UnknownName('List file name cannot be empty')
        self.list_file_name = list_file_name
        self.programs_list = self._get_programs_list()
        file_list = [self.head_svm_list,self.head_lin_list,self.head_knn_list,self.speaker_gmm_list,self.speaker_svm_list,self.spoken_mrf_list,self.written_unknown_list]
        self.file_list = [ff for ll in file_list for ff in ll]        
        self.mode_names = mode_names
    
    def _get_programs_list(self):
        with open(self.list_file_name, 'r') as f_:
            return [line.strip() for line in f_.readlines()] 
    
    def _initialize_dict(self, list_file_name = None,isList = 0):
        """
        construct the dictionary with list file which holds which programs to parse
        isList: 0 for list, 1 for dict
        """
        list_file_name = list_file_name if list_file_name is not None else self.list_file_name
        program_dict = {}
        with open(list_file_name, 'r') as f_:
            for line in f_:
                pr_name1, pr_name2, program_date, program_idx = line.strip().split('_')
                program_name = pr_name1+'_'+pr_name2
                if program_name in program_dict:
                    if program_idx in program_dict[program_name]:
                        program_dict[program_name][program_idx][program_date] = [[],{}][isList]
                    else:
                        program_dict[program_name][program_idx] = {program_date: [[],{}][isList]}
                else:
                    program_dict[program_name] = {program_idx: {program_date: [[],{}][isList]}}
        return program_dict
        
    def _parse_time_interval(self, time_int):
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
        
    def _update_me(self,m1,m2):
        for method_name in m2:
            if method_name in m1:
                m1[method_name].extend(m2[method_name])
            else:
                m1[method_name] = m2[method_name][:]
        
    def _interval_intersect(self, list1, list2):
        """
        divides the time interval of two lists into smallest time_intervals
        ex: time_int1 = 10.2 - 14.8, time_int2 = 12.6 - 15.0 then
        new_time_int = 10.2 - 12.6, 12.6 - 14.8, 14.8 - 15.0
        and updates the person names that share the same time_interval
        """
        if (not list1) and (not list2):
            return []
        if list1 and (not list2):
            return copy.deepcopy(list1)
        if list2 and (not list1):
            return copy.deepcopy(list2)
        big_array = np.zeros((len(list1) + len(list2))*2)  #we collect all timings in the big_array
        for i, (info1, info2) in enumerate(zip(list1, list2)):
            big_array[4*i:4*i+4] = info1['interval'][0], info1['interval'][1], info2['interval'][0], info2['interval'][1]
    
        i = np.min((len(list1),len(list2)))*4  #if list1 has len 10 and list 15 then 20 timings for list1 and 20 timings for list2 will be proeccessed first
        temp_list, cut_point = (list1, len(list2)) if len(list2) <= len(list1) else (list2, len(list1))
        for info in temp_list[cut_point:]:
            big_array[i:i+2] = info['interval'][0], info['interval'][1]
            i+=2
        big_array = np.unique(np.sort(big_array))
        return_list = []  
        for i in xrange(big_array.shape[0]-1):
            return_list.append({'interval':[big_array[i],big_array[i+1]], 'trainer':{}})
        list_of_list = [list1, list2]
        for temp_list in list_of_list:
            for info in temp_list:
                start_ind = np.nonzero(info['interval'][0] == big_array)[0][0]
                end_ind = np.nonzero(info['interval'][1] == big_array)[0][0]
                trainer = info['trainer']
                for big_array_info in return_list[start_ind:end_ind]:
                    self._update_me(big_array_info['trainer'], trainer)
        return return_list
        
    
    def _parse(self, file_name, mode_name):
        program_dict = self._initialize_dict(self.list_file_name)
        with open(file_name,'r') as f_:
            for line in f_:
                line_split = line.strip().split()
                program_detail = line_split[0]
                if not program_detail in self.programs_list:
                    continue
                person_name = line_split[-1]
                time_interval = map(float, self._parse_time_interval(line_split[1:3]))
                pr_name1, pr_name2, program_date, program_idx = program_detail.split('_')
                program_name = pr_name1+'_'+pr_name2
                if program_dict[program_name][program_idx][program_date]:        
                    if program_dict[program_name][program_idx][program_date][-1]['interval']==time_interval: #if two person is in the same time_interval
                        program_dict[program_name][program_idx][program_date][-1]['trainer'][mode_name].append(person_name)
                        continue
                program_dict[program_name][program_idx][program_date].append({'interval':time_interval,'trainer':{mode_name:[person_name,]}})
        return program_dict
                            
            
    def _combine(self, program_dict, temp_dict):
        for program_name in program_dict:
            for program_idx in program_dict[program_name]:
                for program_date in program_dict[program_name][program_idx]:
                    #if program_name == 'BFMTV_BFMStory' and program_idx == '175800' and program_date == '2011-11-28':
                    #    pass
                    program_dict[program_name][program_idx][program_date] = self._interval_intersect(program_dict[program_name][program_idx][program_date], temp_dict[program_name][program_idx][program_date])
                    
    def fit_without_head_nbl(self, json_name=()):
        self.program_dict = self._parse(self.file_list[0],'head_svm')
        for file_name, mode_name in zip(self.file_list[1:],self.mode_names[1:]):
            temp_dict = self._parse(file_name, mode_name)
            self._combine(self.program_dict, temp_dict)
            print mode_name + " combined"
        json_string = self.program_dict
        if json_name:
            json_string = json.dumps(self.program_dict)
            with open(json_name, 'w') as f_:
                f_.write(json_string)
        return json_string
            
            
            
    def _parse_head_nbl(self, nbl_file):
        nbl_dict = {}
        with open(nbl_file,'r') as nbl_file:
            for line in nbl_file:
                line_split = line.strip().split()
                program_detail = line_split[0]
                person_name = line_split[-2] + '<' + line_split[-1] + '>'
                pr_name1, pr_name2, program_date, program_idx, frame_idx = program_detail.split('_')
                frame_idx = float(frame_idx)
                program_name = pr_name1+'_'+pr_name2
                if program_name in nbl_dict:
                    if program_idx in nbl_dict[program_name]:
                        if program_date in nbl_dict[program_name][program_idx]:
                            if frame_idx in nbl_dict[program_name][program_idx][program_date]:
                                nbl_dict[program_name][program_idx][program_date][frame_idx].append(person_name)
                            else:
                                nbl_dict[program_name][program_idx][program_date][frame_idx] = [person_name,]
                        else:
                            nbl_dict[program_name][program_idx][program_date] = {frame_idx:[person_name,]}            
                    else:
                        nbl_dict[program_name][program_idx] = {program_date: {frame_idx:[person_name,]}}
                else:
                    nbl_dict[program_name] = {program_idx:{program_date:{frame_idx:[person_name,]}}}
        return nbl_dict
        
    def _update_nbl(self, info_list, info_dict):
        for frame in info_dict:
            for info in info_list:
                if frame >= info['interval'][0] and frame < info['interval'][1]:
                    if 'nbl' in info['trainer']:
                        info['trainer']['nbl'].append((frame, info_dict[frame][:]))
                    else:
                        info['trainer']['nbl'] = [(frame, info_dict[frame][:])]
    
    def fit_and_add_head_nbl(self, json_name = ()):
        self.fit_without_head_nbl()
        for nbl_file in self.head_nbl_list:
            nbl_dict = self._parse_head_nbl(nbl_file)
            for program_name in self.program_dict:
                if program_name not in nbl_dict: continue
                for program_idx in self.program_dict[program_name]:
                    if program_idx not in nbl_dict[program_name]: continue
                    for program_date in self.program_dict[program_name][program_idx]:
                        if program_date not in nbl_dict[program_name][program_idx]: continue
                        self._update_nbl(self.program_dict[program_name][program_idx][program_date],
                                         nbl_dict[program_name][program_idx][program_date])
        json_string = self.program_dict
        if json_name:
            json_string = json.dumps(self.program_dict)
            with open(json_name, 'w') as f_:
                f_.write(json_string)
        return json_string
    
    
class TargetRel:
    def __init__(self, ref_file_name):
        self.ref_file_name = ref_file_name
        
    def name_map_all(self):
        """
        searches all ref folders, classifier output files and returns name and id map
        """
        
        map1 = {}
        map1_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\groundtruth\mapping.txt'
        with open(map1_file,'r') as f_:
            for line in f_:
                key_, value_ = line.strip().split()
                map1[key_] = value_
                
        map2 = {}
        map2_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\dev\groundtruth\mapping.txt'
        with open(map2_file,'r') as f_:
            for line in f_:
                key_, value_ = line.strip().split()
                map2[key_] = value_
        
        #first ref folders
        ref_train = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\groundtruth\ref'
        ref_test = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\groundtruth\ref'
        ref_dev = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\dev\groundtruth\ref'
        ref_files = [ref_train, ref_test, ref_dev]
        onlyfiles = [ os.path.join(mypath,f) for mypath in ref_files for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) and (f.endswith('0.ref') or f.endswith('ait.ref') ) ]
        print "there are ", len(onlyfiles), " files"
        sayac = 1
        name_id = 0
        name_map = {}
        for file_name in onlyfiles:
            #print "processing ", sayac
            with open(file_name,'r') as h_file:
                doc = h_file.read()
                doc = parse_xml2json(doc)
                for dd in doc['reference']['frame']:
                    for modality in dd:
                        if modality == '@time': continue
                        name_list = dd[modality].split(' ') if  dd[modality] is not None else []
                        for name in name_list:
                            name = name.strip()
                            name = name[1:] if name[0] == '?' else name
                            if map1.get(name, None):
                                name = map1[name]
                                print 'mapping train buldum'
                            if map2.get(name, None):
                                name = map2[name]
                                print 'mapping dev buldum'
                            if name_map.get(name, None) == None:
                                name_map[name] = name_id
                                name_id += 1
                sayac += 1
        #print name_map
        
        #second head mode
        head_folder_train = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\head\recognition'  
        head_folder_test =  r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\auto\head\recognition'
        head_folder_dev = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\dev\auto\head\recognition'
        head_files = [head_folder_train, head_folder_test, head_folder_dev]
        onlyfiles = [ os.path.join(mypath,f) for mypath in head_files for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) and (f.endswith('.repere') or f.endswith('.hyp') ) ]
        print "there are ", len(onlyfiles), " head files"
        for file_name in onlyfiles:
            with open(file_name, 'r') as f_:
                for line in f_:
                    name = line.strip().split()[-1]
                    if map1.get(name, None):
                        name = map1[name]
                        print 'mapping train buldum'
                    if map2.get(name, None):
                        name = map2[name]
                        print 'mapping dev buldum'
                    if name_map.get(name, None) == None:
                        name_map[name] = name_id
                        name_id += 1
        
        #third speak mode
        speaker_folder_train = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\speaker\identification'  
        speaker_folder_test =  r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\auto\speaker\identification'
        speaker_folder_dev = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\dev\auto\speaker\identification'
        speaker_files = [speaker_folder_train, speaker_folder_test, speaker_folder_dev]
        onlyfiles = [ os.path.join(mypath,f) for mypath in speaker_files for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) and f.endswith('1.repere')  ]
        print "there are ", len(onlyfiles), " speaker files"
        for file_name in onlyfiles:
            with open(file_name, 'r') as f_:
                for line in f_:
                    name = line.strip().split()[-1]
                    if map1.get(name, None):
                        name = map1[name]
                        print 'mapping train buldum'
                    if map2.get(name, None):
                        name = map2[name]
                        print 'mapping dev buldum'
                    if name_map.get(name, None) == None:
                        name_map[name] = name_id
                        name_id += 1
                        
        spoken_train = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\spoken\named_entity_detection\spoken.post-processed.repere'
        spoken_test = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\auto\spoken\named_entity_detection\spoken.post-processed.repere'
        spoken_dev = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\dev\auto\spoken\named_entity_detection\spoken.post-processed.repere'
        onlyfiles = [spoken_train, spoken_test, spoken_dev]
        print "there are ", len(onlyfiles), " spoken files"
        for file_name in onlyfiles:
            with open(file_name, 'r') as f_:
                for line in f_:
                    name = line.strip().split()[-1]
                    if map1.get(name, None):
                        name = map1[name]
                        print 'mapping train buldum'
                    if map2.get(name, None):
                        name = map2[name]
                        print 'mapping dev buldum'
                    if name_map.get(name, None) == None:
                        name_map[name] = name_id
                        name_id += 1
                        
        written_train = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\written\named_entity_detection\Overlaid_names_aligned.repere'
        written_test =  r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\auto\written\named_entity_detection\Overlaid_names_aligned.repere'
        written_dev = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\dev\auto\written\named_entity_detection\Overlaid_names_aligned.repere'
        onlyfiles = [written_train, written_test, written_dev]
        print "there are ", len(onlyfiles), " written files"
        for file_name in onlyfiles:
            with open(file_name, 'r') as f_:
                for line in f_:
                    name = line.strip().split()[-1]
                    if map1.get(name, None):
                        name = map1[name]
                        print 'mapping train buldum'
                    if map2.get(name, None):
                        name = map2[name]
                        print 'mapping dev buldum'
                    if name_map.get(name, None) == None:
                        name_map[name] = name_id
                        name_id += 1
        self.map1 = map1
        self.map2 = map2
        print "there are ",name_id," unique names"
        return name_map
        
    def target_names_map(self): 
        """
        uses the ref file path and reads every file in it and returns name_identity map
        name_map['oliver'] = 1
        """
        name_map = {}
        mypath = self.ref_file_name
        onlyfiles = [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) and f.endswith('0.ref') ]
        print "there are ", len(onlyfiles), " files"
        sayac = 1
        name_id = 0
        for file_name in onlyfiles:
            #print "processing ", sayac
            with open(file_name,'r') as h_file:
                doc = h_file.read()
                doc = parse_xml2json(doc)
                for dd in doc['reference']['frame']:
                    for modality in dd:
                        if modality == '@time': continue
                        name_list = dd[modality].split(' ') if  dd[modality] is not None else []
                        for name in name_list:
                            name = name.strip()
                            name = name[1:] if name[0] == '?' else name
                            if name_map.get(name, None) == None:
                                name_map[name] = name_id
                                name_id += 1
                sayac += 1
        #print name_map
        return name_map
        
    def target_id_map(self,name_map):
        id_map = {}
        for name in name_map:
            id_map[name_map[name]] = name
        #id_map[-1] = 'umit'
        return id_map
         
    def ground_truth(self, file_name, name_map = None):
        """
        takes file_name which is a file in the named_ref folder
        takes name_map that is returned by target_names_map
        returns map[frame_time] = {'head':[names], 'speaker':[names],'written':names}
        """
        name_map = self.name_map_all() if name_map is None else name_map
        ground_truth_map = {}
        with open(file_name,'r') as h_file:
            doc = h_file.read()
            doc = parse_xml2json(doc)
            for dd in doc['reference']['frame']:
                frame_time = dd['@time']
                ground_truth_map[frame_time] = {'speaker':[], 'head':[], 'written':[] }
                for modality in dd:
                    if modality == '@time': continue
                    name_list = dd[modality].split(' ') if  dd[modality] is not None else []
                    for name in name_list:
                        name = name.strip()
                        name = name[1:] if name[0] == '?' else name
                        if name in self.map1:
                            name = self.map1[name]
                        if name in self.map2:
                            name = self.map2[name]
                        person_name = name_map.get(name,-1)
                        ground_truth_map[frame_time][modality].append(person_name)
        return ground_truth_map 
        
    def name_search(self, ref_folder, names):
        mypath = ref_folder
        ll = len(mypath)
        search_list = []
        onlyfiles = [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) and f.endswith('0.ref') ]
        for file_name in onlyfiles:
            with open(file_name,'r') as h_file:
                doc = h_file.read()
                doc = parse_xml2json(doc)
                for dd in doc['reference']['frame']:
                    for modality in dd:
                        if modality == '@time': continue
                        name_list = dd[modality].split(' ') if  dd[modality] is not None else []
                        for name in name_list:
                            name = name.strip()
                            name = name[1:] if name[0] == '?' else name
                            if name in self.map1:
                                name = self.map1[name]
                            if name in self.map2:
                                name = self.map2[name]
                            if name in names:
                                search_list.append((file_name[ll:], name))
        return search_list
        
        
class RepereTest:
    def __init__(self,ref_file_name):
        self.target_extract = TargetRel(ref_file_name)
        self.ref_file_name = ref_file_name
        
    
    def get_feature_map2(self, prediction_dict, isTarget = True, pickle_file_name = (), name_map = None,trtst = 'train'):
        """
        takes prediction_dict and extract features according to rules given below.
        """
        """
        extracts person freqs
        """
        file_unknown = 'C:\\Users\\daredavil\\Documents\\Python Scripts\\hbredin' + '\\' + trtst + 'unknown_names2.txt'
        file_unknown_names = open(file_unknown,'w')
        #name_map = self.target_extract.target_names_map() if name_map is None else name_map
        name_map = self.target_extract.name_map_all() if name_map is None else name_map       
        #feature_dim = 9*len(name_map)
        feature_map = {}
        dict_N = len(name_map)
        for program_name in prediction_dict:
            #print program_name
            feature_map[program_name] = {}
            for program_idx in prediction_dict[program_name]:
                feature_map[program_name][program_idx] = {}
                for program_date in prediction_dict[program_name][program_idx]:
                    temp_list = []
                    feature_map[program_name][program_idx][program_date] = temp_list
                    if isTarget:
                        #print program_name + '_' + program_date + '_' + program_idx
                        grf = self.ref_file_name + '\\' + program_name + '_' + program_date + '_' + program_idx + '.ref'
                        ground_truth_map = self.target_extract.ground_truth(grf, name_map)
                    for info in prediction_dict[program_name][program_idx][program_date]:
                        temp_info = {}
                        temp_info['interval'] = info['interval']
                        temp_info['head_target'] = []
                        temp_info['speaker_target'] = []
                        if isTarget:
                            for grt_frame_time in ground_truth_map:
                                if float(grt_frame_time) >= info['interval'][0] and float(grt_frame_time) < info['interval'][1]:
                                    temp_info['head_target'] = copy.copy(ground_truth_map[grt_frame_time]['head'])
                                    temp_info['speaker_target'] = copy.copy(ground_truth_map[grt_frame_time]['speaker'])
                        #feature_vec = np.zeros(feature_dim)
                        feature_vec = {}
                        majority = {}
                        
                        temp_info['feature'] = feature_vec
                        for classifier_name in sorted(info['trainer']):
                            if classifier_name == 'head_knn':
                                for person_name in info['trainer'][classifier_name]:
                                    if person_name in self.target_extract.map1:
                                        person_name = self.target_extract.map1[person_name]
                                    if person_name in self.target_extract.map2:
                                        person_name = self.target_extract.map2[person_name]
                                    person_id = name_map.get(person_name, None)
                                    
                                    if person_id is None:
                                        #print "Unknown name ", person_name
                                        file_unknown_names.write(person_name + '\n')
                                        continue
                                    feature_vec[person_id] = feature_vec.get(person_id,0) + 1                            
                            elif classifier_name == 'head_lin':
                                for person_name in info['trainer'][classifier_name]:
                                    if person_name in self.target_extract.map1:
                                        person_name = self.target_extract.map1[person_name]
                                    if person_name in self.target_extract.map2:
                                        person_name = self.target_extract.map2[person_name]
                                    person_id = name_map.get(person_name, None)
                                    
                                    if person_id is None:
                                        #print "Unknown name ", person_name
                                        file_unknown_names.write(person_name + '\n')
                                        continue
                                    feature_vec[person_id] = feature_vec.get(person_id,0) + 1                               
                            elif classifier_name == 'head_svm':
                                for person_name in info['trainer'][classifier_name]:
                                    if person_name in self.target_extract.map1:
                                        person_name = self.target_extract.map1[person_name]
                                    if person_name in self.target_extract.map2:
                                        person_name = self.target_extract.map2[person_name]
                                    person_id = name_map.get(person_name, None)
                                    
                                    if person_id is None:
                                        #print "Unknown name ", person_name
                                        file_unknown_names.write(person_name + '\n')
                                        continue
                                    feature_vec[person_id] = feature_vec.get(person_id,0) + 1
                            elif classifier_name == 'speaker_gmm':
                                for person_name in info['trainer'][classifier_name]:
                                    name = person_name[:person_name.find('<')]
                                    ll = len(name)
                                    if name in self.target_extract.map1:
                                        name = self.target_extract.map1[name]
                                    if name in self.target_extract.map2:
                                        name = self.target_extract.map2[name]
                                    person_id = name_map.get(name, None)
                                    
                                    if person_id is None:
                                        #print "Unknown name ", name
                                        file_unknown_names.write(name + '\n')
                                        continue
                                    feature_vec[person_id] = feature_vec.get(person_id,0) + 5
                                    prob = person_name[ll + 1:-1]
                                    #feature_vec[num_feature*person_id + 4] = float(prob)
                                
                            elif classifier_name == 'speaker_svm':
                                for person_name in info['trainer'][classifier_name]:
                                    name = person_name[:person_name.find('<')]
                                    ll = len(name)
                                    if name in self.target_extract.map1:
                                        name = self.target_extract.map1[name]
                                    if name in self.target_extract.map2:
                                        name = self.target_extract.map2[name]
                                    person_id = name_map.get(name, None)
                                    
                                    if person_id is None:
                                        #print "Unknown name ", name
                                        file_unknown_names.write(name + '\n')
                                        continue
                                    feature_vec[person_id] = feature_vec.get(person_id,0) + 7
                                    prob = person_name[ll + 1:-1]
                                    #feature_vec[num_feature*person_id + 6] = float(prob)
                            elif classifier_name == 'written_unknown':
                                for person_name in info['trainer'][classifier_name]:
                                    if person_name in self.target_extract.map1:
                                        person_name = self.target_extract.map1[person_name]
                                    if person_name in self.target_extract.map2:
                                        person_name = self.target_extract.map2[person_name]
                                    person_id = name_map.get(person_name, None)
                                    
                                    if person_id is None:
                                        #print "Unknown name ", person_name
                                        file_unknown_names.write(person_name + '\n')
                                        continue
                                    feature_vec[person_id] = feature_vec.get(person_id,0) + 1
                            elif classifier_name == 'spoken_mrf':
                                for person_name in info['trainer'][classifier_name]:
                                    if person_name in self.target_extract.map1:
                                        person_name = self.target_extract.map1[person_name]
                                    if person_name in self.target_extract.map2:
                                        person_name = self.target_extract.map2[person_name]
                                    person_id = name_map.get(person_name, None)
                                    
                                    if person_id is None:
                                        #print "Unknown name ", person_name
                                        file_unknown_names.write(person_name + '\n')
                                        continue
                                    feature_vec[person_id] = feature_vec.get(person_id,0) + 1
                        #    elif classifier_name == 'nbl':
                        #        for nbl_info in info['trainer'][classifier_name]:
                        #            for person_name in nbl_info[1]:
                        #                name = person_name[:person_name.find('<')]
                        #                ll = len(name)
                        #                if name in self.target_extract.map1:
                        #                    name = self.target_extract.map1[name]
                        #                if name in self.target_extract.map2:
                        #                    name = self.target_extract.map2[name]
                        #                person_id = name_map.get(name, None)
                        #                if person_id is None:
                                            #print "Unknown name ", name
                        #                    file_unknown_names.write(name + '\n')
                        #                    continue
                        #                prob = person_name[ll + 1:-1]
                        #                feature_vec[num_feature*person_id + 8] = float(prob)
                        #keys,freqs = majority.keys(), majority.values()
                        #if freqs:
                        #    temp_info['majority'] = keys[np.argmax(freqs)]
                        temp_list.append(temp_info)
        file_unknown_names.close()
        if pickle_file_name:
            with open(pickle_file_name, 'wb') as file_:
                pickle.dump(feature_map, file_)
        return (feature_map, name_map)
        
    def get_feature_map(self, prediction_dict, isTarget = True, pickle_file_name = (), name_map = None,trtst = 'train'):
        """
        takes prediction_dict and extract features according to rules given below.
        """
        """
        head_svm de varmı yokmu
        head_lin de varmı yokmu
        head_knn de varmı yokmu
        nbl deki prob, yosa 0
        speaker_gmm varmı yokmu
        speaker_gmm prob, yoksa 0
        speaker_svm varmı yokmu
        speaker_svm prob, yoksa 0
        written_unknown varmı yokmu
        spoken varmı yokmu
        """
        file_unknown = 'C:\\Users\\daredavil\\Documents\\Python Scripts\\hbredin' + '\\' + trtst + 'unknown_names2.txt'
        file_unknown_names = open(file_unknown,'w')
        #name_map = self.target_extract.target_names_map() if name_map is None else name_map
        name_map = self.target_extract.name_map_all() if name_map is None else name_map       
        #feature_dim = 9*len(name_map)
        feature_map = {}
        num_feature = 10
        for program_name in prediction_dict:
            #print program_name
            feature_map[program_name] = {}
            for program_idx in prediction_dict[program_name]:
                feature_map[program_name][program_idx] = {}
                for program_date in prediction_dict[program_name][program_idx]:
                    temp_list = []
                    feature_map[program_name][program_idx][program_date] = temp_list
                    if isTarget:
                        #print program_name + '_' + program_date + '_' + program_idx
                        grf = self.ref_file_name + '\\' + program_name + '_' + program_date + '_' + program_idx + '.ref'
                        ground_truth_map = self.target_extract.ground_truth(grf, name_map)
                    for info in prediction_dict[program_name][program_idx][program_date]:
                        temp_info = {}
                        temp_info['interval'] = info['interval']
                        temp_info['head_target'] = []
                        temp_info['speaker_target'] = []
                        if isTarget:
                            for grt_frame_time in ground_truth_map:
                                if float(grt_frame_time) >= info['interval'][0] and float(grt_frame_time) < info['interval'][1]:
                                    temp_info['head_target'] = copy.copy(ground_truth_map[grt_frame_time]['head'])
                                    temp_info['speaker_target'] = copy.copy(ground_truth_map[grt_frame_time]['speaker'])
                        #feature_vec = np.zeros(feature_dim)
                        feature_vec = {}
                        majority = {}
                        temp_info['majority'] = 0
                        temp_info['feature'] = feature_vec
                        for classifier_name in sorted(info['trainer']):
                            if classifier_name == 'head_knn':
                                for person_name in info['trainer'][classifier_name]:
                                    if person_name in self.target_extract.map1:
                                        person_name = self.target_extract.map1[person_name]
                                    if person_name in self.target_extract.map2:
                                        person_name = self.target_extract.map2[person_name]
                                    person_id = name_map.get(person_name, None)
                                    majority[person_id] = majority.get(person_id,0) + 0
                                    if person_id is None:
                                        #print "Unknown name ", person_name
                                        file_unknown_names.write(person_name + '\n')
                                        continue
                                    feature_vec[num_feature*person_id] = 1                            
                            elif classifier_name == 'head_lin':
                                for person_name in info['trainer'][classifier_name]:
                                    if person_name in self.target_extract.map1:
                                        person_name = self.target_extract.map1[person_name]
                                    if person_name in self.target_extract.map2:
                                        person_name = self.target_extract.map2[person_name]
                                    person_id = name_map.get(person_name, None)
                                    majority[person_id] = majority.get(person_id,0) + 0
                                    if person_id is None:
                                        #print "Unknown name ", person_name
                                        file_unknown_names.write(person_name + '\n')
                                        continue
                                    feature_vec[num_feature*person_id + 1] = 1
                                
                            elif classifier_name == 'head_svm':
                                for person_name in info['trainer'][classifier_name]:
                                    if person_name in self.target_extract.map1:
                                        person_name = self.target_extract.map1[person_name]
                                    if person_name in self.target_extract.map2:
                                        person_name = self.target_extract.map2[person_name]
                                    person_id = name_map.get(person_name, None)
                                    majority[person_id] = majority.get(person_id,0) + 0
                                    if person_id is None:
                                        #print "Unknown name ", person_name
                                        file_unknown_names.write(person_name + '\n')
                                        continue
                                    feature_vec[num_feature*person_id + 2] = 1
                                
                            elif classifier_name == 'speaker_gmm':
                                for person_name in info['trainer'][classifier_name]:
                                    name = person_name[:person_name.find('<')]
                                    ll = len(name)
                                    if name in self.target_extract.map1:
                                        name = self.target_extract.map1[name]
                                    if name in self.target_extract.map2:
                                        name = self.target_extract.map2[name]
                                    person_id = name_map.get(name, None)
                                    majority[person_id] = majority.get(person_id,0) + 0
                                    if person_id is None:
                                        #print "Unknown name ", name
                                        file_unknown_names.write(name + '\n')
                                        continue
                                    feature_vec[num_feature*person_id + 3] = 1
                                    prob = person_name[ll + 1:-1]
                                    feature_vec[num_feature*person_id + 4] = float(prob)
                                
                            elif classifier_name == 'speaker_svm':
                                for person_name in info['trainer'][classifier_name]:
                                    name = person_name[:person_name.find('<')]
                                    ll = len(name)
                                    if name in self.target_extract.map1:
                                        name = self.target_extract.map1[name]
                                    if name in self.target_extract.map2:
                                        name = self.target_extract.map2[name]
                                    person_id = name_map.get(name, None)
                                    majority[person_id] = majority.get(person_id,0) + 0
                                    if person_id is None:
                                        #print "Unknown name ", name
                                        file_unknown_names.write(name + '\n')
                                        continue
                                    feature_vec[num_feature*person_id + 5] = 1
                                    prob = person_name[ll + 1:-1]
                                    feature_vec[num_feature*person_id + 6] = float(prob)
                            elif classifier_name == 'written_unknown':
                                for person_name in info['trainer'][classifier_name]:
                                    if person_name in self.target_extract.map1:
                                        person_name = self.target_extract.map1[person_name]
                                    if person_name in self.target_extract.map2:
                                        person_name = self.target_extract.map2[person_name]
                                    person_id = name_map.get(person_name, None)
                                    majority[person_id] = majority.get(person_id,0) + 1
                                    if person_id is None:
                                        #print "Unknown name ", person_name
                                        file_unknown_names.write(person_name + '\n')
                                        continue
                                    feature_vec[num_feature*person_id + 7] = 1
                            elif classifier_name == 'spoken_mrf':
                                for person_name in info['trainer'][classifier_name]:
                                    if person_name in self.target_extract.map1:
                                        person_name = self.target_extract.map1[person_name]
                                    if person_name in self.target_extract.map2:
                                        person_name = self.target_extract.map2[person_name]
                                    person_id = name_map.get(person_name, None)
                                    majority[person_id] = majority.get(person_id,0) + 0
                                    if person_id is None:
                                        #print "Unknown name ", person_name
                                        file_unknown_names.write(person_name + '\n')
                                        continue
                                    feature_vec[num_feature*person_id + 9] = 1
                            elif classifier_name == 'nbl':
                                for nbl_info in info['trainer'][classifier_name]:
                                    for person_name in nbl_info[1]:
                                        name = person_name[:person_name.find('<')]
                                        ll = len(name)
                                        if name in self.target_extract.map1:
                                            name = self.target_extract.map1[name]
                                        if name in self.target_extract.map2:
                                            name = self.target_extract.map2[name]
                                        person_id = name_map.get(name, None)
                                        if person_id is None:
                                            #print "Unknown name ", name
                                            file_unknown_names.write(name + '\n')
                                            continue
                                        prob = person_name[ll + 1:-1]
                                        feature_vec[num_feature*person_id + 8] = float(prob)
                        keys,freqs = majority.keys(), majority.values()
                        if freqs:
                            temp_info['majority'] = keys[np.argmax(freqs)]
                        temp_list.append(temp_info)
        file_unknown_names.close()
        if pickle_file_name:
            with open(pickle_file_name, 'wb') as file_:
                pickle.dump(feature_map, file_)
        return (feature_map, name_map)
        
    def unsupervised_pred(self):
        ## read diarization file
        ## read writen names file
        ## apply method 2.
        
        pass
        
    def extract_features(self,  prediction_dict, isTarget = True, target_mode = 'speaker', num_ins = 3191, name_map = None,trtst = 'train',which_func = 1, temporal = 0):
        if which_func == 2:
            feature_map, name_map = self.get_feature_map2(prediction_dict, isTarget = isTarget, name_map = name_map, trtst = trtst)
        else:
            feature_map, name_map = self.get_feature_map(prediction_dict, isTarget = isTarget, name_map = name_map, trtst = trtst)
        print "there are ", len(name_map), " unique people for train"
        num_ins = len(name_map)        
        target_mode = target_mode + '_target'
        
        
        if isTarget:
            
            sayac = 0
            sayac_toplam = 0
            sayac_unknown = 0
            sp_reduced_map = {}
            for program_name in feature_map:
                sp_reduced_map[program_name] = {}
                for program_idx in feature_map[program_name]:
                    sp_reduced_map[program_name][program_idx] = {}
                    for program_date in feature_map[program_name][program_idx]:
                        temp_list = []
                        sp_reduced_map[program_name][program_idx][program_date] = temp_list
                        pre_ind,pre_val = [], []
                        eklenen= 0
                        silinecek = []
                        for info in feature_map[program_name][program_idx][program_date]:
                            for inds in info['feature']:
                                pre_ind.append(inds)
                                pre_val.append(info['feature'][inds])
                            last_uzun = len(info['feature'])
                            if eklenen > 0:
                                for pri, prv in zip(pre_ind[:-last_uzun], pre_val[:-last_uzun]):
                                    info['feature'][pri] = info['feature'].get(pri,0) + prv
                            eklenen += 1
                            silinecek.append(last_uzun)
                            if eklenen == temporal + 1:
                                eklenen = temporal
                                pre_ind = pre_ind[silinecek[0]:]
                                pre_val = pre_val[silinecek[0]:]
                                silinecek = silinecek[1:]
                            sayac_toplam += 1
                            if -1 in info[target_mode]:
                                sayac_unknown += 1
                                tt = np.array(info[target_mode])
                                tt = list(tt[tt != -1])
                                info[target_mode] = tt
                            if info[target_mode]:
                                temp_list.append(info)  #copy.deepcopy can be used
                                sayac += 1
                   
            print "there are %d train instance %d unknown target for %s in %d"%(sayac, sayac_unknown, target_mode[:-7], sayac_toplam)
        else:
            sp_reduced_map = feature_map
            sayac = 0
            sayac_toplam = 0
            for program_name in feature_map:
                for program_idx in feature_map[program_name]:
                    for program_date in feature_map[program_name][program_idx]:
                        pre_ind,pre_val = [], []
                        eklenen = 0
                        silinecek = []
                        for info in feature_map[program_name][program_idx][program_date]:
                            sayac += 1
                            for inds in info['feature']:
                                pre_ind.append(inds)
                                pre_val.append(info['feature'][inds])
                            last_uzun = len(info['feature'])
                            if eklenen > 0:
                                for pri, prv in zip(pre_ind[:-last_uzun], pre_val[:-last_uzun]):
                                    info['feature'][pri] = info['feature'].get(pri,0) + prv
                            silinecek.append(last_uzun)
                            eklenen += 1
                            if eklenen == temporal + 1:
                                eklenen = temporal
                                pre_ind = pre_ind[silinecek[0]:]
                                pre_val = pre_val[silinecek[0]:]
                                silinecek = silinecek[1:]
                            
        print "there are %d %s instance for %s"%(sayac,trtst, target_mode[:-7])
        id_map = self.target_extract.target_id_map(name_map)
        num_feature = 10 if which_func == 1 else 1
        X_train = np.zeros((sayac, num_feature*num_ins))
        majority_result = np.zeros(sayac)
        if isTarget:
            y_train = np.zeros(sayac)
            file_ = open(r'C:\Users\daredavil\Documents\Python Scripts\hbredin\test_labels2.txt','w')
        index = 0
        y_pred = np.zeros(sayac)
        unique_id = 0
        mode_counter = {}
        for program_name in sp_reduced_map:
            for program_idx in sp_reduced_map[program_name]:
                for program_date in sp_reduced_map[program_name][program_idx]:
                    for info in sp_reduced_map[program_name][program_idx][program_date]:
                        info[unique_id] = unique_id
                        unique_id += 1
                        majority_result[index] = info['majority'] if which_func == 1 else 0
                        for idx in info['feature']:
                            value = info['feature'][idx]
                            X_train[index,idx] = value
                        #y_train[index,info['speaker_target']] = 1
                        if isTarget:
                            mode_counter[index] = {}
                            t_id =  info[target_mode][0] if info[target_mode] else 0
                            ### Bu sırayı head için değiştir
                            if which_func == 1:
                                if X_train[index,t_id*num_feature+9]: 
                                    mode_counter[index]['spoken_mrf'] = mode_counter[index].get('spoken_mrf',0) + 1
                                    y_pred[index] = t_id
                                if X_train[index,t_id*num_feature+8]: 
                                    mode_counter[index]['nbl'] = mode_counter[index].get('nbl',0) + 1 
                                    y_pred[index] = t_id
                                if X_train[index,t_id*num_feature+2]: 
                                    mode_counter[index]['head_svm'] = mode_counter[index].get('head_svm',0) + 1
                                    y_pred[index] = t_id
                                if X_train[index,t_id*num_feature+1]: 
                                    mode_counter[index]['head_lin'] = mode_counter[index].get('head_lin',0) + 1
                                    y_pred[index] = t_id
                                if X_train[index,t_id*num_feature]: 
                                    mode_counter[index]['head_knn'] = mode_counter[index].get('head_knn',0) + 1
                                    y_pred[index] = t_id
                                if X_train[index,t_id*num_feature+3]: 
                                    mode_counter[index]['speaker_gmm'] = mode_counter[index].get('speaker_gmm',0) + 1
                                    y_pred[index] = t_id
                                if X_train[index,t_id*num_feature+5]: 
                                    mode_counter[index]['speaker_svm'] = mode_counter[index].get('speaker_svm',0) + 1
                                    y_pred[index] = t_id
                                if X_train[index,t_id*num_feature+7]: 
                                    mode_counter[index]['written_unknown'] = mode_counter[index].get('written_unknown',0) + 1
                                    y_pred[index] = t_id
                                #else:
                                #    y_pred[index] = 0
                            y_train[index] = t_id
                            string_ = "%s_%s_%s %s %s %s %s\n"%(program_name,program_date,program_idx,info['interval'][0],
                                                                    info['interval'][1], 'speaker', id_map[info[target_mode][0]])
                            file_.write(string_)       
                        index+=1
        if isTarget: file_.close()
        non_zero = []
        if isTarget:
            for i in xrange(X_train.shape[1]):
                if np.any(X_train[:,i]):
                    non_zero.append(i)
        rr = (X_train, y_train, non_zero, sp_reduced_map, id_map,name_map,majority_result,mode_counter,y_pred) if isTarget else (X_train,sp_reduced_map,id_map,name_map,majority_result,y_pred)
        return rr
        
class HypWriter:
    def __init__(self, mode = 'speaker'):
        self.mode = mode
        
    def write_as_hype(self,reduced_map, y_pred, id_map, fus_file_name):
        with open(fus_file_name,'w') as file_:
            iteration = 0
            unique_id = 0
            for program_name in reduced_map:
                for program_idx in reduced_map[program_name]:
                    for program_date in reduced_map[program_name][program_idx]:
                        for info in reduced_map[program_name][program_idx][program_date]:
                            if info[unique_id] != unique_id: print "yanlis yazma"
                            unique_id +=1 
                            id_name = y_pred[iteration]
                            iteration += 1
                            pred_name = id_map[id_name]
                            string_ = "%s_%s_%s %s %s %s %s\n"%(program_name,program_date,program_idx,info['interval'][0],
                                                                info['interval'][1], self.mode, pred_name)
                            file_.write(string_)
                            
    def _get_person_name_template(self,info, pred_info_list, id_map):
        mode_name = 'pred_' + self.mode
        t0,t1 = info['interval']
        person_name = info['person_name']
        pred_name = []
        for info_pred in pred_info_list:
            t00,t11 = info_pred['interval']
            if t00 > t1:
                break
            if t00 <= t0 and t11 > t0 and t11 <= t1:
                sure = t11 - t0
                pred_name.append((sure, id_map[info_pred[mode_name]]))
            elif t00 >= t0 and t11 <= t1:
                sure = t11-t00
                pred_name.append((sure, id_map[info_pred[mode_name]]))
            elif t00 >= t0 and t00 <= t1 and t11 >= t1:
                sure = t1-t00
                pred_name.append((sure, id_map[info_pred[mode_name]]))
        if pred_name:
            pred_name.sort()
            person_name = (pred_name[-1][1],)
        return person_name
        
        
    def _parse_time_interval(self, time_int):
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
        
    def _get_temp_dict(self,template_file):
        temp_dict = {}
        with open(template_file) as file_:
            for line in file_:
                line_split = line.split()
                program_detail = line_split[0]
                person_name = line_split[-1]
                time_interval = map(float, self._parse_time_interval(line_split[1:3]))
                pr_name1, pr_name2, program_date, program_idx = program_detail.split('_')
                program_name = pr_name1+'_'+pr_name2
                if program_name in temp_dict:
                    if program_idx in temp_dict[program_name]:
                        if program_date in temp_dict[program_name][program_idx]:
                            if temp_dict[program_name][program_idx][program_date][-1]['interval']==time_interval:
                                temp_dict[program_name][program_idx][program_date][-1]['person_name'].append(person_name)
                            else:
                                temp_dict[program_name][program_idx][program_date].append({'interval':time_interval,'person_name':[person_name,]})
                        else:
                            temp_dict[program_name][program_idx][program_date] = [{'interval':time_interval,'person_name':[person_name,]}]
                
                    else:
                        temp_dict[program_name][program_idx] = {program_date:
                                                            [{'interval':time_interval,'person_name':[person_name,]}]
                                                       }
                else:
                    #print program_name
                    temp_dict[program_name] = {program_idx: {program_date: [{'interval':time_interval,'person_name':[person_name,]}]}}
        return temp_dict
                            
    def write_as_hyp_template(self,reduced_map, template_file, id_map,y_pred,fuse_file_name):
        template_map = self._get_temp_dict(template_file)
        iteration = 0
        unique_id = 0
        mode_name = 'pred_'+ self.mode
        for program_name in reduced_map:
            for program_idx in reduced_map[program_name]:
                for program_date in reduced_map[program_name][program_idx]:
                    for info in reduced_map[program_name][program_idx][program_date]:
                        if info[unique_id] != unique_id: print "yanlis yazma template"
                        unique_id += 1
                        info[mode_name] = y_pred[iteration]
                        iteration += 1
        with open(fuse_file_name,'w') as file_:
            for program_name in template_map:
                for program_idx in template_map[program_name]:
                    for program_date in template_map[program_name][program_idx]:
                        for info in template_map[program_name][program_idx][program_date]:
                            pred_name = self._get_person_name_template(info, reduced_map[program_name][program_idx][program_date], id_map)
                            for per_name in pred_name:
                                string_ = "%s_%s_%s %s %s %s %s\n"%(program_name,program_date,program_idx,info['interval'][0],
                                                                    info['interval'][1], self.mode, per_name)
                                file_.write(string_)          
        
        
    


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    file_names_dict = {}
    head_svm_trnA_tstB = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\head\recognition\trnA_tstB_KIT-DCT-SVM.hyp'
    head_svm_trnB_tstA = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\head\recognition\trnB_tstA_KIT-DCT-SVM.hyp'
    head_knn_trnA_tstB = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\head\recognition\trnA_tstB_LEAR_LDML_KNN_3.repere'
    head_knn_trnB_tstA = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\head\recognition\trnB_tstA_LEAR_LDML_KNN_3.repere'
    head_lin_trnA_tstB = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\head\recognition\trnA_tstB_LEAR_LDML_SVM_LIN_3.repere'
    head_lin_trnB_tstA = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\head\recognition\trnB_tstA_LEAR_LDML_SVM_LIN_3.repere'
    speaker_gmm_trnA_tstB = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\speaker\identification\trnA_tstB_GMMUBM1_umit.repere'
    speaker_gmm_trnB_tstA = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\speaker\identification\trnB_tstA_GMMUBM1_umit.repere'
    speaker_svm_trnA_tstB = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\speaker\identification\trnA_tstB_GSVSVM1_umit.repere'
    spoken_mrf_trnA_tstB = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\spoken\named_entity_detection\spoken.post-processed.repere'
    written_unknown_trnA_tstB = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\written\named_entity_detection\Overlaid_names_aligned.repere'
    
    file_names_dict['head_svm'] = [head_svm_trnA_tstB,head_svm_trnB_tstA]
    file_names_dict['head_knn'] = [head_knn_trnA_tstB,head_knn_trnB_tstA]
    file_names_dict['head_lin'] = [head_lin_trnA_tstB,head_lin_trnB_tstA]
    file_names_dict['speaker_gmm'] = [speaker_gmm_trnA_tstB,speaker_gmm_trnB_tstA]
    file_names_dict['speaker_svm'] = [speaker_svm_trnA_tstB,]
    file_names_dict['spoken_mrf'] = [spoken_mrf_trnA_tstB,]
    file_names_dict['written_unknown'] = [written_unknown_trnA_tstB,]
    list_file_name = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\lists\uri.lst'
    head_nbl_trnA_tstB = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\head\recognition\trnA_tstB_KIT-DCT-SVM.nbl'
    head_nbl_trnB_tstA = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\auto\head\recognition\trnB_tstA_KIT-DCT-SVM.nbl'
    head_nbl = [head_nbl_trnA_tstB,head_nbl_trnB_tstA]  
    mode_names = ('head_svm', 'head_svm', 'head_lin', 'head_lin', 'head_knn', 'head_knn', 'speaker_gmm', 'speaker_gmm', 'speaker_svm', 'spoken_mrf', 'written_unknown')
    #json_name = 'train_json'
    parserTrain = ParseTextToJson(file_names_dict, head_nbl, list_file_name, mode_names)
    train_dict = parserTrain.fit_and_add_head_nbl()
    print "train dict finished"    
    
    file_names_dict = {}
    base_path = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\auto'
    head_path = base_path + r'\head\recognition'
    speaker_path = base_path + r'\speaker\identification'
    spoken_path = base_path + r'\spoken\named_entity_detection'
    written_path = base_path + r'\written\named_entity_detection'
    file_names_dict['head_svm'] = [head_path + r'\trnA_tstTST_KIT-DCT-SVM.hyp',]
    file_names_dict['head_knn'] = [head_path + r'\trnA_tstTest_LEAR_LDML_KNN_3.repere',]
    file_names_dict['head_lin'] = [head_path + r'\trnA_tstTest_LEAR_LDML_SVM_LIN_3.repere',]
    file_names_dict['speaker_gmm'] = [speaker_path + r'\trnA_tstTEST_GMMUBM1_umit.repere',]
    file_names_dict['speaker_svm'] = [speaker_path + r'\trnA_tstTEST_GSVSVM1_umit.repere',]
    file_names_dict['spoken_mrf'] = [spoken_path + r'\spoken.post-processed.repere',]
    file_names_dict['written_unknown'] = [written_path + r'\Overlaid_names_aligned.repere',]
    list_file_name = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\lists\uri.lst'
    head_nbl = [head_path + r'\trnA_tstTST_KIT-DCT-SVM.nbl',]
    mode_names = ('head_svm', 'head_lin', 'head_knn', 'speaker_gmm', 'speaker_svm', 'spoken_mrf', 'written_unknown')
    parserTest = ParseTextToJson(file_names_dict, head_nbl,list_file_name, mode_names)
    test_dict = parserTest.fit_and_add_head_nbl()
    print "test dict finished"    
    
    print "train feature extraction starts"    
    repere_test = RepereTest(r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\groundtruth\ref')
    X_train, y_train, non_zero, sp_reduced_map, id_map_train, name_map_train, _ ,mc_train,y_pred_train= repere_test.extract_features(train_dict, isTarget = True, target_mode = 'speaker', which_func = 1,temporal = 700)    
    print "correct: ", np.sum(y_pred_train==y_train), " in ", y_train.shape[0]    
    #X_train = X_train[:,non_zero]    
    #random_state = 500  
    #clf = RandomForestClassifier(n_estimators = 30)
    #clf = SGDClassifier(loss="hinge", penalty="l2", n_iter = 100, alpha = 0.0001, random_state=random_state)
    #clf = LogisticRegression(C = 10, random_state=random_state)    
    #clf.fit(X_train,y_train)
    #y_pred = clf.predict(X_train)
    #hyp_writer = HypWriter(mode = 'speaker')
    #fuse_file_name = r'C:\Users\daredavil\Documents\Python Scripts\hbredin\fusTrain_speaker.hyp'
    #hyp_writer.write_as_hype(sp_reduced_map,y_pred,id_map_train,fuse_file_name)
    
    print "test feature extraction starts"
    repere_test.ref_file_name = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\groundtruth\ref'
    X_test,y_test,_,sp_reduced_map, id_map_test, name_map_test, mjr,mc_test,y_pred_test= repere_test.extract_features(test_dict, isTarget = True, target_mode = 'speaker', name_map = name_map_train,trtst = 'test',which_func = 1,temporal = 700)
    print "correct: ", np.sum(y_pred_test==y_test), " in ", y_test.shape[0] 
    wr_ind = extract_y_pred(X_test)
    print "correct: ", np.sum(wr_ind==y_test), " in ", y_test.shape[0] 
    #mjr = X_test.argmax(axis = 1)
    #y_pred = mjr    
    #X_test = X_test[:,non_zero]
    #y_pred = mjr
    #y_pred = clf.predict(X_test)
    #print "correct: ", np.sum(y_pred==y_test), " in ", y_test.shape[0]
    #mjr = X_test.argmax(axis = 1)
    #print "correct: ", np.sum(mjr==y_test), " in ", y_test.shape[0]   
    #print "correct: ", np.sum(X_test.argmax(axis = 1)==y_test), " in ", y_test.shape[0] 

    #tt = 0    
    #for i,y_id in enumerate(y_test):
    #    if not X_test[i,y_id]:
    #        tt += 1
    #print "imposible is ", tt, " possible is ",y_test.shape[0] - tt
    
    #names = [id_map_test[idx] for idx in (set(y_test)- set(y_train))]
    #search_list = repere_test.target_extract.name_search(r'C:\Users\daredavil\Documents\hbredin-repere\phase1\train\groundtruth\ref', names)
    
    #y_pred = y_pred_test
    #hyp_writer = HypWriter(mode = 'speaker')
    #fuse_file_name = r'C:\Users\daredavil\Documents\Python Scripts\hbredin\fusTest_speaker.hyp'
    #hyp_writer.write_as_hype(sp_reduced_map,y_pred,id_map_test,fuse_file_name)

    #template_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\submissions\T1.1_supervised_multimodal_speaker\QCompere.PRIMARY.Tintin_Et_Milou.repere'
    #fuse_file_name = r'C:\Users\daredavil\Documents\Python Scripts\hbredin\fusTestTemplate_speaker.hyp'
    #hyp_writer.write_as_hyp_template(sp_reduced_map,template_file,id_map_test,y_pred,fuse_file_name)
    
    #template_file = r'C:\Users\daredavil\Documents\hbredin-repere\phase1\test\auto\fusion\unsupervised_M2_cross.repere'
    #fuse_file_name = r'C:\Users\daredavil\Documents\Python Scripts\hbredin\fusTestTemplate2_speaker.hyp'
    #hyp_writer.write_as_hyp_template(sp_reduced_map,template_file,id_map_test,y_pred,fuse_file_name)
    
   # to_pickle1 = {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}
   # to_pickle2 = {'majority':X_test.argmax(axis = 1)}    
   # with open(r'repere_data_temp10.pkl','wb') as f_:
   #     pickle.dump(to_pickle1, f_)
   # with open(r'test_majority_temp10.pkl','wb') as f_:
   #     pickle.dump(to_pickle2, f_)
    
    print "testing finished"
    
    mm_train = dict(zip(['head_svm', 'head_lin', 'head_knn', 'speaker_gmm', 'speaker_svm', 'spoken_mrf', 'written_unknown','nbl'],[0]*8))
    for index in mc_train:
        for mode in mc_train[index]:
            mm_train[mode] += 1
            
    mm_test = dict(zip(['head_svm', 'head_lin', 'head_knn', 'speaker_gmm', 'speaker_svm', 'spoken_mrf', 'written_unknown','nbl'],[0]*8))
    for index in mc_test:
        for mode in mc_test[index]:
            mm_test[mode] += 1
            
    for mode in mm_train:
        print mode, ': ',mm_train[mode]
    print        
    for mode in mm_test:
        print mode, ': ',mm_test[mode]