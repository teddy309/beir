## Code by LSS(github: teddy309)
## date: 2022.06.06 ~ Current(on updating)

import os
import jsonlines
import json
from collections import OrderedDict

import pandas as pd

'''
settings: argparse
'''
import argparse

def addArguments(parser):
    parser.add_argument('--dataset_name', default='scifact',
                        help='dataset name', choices=['scifact', 'ms_marco', 'fever'])
    parser.add_argument('--colloquial_index', default='0',
                        help='dataset name', choices=['0','1','2'])
    
    parser.add_argument('-preprocess','--run_preprocess_collquial2beir_pythonfile', default=False,
                        help='if run this .py file for preprocessing fever-style beir-data into colloquial-style at datasets/colloquial/',type=bool)
    parser.add_argument('-ifmake2','--if_preprocess2_mk_qrel_tsv', default=False,
                        help='Preprocess 2: open qrel folder(train/dev/test) for each id, store (query/corpus) id at qrel/ by tsv file (default false)',type=bool)
    parser.add_argument('-ifmake3','--if_preprocess3_mk_col2fever_style_jsonl', default=True,
                        help='Preprocess 3: open tsv files(train/dev/test) at qrel/ folder at preprocess2, with each ids, transfer each for fever claims into colloquial text and store (query/corpus).jsonl at colloquial_dataset path (default false)',type=bool)
    parser.add_argument('--view_IRresult_samples', default=True,
                        help='print first each 10 data samples at main (default false)',type=bool)

    parser.add_argument('--home_path', default=os.getcwd(),
                        help='home directory path of repository')
    parser.add_argument('--dataset_path', default=os.path.join(os.getcwd(), "datasets/"),
                        help='dataset path', choices=['/../datasets/', '/..','/datasets/'])
    parser.add_argument('-ccjp','--colloquialclaims_jsonl_dir', default= '/home/nlplab/hdd4/lss/dataset_ir/colloquial-claims/colloquial_data/',
                        help='logging file(.log) directory name') #colClaims_jsonl_dir = '/home/nlplab/hdd4/lss/dataset_ir/colloquial-claims/colloquial_data/'
    parser.add_argument('-fvjp','--feverclaims_jsonl_dir', default= os.path.join(os.getcwd(), "datasets/fever/"),
                        help='logging file(.log) directory name')
    parser.add_argument('--out_dir_name', default= '/runs',
                        help='logging file(.log) directory name')
    parser.add_argument('--fever_dataPath_dir', default= os.path.join(os.getcwd(),"datasets/fever/"),
                        help='data directory path for fever(beir) output for testing. corpus,queries.jsonl and /qrels/train,valid,test.tsv')
    parser.add_argument('--colloquial_dataPath_dir', default= os.path.join(os.getcwd(),"datasets/colloquial/"),
                        help='data directory path for colloquial(preprocessed) output for testing. corpus,queries.jsonl and /qrels/train,valid,test.tsv')
    return parser

  ## 3. from feverID_lists, get jsonl_queries : read fever(corpus,queries) -> colloquial에서 각각의 jsonl_queries 리스트 받아와서 fever의 corpus/queries에 추가.
    ##       -> 대화형태의 corpus/queries.json을 colloquial/에 저장.
    ##      -> jsonl 형태에

'''
TODO: colloquial 데이터 qrel 형태로 만들기
- from_dir(/home/nlplab/hdd4/lss/dataset_ir/colloquial-claims/colloquial_data/)
- out_dir(/home/nlplab/hdd4/lss/beir/datasets/colloquial/)
- corpus/json.jsonl 하나랑, qrels 안에 (train/test.tsv) 형태로
'''

def readJsonl_idList(jsonl_path):
    id_list = []
    dict_colclaims = {}
    with jsonlines.open(jsonl_path) as f:
        for i, line in enumerate(f):
            #id_list.append(line['fever_id'])
            dict_colclaims[line['fever_id']] = line['colloquial_claims'][0]

            #print(f'[{i}] {id_list}') #dict{'colloquial_claims':['str'*3],'fever_claim':"str",'fevr_label':'SUPPORTS','evidences':[],'gold_evidence_set':[],'fever_id':int}
            #dict_keys - corpus:['_id', 'title', 'text', 'metadata'], queries:['_id', 'text', 'metadata']
            dict_line = OrderedDict()
            dict_metadata = OrderedDict()

            dict_metadata["verifiable"] = "VERIFIABLE"
            dict_metadata["label"] = line['fever_label']
            dict_metadata["evidence"] = []

            dict_line["_id"] = "ss"
            dict_line["text"] = ""
            dict_line["metadata"] = dict_metadata
    
    id_list = list(dict_colclaims.keys()) #list(int)
    #print(f'print here... id_list: {len(id_list), type(id_list), type(id_list[0])}') 

    return id_list, dict_colclaims

def checkJsonl_fromIDlist(jsonl_path, idList): #colloquial 다 있음.
    idx_exist = []
    with jsonlines.open(jsonl_path) as f:
        for i, line in enumerate(f):
            if line['fever_id'] in idList:
                idx_exist.append(line['fever_id'])
    print(f'og_idList_len:{len(idList)}, len_idxExist:{len(idx_exist)}')
    
    return idx_exist

#colloquial jsonl 읽어와서 DF(df_qrels)을 리턴.(main에서 train/dev/test 각각 )
def getDF_qrels_fromID(jsonl_path, idList):
    #id_list = []
    df_qrels = pd.DataFrame([], columns=['query-id','corpus-id','score']) # query-id	corpus-id	score
    print(df_qrels) #
    json_lines = [] # list(dict{'_id':int, 'text':[str], 'metadata':OrderedDict([])})
    qrel_lines = [] # 

    with jsonlines.open(jsonl_path) as f:
        for i, line in enumerate(f):
            if line['fever_id'] in idList:
                line_id = line['fever_id']
                list_wikiURLs = line["gold_evidence_set"]
                for wikiURL in list_wikiURLs:
                    line_title = wikiURL[0].get('title')
                    #line_score = 1
                    
                    qrel_colDict = {'query-id':line_id, 'corpus-id':line_title, 'score':1}
                    #qrel_column = pd.Series(qrel_colDict)
                    #qrel_column = pd.DataFrame([[line_id,line_title,1]], columns=['query-id','corpus-id','score'])
                    #print(qrel_column)
                    
                    df_qrels = df_qrels.append(pd.Series(qrel_colDict), ignore_index=True) #list_qrels.append(dict_qrel)
                    #print(f'[{i}] df_qrels len: {len(df_qrels)}')
                    #print(df_qrels.head())

        print(f'df_qrels() : len({len(df_qrels)})')
        print(df_qrels.head())

    return df_qrels

def modifyJsonl_feverText2colloquial(fever_queries_path, col_claims_dict, colIdx):
    colloquialClaimDicts = []
    with jsonlines.open(fever_queries_path) as queries_jsonl:
        for line in queries_jsonl.iter():
            if line["_id"] in col_claims_dict.keys():
                line["text"] = col_claims_dict[line["_id"]][colIdx]
            colloquialClaimDicts.append(line)
    return colloquialClaimDicts

def readQRELs_fromID(jsonl_path, idList):
    qrel_lines = [] # 

    print(f'read qrels file at jsonl_path={jsonl_path}')
    with jsonlines.open(jsonl_path) as f:
        for i, line in enumerate(f):
            dict_qrel = OrderedDict()
            
            list_wikiURLs = line["gold_evidence_set"]
            dict_qrel["query_id"] = line['fever_id']
            dict_qrel["score"] = 1
            for wikiURL in list_wikiURLs:
                #print(wikiURL) #title,evidence
                #print(*wikiURL) #dict(title,evidence)
                # print(type(*wikiURL))
                title = wikiURL[0].get('title')
                #title, _ = *wikiURL["title"], *wikiURL["evidence"]
                dict_qrel["corpus_id"] = title
                qrel_lines.append(dict_qrel) #list_qrels.append(dict_qrel)
    return qrel_lines
def readJsonl_fromID(jsonl_path, idList):
    json_lines = [] # list( dict{'_id':int, 'text':[str], 'metadata':OrderedDict([])} )
    #qrel_lines = [] # 

    print(f'read jsonl file at jsonl_path={jsonl_path}')
    with jsonlines.open(jsonl_path) as f:
        for i, line in enumerate(f):
            print(f'[{i}] idList: {id_list}') #dict{'colloquial_claims':['str'*3],'fever_claim':"str",'fevr_label':'SUPPORTS','evidences':[],'gold_evidence_set':[],'fever_id':int}
            # dict_keys - corpus:['_id', 'title', 'text', 'metadata'], queries:['_id', 'text', 'metadata']
            dict_line = dict() #OrderedDict()
            dict_metadata = OrderedDict()
            #dict_qrel = OrderedDict()

            #if line["metadata"][]
            print(f'[{i}] line: {line}') #keys: [ _id, text, metadata(verifiable, label, evidence) ]  # -evidence_example:[92206, 104971, 'Nikolaj_Coster-Waldau', 7]
            #line_id = line['_id']
            #print(f'idList: {len(idList), idList[0]}, type:{type(idList[0]), type(line_id)}') #list(int): 126502, 53037(int) #int,str

            if int(line['_id']) in idList:
                dict_metadata["verifiable"] = "VERIFIABLE"
                dict_metadata["label"] = "SUPPORTS" #line['fever_label'] 
                dict_metadata["evidence"] = [] # list(tuple[Annotation_id, Evidence_id, Wiki_URL(null), sentence_id(null)]) #나중에 채우기.일단 비워둠.

                dict_line["_id"] = line['_id'] #"ss", int(203870)
                dict_line["text"] = line['colloquial_claims'][0] #"", list(str*3), colloquial_text를 사용.
                dict_line["metadata"] = dict_metadata #OrderedDict( [(verificable),(label),(evidence,[])] )
                print(f'line[metadata] - {dict_line["metadata"], type(dict_line["metadata"])}')
                # for key,val in dict_line["metadata"].items():
                #     print(key, val)
                # print(f'- {dict_line["metadata"]["verifiable"], dict_line["metadata"]["evidence"]}')
                #print(f'dict_line elems: id:{dict_line["_id"]}')
                #print(f'dict_line elems: text:{dict_line["text"]}')
                #print(f'dict_line elems: metadata:{dict_line["metadata"]}')

                #print(dict_line) #OrderDict([ (_id, int), (text,[3_str]), (metadata, dict([])) ])
                json_lines.append(dict_line)

    return json_lines#json_lines, list_qrels

'''
Setting:
- 1) Argument setting
- 2) Paths settings

Preprocessing output: 
- 1) feverID_lists(list of int): qrel format query/corpus-id list for train/valid/test
- 2) qrel/*.tsv files(stored in 'datasets/colloquials/qrel/')
- 3) query/corpus.jsonl at 'datasets/colloquials/'. from qrel/*.tsv
'''

def main(args):
    ## Step 1: Arguments Setting ##

    ## Step 2: Path setting ##
    #from_dir = '/home/nlplab/hdd4/lss/dataset_ir/colloquial-claims/colloquial_data/'
    #colClaims_jsonl_dir = args.colloquialclaims_jsonl_dir #'/home/nlplab/hdd4/lss/dataset_ir/colloquial-claims/colloquial_data/'
    out_dir = args.colloquial_dataPath_dir[:-1] + '_' + args.colloquial_index + '/' #args.colloquial_dataPath_dir

    colloquial_filenames = ['colloquial_claims_train.jsonl','colloquial_claims_valid.jsonl','colloquial_claims_test.jsonl']

    ## 1. Get FEVER_id from jsonl(colClaims_train/valid/test) -> 각각 feverID_lists(list)
    feverID_lists, colloquialClaims_dictList = [], [] #list[], len:3, order:train/valid/test
    for filename in colloquial_filenames:
        data_path = os.path.join(args.colloquialclaims_jsonl_dir,filename)
        list_jsonlFile_id, dict_jsonlFile_colclaims = readJsonl_idList(data_path)

        print(f'id_list: len({len(list_jsonlFile_id)}),set:{len(set(list_jsonlFile_id))} - len_colDictList:{len(dict_jsonlFile_colclaims)}') #다 unique. 갯수대로 나옴.(126502, 8532, 8327)
        print(f' - fever sample(id:colClaim) : {list_jsonlFile_id[10]}:{dict_jsonlFile_colclaims[list_jsonlFile_id[10]]}')
        
        feverID_lists.append(list_jsonlFile_id)
        colloquialClaims_dictList.append(dict_jsonlFile_colclaims)
        #print(f'id_list: len({len(id_list)}), set:{len(set(id_list))}, sample:{id_list[10]})') #다 unique. 갯수대로 나옴.(126502, 8532, 8327)
    #idx_trainFEVER, idx_validFEVER, idx_testFEVER = feverID_lists #len: 126502 8532 8327

    ## 2. get qrels/ -> qrel 폴더 열어서 각각의 id 확인? 뭔진 모름. : 각각의 (query/corpus) id를 qrel/에 tsv 저장. 
    qrel_filenames = ['train.tsv', 'dev.tsv', 'test.tsv']
    for filename, qrel_outfile, idList in zip(colloquial_filenames, qrel_filenames, feverID_lists):
        colloquial_data_path = os.path.join(args.colloquialclaims_jsonl_dir,filename)
        if args.if_preprocess2_mk_qrel_tsv: # args: ifmake2(if_preprocess2_mk_qrel_tsv)
            curDF = getDF_qrels_fromID(colloquial_data_path, idList)
        
            outQREL_path = os.path.join(out_dir,'qrels/',qrel_outfile)
            curDF.to_csv(outQREL_path, index=False, sep='\t')


    ## 3. from feverID_lists, get jsonl_queries : read fever(corpus,queries) -> colloquial에서 각각의 jsonl_queries 리스트 받아와서 fever의 corpus/queries에 추가.
    ##       -> 대화형태의 corpus/queries.json을 colloquial/에 저장.
    ##      -> jsonl 형태에 corpus는 그대로, queries는 'text'만 'id'에 맞춰서 바꿔서.
    from_dir = os.path.join(os.getcwd(),"datasets/") #"datasets/fever/"
    fever_filenames = ['queries.jsonl', 'corpus.jsonl']
    colloquial_filename = 'queries.jsonl'

    jsonl_corpus, jsonl_queries = [], []

    # colQREL_dir = os.path.join(out_dir,"qrels/") #os.path.join(args.colloquial_dataPath_dir,"colloquial/qrels/")
    #feverJSONL_dir = args.fever_dataPath_dir #os.path.join(from_dir,"fever/")
    ## beir.data.fever의 qrels의 TDT.tsv에서 q/c_id 따다가 fever/queries.jsonl의 'text'만 colloquial로 바꿔서 colloquial에 넣어주기.

    fever_queries_path = os.path.join(args.fever_dataPath_dir,'queries.jsonl') #fever/queries.jsonl file path #path:'datasets/fever/queries.jsonl'
    coloquial_queries_path = os.path.join(out_dir,'queries.jsonl') #colloquial_dataPath_dir

    colClaimsDict_qrel = colloquialClaims_dictList[0] #train set
    colClaimsDict_qrel.update(colloquialClaims_dictList[1]) #valid set
    colClaimsDict_qrel.update(colloquialClaims_dictList[2]) #test set
    print(f'colClaimsDict_qrel - len({len(colClaimsDict_qrel)})')

    dictQueries = modifyJsonl_feverText2colloquial(fever_queries_path, colClaimsDict_qrel, int(args.colloquial_index)) #modifyJsonl_feverText2colloquial(fever_queries_path, colClaimsDict_qrel, 0)
    print(f'list of fever2Col dict_list(queries.jsonl): len({len(dictQueries)}), type({type(dictQueries)}),')
    #print(f' - sample[0]({dictQueries[0]},')
    #print(f' - sample[end]({dictQueries[-1]}')
    with open(coloquial_queries_path,'w') as write_file: #open(coloquial_queries_path, encoding='utf-8')
        for i, dictQuery in enumerate(dictQueries):
            json.dump(dictQuery,write_file)
            write_file.write('\n')
        #write_file.write(json.dumps(dictQueries)+"\n")
    
    fever_corpus_path = os.path.join(args.fever_dataPath_dir,'corpus.jsonl')
    coloquial_corpus_path = os.path.join(out_dir,'corpus.jsonl')
    feverCorpus_dicts = []
    with jsonlines.open(fever_corpus_path) as corpus_jsonl:
        for line in corpus_jsonl.iter():
            feverCorpus_dicts.append(line)
    print(f'list of fever2Col dict_list(corpus.jsonl): len({len(feverCorpus_dicts)}), type({type(feverCorpus_dicts)}),')
    with open(coloquial_corpus_path,'w') as write_file: #open(coloquial_queries_path, encoding='utf-8')
        for i, dict_line in enumerate(feverCorpus_dicts):
            json.dump(dict_line,write_file)
            write_file.write('\n')

    '''
    with jsonlines.open(fever_queries_path) as queries_jsonl:
        dictQueries = []
        for line in queries_jsonl.iter():
            dictQueries.append(line)
        
        for qrel_filename, idList, col_claims_dict in zip(qrel_filenames, feverID_lists, colloquialClaims_dictList): #str, str, list(int_id), dict
            colloquial_qrelID_path = os.path.join(args.colloquial_dataPath_dir,"qrels/",qrel_filename) #tsv:id -> jsonl[id]["colloquial_claims"][0]=str  #"datasets/colloquials/qrels/filename.jsonl"  
            
            dictQueries = modifyJsonl_feverText2colloquial(dictQueries, col_claims_dict, 0)

            list_jsonlFile, list_qrels = readJsonl_fromID(fever_data_path, idList, col_claims_dict) # #(idList:col_claims_dict) 읽어서 fever_data_path 수정.
            
            for i, idNum in enumerate(idList):
                #checkJsonl_fromIDlist(data_path,idList)
                if args.if_preprocess3_mk_col2fever_style_jsonl: # args: ifmake3(if_preprocess3_mk_col2fever_style_jsonl)
                    print(f'list_jsonlFile:{len(list_jsonlFile)}, list_qrels:{len(list_qrels)}') #list_jsonlFile:126502, list_qrels:192812
                    print(*list_jsonlFile[:5]) #
                    
                    jsonl_queries.append(*list_jsonlFile) #
        with open(coloquial_queries_path, encoding='utf-8') as write_file:


    for qrel_filename, idList, col_claims_dict in zip(qrel_filenames, feverID_lists, colloquialClaims_dictList): #str, str, list(int_id), dict
        fever_filename = fever_filenames[0] #str 'queries.jsonl'

        fever_data_path = os.path.join(args.fever_dataPath_dir,fever_filename) #fever/queries.jsonl file path #path:'datasets/fever/queries.jsonl'
        colloquial_qrelID_path = os.path.join(out_dir,"qrels/",qrel_filename) #tsv:id -> jsonl[id]["colloquial_claims"][0]=str  #"datasets/colloquials/qrels/filename.jsonl"  
        colloquial_data_path = os.path.join(args.colloquialclaims_jsonl_dir,colloquial_filename) #col_claim(text): original colloquial-claim jsonlfile #'/home/nlplab/hdd4/lss/dataset_ir/colloquial-claims/colloquial_data/'
        
        list_jsonlFile, list_qrels = readJsonl_fromID(fever_data_path, idList, col_claims_dict) # #(idList:col_claims_dict) 읽어서 fever_data_path 수정.
        
        for i, idNum in enumerate(idList):
            #checkJsonl_fromIDlist(data_path,idList)
            if args.if_preprocess3_mk_col2fever_style_jsonl: # args: ifmake3(if_preprocess3_mk_col2fever_style_jsonl)
                print(f'list_jsonlFile:{len(list_jsonlFile)}, list_qrels:{len(list_qrels)}') #list_jsonlFile:126502, list_qrels:192812
                print(*list_jsonlFile[:5]) #
                
                jsonl_queries.append(*list_jsonlFile) #
    
    print(len(jsonl_queries))
    
    '''


    ## 3. get qrels/
    '''
    jsonl_queries, jsonl_corpus 각각 받아서 다음으로 각각 저장.
    - datasets/fever_colloquial/qrels/에 (train/dev/test).tsv
    - datasets/fever_colloquial/에 (corpus,queries).jsonl

    이후에는 똑같이 실험.
    '''


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description='BEIR_demo: scifact, cos-sim')
    argParser = addArguments(argParser)
    # print('argParser type: ',type(argParser)) #<class 'argparse.ArgumentParser'>
    args = argParser.parse_args() #args 파라미터 객체 추가. 
    if args.run_preprocess_collquial2beir_pythonfile:
        main(args)

    print('preprocess_colloquial2beir.py Ended')