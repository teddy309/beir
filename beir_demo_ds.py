## Code by LSS(github: teddy309)
## date: 2022.07.04 ~ Current(복사. deepspeed 달아서 돌리는중.)

## TODO: deepspeed 달아서 DPR 돌아가게 고치기. 

'''
settings: argparse
'''
import argparse

def addArguments(parser):
    parser.add_argument('--dataset_name', default='scifact',
                        help='dataset name', choices=['scifact', 'ms_marco', 'fever','colloquial','colloquial_0','colloquial_1','colloquial_2'])
    parser.add_argument('--dataset_path', default=os.path.join(os.getcwd(), "datasets"),
                        help='dataset path', choices=['/../datasets', '/..','/datasets'])
    parser.add_argument('--out_dir_name', default= '/runs',
                        help='logging file(.log) directory name')
    parser.add_argument('--search_method', default='DRES',
                        help='IR method available with beir', choices=['DRES', 'DRFS', 'BM25Search', 'SparseSearch'])
    parser.add_argument('--model_name', default='SentenceBERT',
                        help='embedding(context/query) model name in beir', choices=['SentenceBERT', 'DPR', 'UseQA','TLDR','SPLADE','SPARTA','BinarySentenceBERT','UniCOIL'])
    parser.add_argument('--model_batch_size', default=16,
                        help='select embedding-model batch size (default=16)', type=int, choices=[0,1,5,10,16,128])
    parser.add_argument('--view_dataset_samples', default=False,
                        help='print first each 10 data samples at main (default false)',type=bool)
    parser.add_argument('--view_IRresult_samples', default=True,
                        help='print first each 10 data samples at main (default false)',type=bool)
    #parser.add_argument('--disable_cuda', action='store_true',
    #                    help='Disable CUDA')
    parser.add_argument('--disable_cuda', default=True,
                        help='Disable CUDA (default True)', type=bool)
    parser.add_argument('--cuda_gpuNum', default='0',
                        help='gpu:number for cuda(default=0, 0 if disable_cuda==True)', choices=['0','1'])
    #parser.add_argument('--disable_wandb', action='store_true',
    #                    help='Disable WandB logging (default true)')
    parser.add_argument('-wb','--use_wandb', default=False,
                        help='Disable WandB logging (default false)',type=bool)
    return parser


'''
Logger: tensorboard, wandb 중 선택
'''
from beir import util, LoggingHandler

import logging
import pathlib, os

from torch.utils.tensorboard import SummaryWriter

from beir.util_my import initWandB, beir_logger_Class


#### Just some code to print debug information to stdout
#logger = logging.getLogger(__name__) #if not wandb, use tensorboard with logger
#os.makedirs(out_dir, exist_ok=True)

#### /print debug information to stdout

'''
DataLoader, Model import : (from beir github)
'''
###Data Loading
from beir.datasets.data_loader import GenericDataLoader

###Dense Retreival
from beir.util_my import EvaluateRetrieval
from beir.retrieval import models #from beir.util_my import SentenceBERT, DPR
from beir.util_my import DRES, DRFS #, BM25Search, SparseSearch

#### Dense Retrieval using SBERT (Sentence-BERT) ####
#### Provide any pretrained sentence-transformers model
#### The model was fine-tuned using cosine-similarity.
#### Complete list - https://www.sbert.net/docs/pretrained_models.html


'''
Main : 
    1. Set Argument values for params setting (+logging tools)
    2. Get Dataset&Models from BEIR
    3. Retriever Model & Evaluation
'''
import torch
import torch.backends.cudnn as cudnn

def main():
    ## Step 1: Arguments Setting ##
    argParser = argparse.ArgumentParser(description='BEIR_demo: scifact, cos-sim')
    argParser = addArguments(argParser)

    #print('argParser type: ',type(argParser)) #<class 'argparse.ArgumentParser'>
    args = argParser.parse_args() #args 파라미터 객체 추가. 

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:'+args.cuda_gpuNum)
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    
    ##logging Tool: wandb / logging(tensorboard)
    if args.use_wandb:
        #use_logger = initWandB() #init, config(lr,epoch,bs)
        initWandB() #init, config(lr,epoch,bs)
    else:
        logger = logging.getLogger(__name__)
        #os.makedirs(args.out_dir_name, exist_ok=True)

        summaryWriter = SummaryWriter()
        logging.basicConfig(filename=os.path.join(summaryWriter.log_dir, 'training.log'), level=logging.DEBUG)
    
    ## Step 2: Download&Set Dataset&Model from BEIR ##
    '''
    BEIR dataset: [sci-fact, fever, colloquial]
        - download dataset(scifact) in dataPath(args.dataset_path)
        - if model runs OK, store results() at out_dir(runs/args.out_dir_name)
    
    BEIR model: pretrained-SentenceBERT(msmarco-distilbert_v3), 
                DPR(["facebook/dpr-question_encoder-single-nq-base","...-ctx_encoder-..."])
        - qrel: ???
        - input: corpus,queries
        - output: results(corpus) for queries
    '''
    #import pathlib, os
    #from beir import util

    dataset = args.dataset_name
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    homePth = os.getcwd()+'/..' #hdd4/lss
    data_dir = args.dataset_path #os.path.join(os.getcwd(), "datasets")
    print(f'dataset paths : (url,out_dir)')
    print(f' - url:{url}')
    print(f' - out_dir:{data_dir}')
    data_path = util.download_and_unzip(url, data_dir)
    print("Dataset downloaded here: {}".format(data_path))

    data_path = data_dir+"/"+dataset #/scifact"
    os.system(f'ls {data_path}/')

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test") # or split = "train" or "dev"
    print(f'outdir_name(runs/.log file) : {os.getcwd()+args.out_dir_name}')
    os.makedirs(os.getcwd()+args.out_dir_name, exist_ok=True)

    logging.info(f"CUDA info: disable_cuda({args.disable_cuda}), cuda_availiable({torch.cuda.is_available()}), cuda_device({args.device}).")
    logging.info(f"Start BEIR.DRES model IR start with {dataset} dataset.")
    logging.info(f"Inference with gpu: {args.device}. (batch-size:{args.model_batch_size})")

    ## Step 3: Retriever Model & Evaluation ##
    ## model: (DRES(SBERT), DPR)
    # model = DRES(models.SentenceBERT("msmarco-distilbert-base-v3"), batch_size=128) #여기서 Batches는 뭐지? (각각 3,41인데...)
    if args.search_method=="DRES" and args.model_name=="SentenceBERT":
        model = DRES(models.SentenceBERT("msmarco-distilbert-base-v3"), batch_size=128)
    elif args.search_method=="DRES" and args.model_name=="DPR":
        model = DRES(models.DPR(["facebook/dpr-question_encoder-single-nq-base","facebook/dpr-ctx_encoder-single-nq-base"]), batch_size=args.model_batch_size) #DPR,DRES, bs=1
    else:
        model = DRFS(models.DPR(["facebook/dpr-question_encoder-single-nq-base","facebook/dpr-ctx_encoder-single-nq-base"]), batch_size=128) #DPR,DRES
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    print(f'Ret. Results: ') # Ret(C,q) -> R 예시 프린트
    print(f'    - type      : C_{type(corpus)},Q_{type(queries)} -> {type(results)}') #dict,dict,dict
    print(f'    - len       : C_{len(corpus)},Q_{len(queries)} -> {len(results)}') #5183, 300, 300
    print(f'    - Ex.(corpus)   : first 10')
    c_keys = list(corpus.keys())[:10]
    q_keys = list(queries.keys())[:10]
    R_keys = list(results.keys())[:10]
    if args.view_dataset_samples:
        for i,key in enumerate(c_keys):
            print(i, corpus[key]["title"]) # 3 The DNA Methylome of Human Peripheral Blood Mononuclear Cells
        for i,key in enumerate(q_keys):
            print(i, queries[key]) # 3 5% of perinatal mortality is due to low birth weight.
        print(type(R_keys[0]), len(R_keys[0]), R_keys) #str, 1, ['1', '3', '5', '13', '36', '42', '48', '49', '50', '51'](test.tsv 그대로 나옴. 앞부터)
        #print(results(R_keys[0])) #
        #print(max(results(R_keys[0]).values())) # 1 1
        #print(R_keys[0]['18670'], R_keys[0]['152245'], R_keys[0]['313394']) #Error
        for i, key in enumerate(R_keys):
            #print(type(results[key]), len(results[key])) #dict, 1001(왜 1001이지...? corpus에는 더 많이 있는데....)
            #print(type(key), key) #str, 1
            #print(type(results[key]['10743131']), results[key]['38528892']) # float, 0.1957xxxx...
            # print(results[key]) #ex- {'38528892': 0.19577528536319733, '10743131': 0.1821703463792801, '17967608': 0.17266175150871277}

            resultIR_forKey = results[key] #dict{}
            print(f'[{i}] qKey_{key} -> R: {max(resultIR_forKey, key=resultIR_forKey.get)} ') #q:{queries[q_keys[i]]}    

    for i, (q_key, R_key) in enumerate(zip(q_keys, R_keys)):
        resultIR_forKey = results[R_key] #dict{key:dic(corpusKey:score)}
        maxIR_key = max(resultIR_forKey, key=resultIR_forKey.get)
        maxIR_context = corpus[maxIR_key]

        #IRkey_maxN = max(resultIR_forKey, key=resultIR_forKey.get, n=5)
        #print(f'len:{len(IRkey_maxN)}, max_values:{IRkey_maxN}')
        #maxIR_key = IRkey_maxN[0]
        #maxIR_context = corpus[maxIR_key]

        logging.debug(f"[{i}] query:{queries[q_key]}")
        #logging.debug(f"[{i}] top1_contextTitle:{maxIR_context["title"]}")
        if args.view_IRresult_samples:
            print(f'[{i}] q_key: {q_key}, R_keys:{R_key, maxIR_key, resultIR_forKey[maxIR_key]}')
            print(f'- query: {queries[q_key]}')
            print(f'- IR corpus title: {maxIR_context["title"]}')

    #for i, c_title in enumerate(corpus[:10]["title"]):
    #    print(f'[{i}]th context: {corpus[c_title]}')
    #print(f'    - Example(query-result)   : {queries} -> {results}')


    ''' 
    IR_evaluation : 
    ''' 
    # Evaluation (copied from SimCSE. 아직 안고침. beir가 나을듯?)
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values) #logger남김. k_values단위로 qrel 내의 results 평가.
    
    logging.info("Retriever-Evaluation phase has finished.")

    return results

if __name__ == "__main__":
    main()

    print('beir_demo.py Ended')





