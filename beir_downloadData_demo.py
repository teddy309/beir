## Code by LSS(github: teddy309)
## date: 2022.06.06 ~ Current(on updating)

'''
settings: argparse
'''
import argparse

def addArguments(parser):
    parser.add_argument('--dataset_name', default='fever',
                        help='dataset name', choices=['scifact', 'ms_marco', 'fever'])
    parser.add_argument('--dataset_path', default=os.path.join(os.getcwd(), "datasets"),
                        help='dataset path', choices=['/../datasets', '/..','/datasets'])
    parser.add_argument('--out_dir_name', default= '/runs',
                        help='logging file(.log) directory name')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--cuda_gpuNum', default=0,
                        help='gpu:number for cuda(default=0, 0 if disable_cuda==True)', choices=[0,1])
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

import wandb

def initWandB(): 
    #pip install wandb
    #wandb login
    wandb.init(project="test-project-beir_ver1.0", entity="gachon")
    wandb.config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128
    }

    #logging: wandb.log({"loss": loss})
    #Optional: wandb.watch(model)
    return False #for logger tensorboard

#아래 클래스 여기선 안쓸듯??
class beir_logger_Class(object):

    def __init__(self, *args, **kwargs):
        #self.args = kwargs['args']
        #self.model = kwargs['model'].to(self.args.device)
        #self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        #### Just some code to print debug information to stdout
        '''logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])'''
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
    
    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
    
    def evaluate(self, ):
        # Evaluation
        results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            results = trainer.evaluate(eval_senteval_transfer=True)

            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key, value in sorted(results.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

        return results

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
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES #search 방식으로는 DRES,DRFS,BM25,SS 지원하는 듯. 일단. 

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

    print('argParser type: ',type(argParser))
    args = argParser.parse_args() #args 파라미터 객체 추가. 

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
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
        logging.basicConfig(filename=os.path.join(summaryWriter.log_dir, 'download_dataset.log'), level=logging.DEBUG)
    
    ## Step 2: Download&Set Dataset&Model from BEIR ##
    '''
    BEIR dataset: fever
        - download dataset(scifact) in dataPath(args.dataset_path)
        - if model runs OK, store results() at out_dir(runs/args.out_dir_name)
    
    BEIR model: pretrained-SentenceBERT(msmarco-distilbert_v3)
        - qrel: ???
        - input: corpus,queries
        - output: results(corpus) for queries
    '''
    #import pathlib, os
    #from beir import util

    dataset = args.dataset_name #"scifact"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    #homePth = os.getcwd()+'/..' #hdd4/lss
    data_dir = args.dataset_path #default: os.path.join(os.getcwd(), "datasets") #이건뭐지?homePth로 원하는 절대경로에서 끌어올때 쓰는듯.: os.path.join(homePth, "/dataset_ir/dataset_beir") 
    print(f'dataset paths : (url,out_dir)')
    print(f' - url:{url}')
    print(f' - out_dir:{data_dir}')
    data_path = util.download_and_unzip(url, data_dir)
    print("Dataset downloaded here: {}".format(data_path))

    os.system(f'ls {data_dir}/{dataset}/')

    data_path = data_dir+"/"+dataset #"/scifact" #os.path.join(data_dir,dataset) #update 'data_path'
    logging.info(f"Download Dataset Finished: {dataset} dataset in path {data_path}.")

    return data_path

if __name__ == "__main__":
    download_path = main()
    print(f'data downloaded at {download_path}')

    print('beir_downloadData_demo.py Ended')





