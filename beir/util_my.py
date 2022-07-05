## Code by LSS(github: teddy309)
## date: 2022.07.03 ~ Current(on updating)

'''
Logger: tensorboard, wandb 중 선택
'''

#from beir import util, LoggingHandler

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


'''
DataLoader, Model import : (from beir github)
'''
###IR-Evaluation
from beir.retrieval.evaluation import EvaluateRetrieval
###Dense Retreival
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES #search 방식으로는 DRES,DRFS,BM25,SS 지원하는 듯. 일단. 
from beir.retrieval.search.dense import DenseRetrievalFaissSearch as DRFS
###Other Retrievals
from beir.retrieval.search.lexical import BM25Search 
from beir.retrieval.search.sparse import SparseSearch
###Embedding Models
# from beir.retrieval.models import SentenceBERT, DPR 
# from beir.retrieval.models import UseQA, TLDR, BinarySentenceBERT, SPLADE, SPARTA, UniCOIL
'''
- DPR: ["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-ctx_encoder-single-nq-base"] #"facebook/dpr-ctx_encoder-single-nq-base", "facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-reader-single-nq-base"
- SentenceBERT: "msmarco-distilbert-base-v3"
'''