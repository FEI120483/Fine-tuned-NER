"""
主函数
"""

from tqdm import trange, tqdm
import torch
import os
import numpy as np
import logging
import torch.nn as nn
import time
from config import Config
from model import Model, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model,LoraConfig, TaskType, PrefixTuningConfig
import pandas as pd

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
定义用于记录结果函数
"""
results = pd.DataFrame(columns=['Tuning Tactics',"Epoch", "Train Loss", 'Micro_F1', "Macro_f1", 'Train Time'])

def record_results(tuning_tactics, epoch, train_loss, Micro_f1, Macro_f1, train_time):
    results.loc[len(results)] = [tuning_tactics, epoch, np.mean(train_loss), Micro_f1, Macro_f1, train_time]
    
"""
模型训练主程序
"""
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
        
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    basic_model = Model
    for i in trange(len(config["tuning_tactics"])):
        tuning_tactics = config["tuning_tactics"][i]
        print(f"当前调参策略为：{tuning_tactics}")
        if tuning_tactics == "lora_tuning":
            peft_config = LoraConfig(
                task_type = TaskType.TOKEN_CLS,
                inference_mode = False,
                r = 8,
                lora_alpha = 32,
                lora_dropout = 0.1,
                target_modules = ["query", "key", "value"]
            )
        elif tuning_tactics == "prefix_tuning":
            peft_config = PrefixTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
        
        ft_model = get_peft_model(basic_model, peft_config)
        
        # 标识是否使用gpu
        cuda_flag = torch.cuda.is_available()
        if cuda_flag:
            logger.info("gpu可以使用,迁移模型至gpu")
            ft_model = ft_model.cuda()
        #加载优化器
        optimizer = choose_optimizer(config, ft_model)
        #加载效果测试类
        evaluator = Evaluator(config, ft_model, logger)
        #训练
        start_time = time.time()
        for epoch in trange(config["epoch"]):
            epoch += 1
            ft_model.train()
            logger.info("epoch %d begin" % epoch)
            train_loss = []

            for index, (input_ids, attention_mask, labels) in tqdm(enumerate(train_data)):
                optimizer.zero_grad()
                if cuda_flag:
                    input_ids = input_ids.cuda()
                    labels = labels.cuda()

                outputs = ft_model(input_ids, attention_mask = attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                if index % int(len(train_data) / 2) == 0:
                    logger.info("batch loss %f" % loss)
            logger.info("epoch average loss: %f" % np.mean(train_loss))
            evaluator.eval(epoch)
            train_time = time.time() - start_time
            macro_f1, micro_f1 = evaluator.show_stats()
            record_results(tuning_tactics, epoch, train_loss, micro_f1, macro_f1, train_time)
        model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
        results.to_csv(os.path.join(config["model_path"], "training_results.csv"), index=False)
        torch.save(ft_model.state_dict(), model_path)
    return ft_model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)