"""
模型评估
"""

import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果:" % epoch)
        self.stats_dict = {"NS": defaultdict(int),
                           "NR": defaultdict(int),
                           "NT": defaultdict(int),}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, attention_mask, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_id, attention_mask = attention_mask) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results.logits, sentences)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["NS", "NR", "NT"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        F1_scores = []
        for key in ["NS", "NR", "NT"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体,准确率:%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        macro_f1 = np.mean(F1_scores)
        self.logger.info("Macro-F1: %f" % macro_f1)
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["NS", "NR", "NT"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["NS", "NR", "NT"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["NS", "NR", "NT"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return macro_f1, micro_f1
    '''
    {
        "B-ns": 0,
        "M-ns": 1,
        "E-ns": 2,
        "B-nr": 3,
        "M-nr": 4,
        "E-nr": 5,
        "B-nt": 6,
        "M-nt": 7,
        "E-nt": 8,
        "o": 9
    }
    '''
    
    def decode(self, sentence, labels):
        sentence = '$' + sentence
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)

        for location in re.finditer("(012+)", labels):
            s, e = location.span()
            results["NS"].append(sentence[s:e])
        for location in re.finditer("(345+)", labels):
            s, e = location.span()
            results["NR"].append(sentence[s:e])
        for location in re.finditer("(678+)", labels):
            s, e = location.span()
            results["NT"].append(sentence[s:e])
        return results
    

if __name__ == "__main__":
    from config import Config
    from transformers import AutoModelForTokenClassification
    import logging
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    evaluator = Evaluator(Config, AutoModelForTokenClassification.from_pretrained(Config["pretrain_model_path"]), logger)