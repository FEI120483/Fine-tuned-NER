"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "schema_path": "ner_data/schema.json",
    "model_type":"bert", 
    "max_length": 20,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 64,
    "tuning_tactics":["p_tuning", "prefix_tuning", "lora_tuning", 'prompt_tuning'],
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"bert-base-chinese",
    "class_num" : 10,
    "seed": 987
}