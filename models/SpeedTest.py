
import sys,os
import time





print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from RiceTransformer import RiceTransformer
sys.path.append('/home/junsheng/ViLT')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from ViST import ViST
from models.DNNF2 import DNNF2

# 选择要测试的设备，例如CPU或GPU


import torch
from torch import nn
class config:
    debug = False
    sensor_only = False
    crop_name = "total"

    
    exp_name = "vilt"
    seed = 101
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    train_batch_size = 32
    valid_batch_size = 4
    device = "cuda:0"
    n_fold = 5

    model_name = "sensorViLOnlyTransformerSS" #仅图片
    # model_name = "sensorOnlyViLTransformerSS"  #仅vilt传感器
    # model_name = "sensorViLTransformerSS"  #vilt图像+传感器
    # model_name = "DNNF1"  #DNNF1图像+传感器
    # model_name = "DNNF1PictureOnly"  #DNNF1图像
    # model_name = "DNNF1SensorOnly"  #DNNF1传感器

    # model_name = "DNNF2"  #DNNF2图像+传感器
    # model_name = "DNNF2PictureOnly"  #DNNF1图像
    # model_name = "DNNF2SensorOnly"  #DNNF1传感器
    # wandb 
    # wandb_name = "vilt|大豆|290图像加传感器"
    # wandb_name = "vilt|大豆|290仅传感器"
    wandb_name = "vilt|大豆|290仅图片"

    # wandb_name = "DNNF1|大豆|290图像加传感器"
    # wandb_name = "DNNF1|大豆|290仅图像"
    # wandb_name = "DNNF1|大豆|290仅传感器"
    
    # wandb_name = "DNNF2|大豆|290图像加传感器"
    # wandb_name = "DNNF2|大豆|290仅图像"
    # wandb_name = "DNNF2|大豆|290仅传感器"
    

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    img_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Sensor
    # senser_input_num = 11 # 翔冠的传感器参数
    senser_input_num = 19 # 天航的传感器参数
    
    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522 # vocabulary词汇数量
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768  # 嵌入向量大小
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-3 #0.0015#2e-3 #
    weight_decay = 1e-4 # 0.01 ->1e-4
    decay_power = 1
    max_epoch = 50
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads
    # T_max = 8000/train_batch_size*max_epoch 
    # T_max = 4632/train_batch_size*max_epoch # total 7237.5
    T_max = 2126/train_batch_size*max_epoch # soybean 3321.875

    # Downstream Setting
    get_recall_metric = False


    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = "weights/vilt_200k_mlm_itm.ckpt"
    # load_path = "save_model_dict.pt"
    num_workers = 1
    precision = 16

# config = vars(config)
# config = dict(config)
config
from RiceFusionMLP import RiceFusionMLP
from RiceFusionCNN import RiceFusionCNN
from RiceFusion import RiceFusion
from RiceTransformer import RiceTransformer

from CNNTransformer import CNNTransformer
from DNNF1 import *


# 测试数据
img = torch.randn([32, 3, 384, 384]).to(torch.float32).cuda()
sensor = torch.randn([32, 1, 19]).cuda()
batch = {"image":img,"sensor":sensor}


# 运行次数
times = 10



import torch.utils.benchmark as benchmark

def test(model,times):
    # with torch.no_grad():
    #     start_time = time.perf_counter()
    #     for i in range(times):
    #         model(batch)
    #     end_time = time.perf_counter()
    #     elapsed_time = end_time - start_time
    #     print(f"{model.__class__.__name__}代码运行时间: {elapsed_time:.2f}秒")
    with torch.no_grad():
        benchmark_result = benchmark.Timer(
            stmt='model(input_data)',
            setup='pass',
            globals={'model': model, 'input_data': batch}
        ).timeit(times)
        print("{}代码运行时间:{:.2f}ms".format(model.__class__.__name__,benchmark_result.mean * 1000))

def fetch_model(model_name):
    if model_name == "RiceTransformer":
        return RiceTransformer(sensor_nums=config.senser_input_num,config=config)
    elif model_name =="CNNTransformer":
        return CNNTransformer(sensor_nums=config.senser_input_num,config=config)
    elif model_name == "ViST":
        return ViST(sensor_class_n=config.senser_input_num,output_class_n = 1,config=config)
    elif model_name =="RiceFusion":
        return RiceFusion(sensor_nums=config.senser_input_num,config=config)
    elif model_name == "RiceFusionCNN":
        return RiceFusionCNN(config=config)
    elif model_name == "DNNF1":
        return DNNF1(sensor_nums=config.senser_input_num,config=config)
    elif model_name == "DNNF2":
        return DNNF2(sensor_nums=config.senser_input_num,config=config)

model_list = ["RiceTransformer","CNNTransformer","ViST","RiceFusion","RiceFusionCNN","DNNF1","DNNF2"]
# model_list = ["ViST",]

for name in model_list:
    print(f"当前times:{times}")
    model = fetch_model(name)
    model.to(config.device)
    test(model,times)



