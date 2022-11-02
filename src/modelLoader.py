import wandb
import torch
import gdown
from transformers import(
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer
)

def loadModelForSequenceClassification(model_name, artifact_version, model_path = '/best_model_at_end/pytorch_model.bin'):
    run = wandb.init()
    artifact = run.use_artifact(artifact_version, type='model')
    artifact_dir = artifact.download()

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.load_state_dict(torch.load(artifact_dir+model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return (model, tokenizer)

def gdLoadModelForSequenceClassification(model_name, artifact_version = '1IBoQCiD9cW9D7AmcE3BbLDG6TRm9zA02'):
    url = 'https://drive.google.com/uc?id=' + artifact_version
    artifact_dir = 'modelSequenceClassification.bin'

    gdown.download(url, artifact_dir, quiet=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.load_state_dict(torch.load(artifact_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return (model, tokenizer)

def loadModelForCausalLM(model_name, artifact_version, model_path = '/best_model_at_end/pytorch_model.bin'):
    run = wandb.init()
    artifact = run.use_artifact(artifact_version, type='model')
    artifact_dir = artifact.download()

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.load_state_dict(torch.load(artifact_dir+model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                              pad_token='<pad>', mask_token='<mask>')
    return (model, tokenizer)

def gdLoadModelForCausalLM(model_name, artifact_version = '1_PJoPgx0gsE2HrXdXmcW-uhPh5R3CgxC'):
    url = 'https://drive.google.com/uc?id=' + artifact_version
    artifact_dir = 'modelCausalLM.bin'

    gdown.download(url, artifact_dir, quiet=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.load_state_dict(torch.load(artifact_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                              pad_token='<pad>', mask_token='<mask>')
    return (model, tokenizer)