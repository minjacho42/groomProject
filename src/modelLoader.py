import wandb
import torch
from transformers import(
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer
)

def loadModelForSequenceClassification(model_name, artifact_version, model_path = '/best_model_at_end/pytorch_model.bin'):
    wandb.login()
    run = wandb.init()
    artifact = run.use_artifact(artifact_version, type='model')
    artifact_dir = artifact.download()

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.load_state_dict(torch.load(artifact_dir+model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return (model, tokenizer)

def loadModelForCasualLM(model_name, artifact_version, model_path = '/best_model_at_end/pytorch_model.bin'):
    wandb.login()
    run = wandb.init()
    artifact = run.use_artifact(artifact_version, type='model')
    artifact_dir = artifact.download()

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.load_state_dict(torch.load(artifact_dir+model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                              pad_token='<pad>', mask_token='<mask>')
    return (model, tokenizer)