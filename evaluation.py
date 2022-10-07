import argparse
import os, sys
import json
import csv
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler
import sentence_transformers as sent_trans
import transformers
from transformers import set_seed
import accelerate
from accelerate import Accelerator
from dataset import SimpleDataset, padding_util
from model import build_encoder, DualEncoderModel
from utils import evaluate, parse_args
import warnings

def evaluation(label_list, articles_input):
	args = parse_args()
	warnings.filterwarnings("ignore")

	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
	args = parse_args()
	distributed_args = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
	accelerator = Accelerator(kwargs_handlers=[distributed_args])
	device = accelerator.device

	# accelerator.is_local_main_process is only True for one process per machine.
	if accelerator.is_local_main_process:
		transformers.utils.logging.set_verbosity_info()
	else:
		transformers.utils.logging.set_verbosity_error()

	# If passed along, set the training seed now.
	if args.seed is not None:
		set_seed(args.seed)

	# Load pretrained model and tokenizer
	if args.model_name_or_path == 'bert-base-uncased' or args.model_name_or_path == 'sentence-transformers/paraphrase-mpnet-base-v2':
		label_encoder = build_encoder(
			args.model_name_or_path,
			args.max_label_length,
			args.pooling_mode,
			args.proj_emb_dim,
		)
	else:
		label_encoder = sent_trans.SentenceTransformer(args.model_name_or_path)

	tokenizer = label_encoder._first_module().tokenizer

	instance_encoder = label_encoder

	model = DualEncoderModel(
		label_encoder,
		instance_encoder,
	)
	model = model.to(device)

	# label input
	label_data = SimpleDataset(label_list, transform=tokenizer.encode)
	# label dataloader for searching
	sampler = SequentialSampler(label_data)
	label_padding_func = lambda x: padding_util(x, tokenizer.pad_token_id, 64)
	label_dataloader = DataLoader(label_data, sampler=sampler, batch_size=16, collate_fn=label_padding_func)

	# articles data
	all_instances = []
	all_titles = []
	for inst in articles_input:
		all_titles.append(inst['title'])
		all_instances.append(inst['title'] + '\t' + inst['content'])
	
	simple_transform = lambda x: tokenizer.encode(x, max_length=288, truncation=True)
	instances_data = SimpleDataset(all_instances, transform=simple_transform)
	instances_num = len(instances_data)

	sampler = SequentialSampler(instances_data)
	sent_padding_func = lambda x: padding_util(x, tokenizer.pad_token_id, 288)
	instance_dataloader = DataLoader(instances_data, sampler=sampler, batch_size=128, collate_fn=sent_padding_func)

	# Prepare everything with our `accelerator`.
	model, label_dataloader, instance_dataloader = accelerator.prepare(model, label_dataloader, instance_dataloader)

	result = evaluate(args, 0, accelerator.unwrap_model(model), label_dataloader, label_list, 
						instance_dataloader, instances_num, all_titles, accelerator)
	return result