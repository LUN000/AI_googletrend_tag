import numpy as np
import math
import faiss
import os
import torch
from tqdm.auto import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval
import argparse
from transformers import (
	AdamW,
	SchedulerType,
	get_scheduler,
	set_seed,
)

def parse_args():
	parser = argparse.ArgumentParser(description="Pretrain two-tower Transformer models with ICT")
	# Data
	parser.add_argument(
		"--corpus-pkl-path",
		type=str,
		help="a processed pickle file that contains title_list and block_list",
	)
	parser.add_argument(
		"--mode",
		type=str, default="evaluate",
		help="the mode of the training procedure",
	)
	
	parser.add_argument(
		"--log",
		type=str, default="test",
		help="log file",
	)
	parser.add_argument(
		"--ratio",
		type=float, default=0.01,
		help="Sampling ratio",
	)
	# Model
	parser.add_argument(
		"--model-name-or-path",
		type=str, default="LF-Amazon-1M",
		help="Path to pretrained model or model identifier from huggingface.co/models.",
	)
	parser.add_argument(
		"--max-label-length",
		type=int, default=64,
		help="maximum label length for pre-training (default: 64)",
	)
	parser.add_argument(
		"--max-inst-length",
		type=int, default=288,
		help="maximum block (i.e., title + text) length for pre-training (default: 288)",
	)
	parser.add_argument(
		"--pooling-mode",
		type=str, default="cls",
		help="Can be a string: mean/max/cls.",
	)
	parser.add_argument(
		"--proj-emb-dim",
		type=int, default=512,
		help="embedding size of the projection layer in two-tower models",
	)
	# Optimizer
	parser.add_argument(
		"--per-device-train-batch-size",
		type=int, default=16,
		help="training batch size per GPU device (default: 8)",
	)
	parser.add_argument(
		"--learning-rate",
		type=float, default=3e-5,
		help="Initial learning rate (after the potential warmup period) to use.",
	)
	parser.add_argument(
		"--weight-decay",
		type=float, default=0,
		help="Weight decay to use.",
	)
	parser.add_argument(
		"--max-train-steps",
		type=int, default=50000,
		help="Total number of training steps to perform (default: 10,000)",
	)
	parser.add_argument(
		"--gradient-accumulation-steps",
		type=int, default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument(
		"--lr-scheduler-type",
		type=SchedulerType,
		default="linear",
		help="The scheduler type to use.",
		choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
	)
	parser.add_argument(
		"--num-warmup-steps",
		type=int, default=2000,
		help="Number of steps for the warmup in the lr scheduler (default: 1,000)",
	)
	parser.add_argument(
		"--logging-steps",
		type=int, default=100,
		help="Number of steps for the logging information (default 100)",
	)
	parser.add_argument(
		"--eval-steps",
		type=int, default=1000,
		help="Number of steps for evaluation (default 100)",
	)
	parser.add_argument(
		"--saving-steps",
		type=int, default=2000,
		help="Number of steps for the saving checkpoint (default 1000)",
	)
	# Output
	parser.add_argument(
		"--output-dir",
		type=str, default='',
		help="Where to store the final model.",
	)
	parser.add_argument(
		"--seed",
		type=int, default=None,
		help="A seed for reproducible training.",
	)
	args = parser.parse_args(args=[])
	# sanity check
	args.output_dir = os.path.join(os.path.abspath(os.getcwd()), args.output_dir)
	if args.output_dir is not None:
		os.makedirs(args.output_dir, exist_ok=True)
	return args

def perform_eval(model, label_dataloader, label_list, instance_dataloader, instances_num, all_titles, accelerator):
	label_bert = model.label_encoder
	inst_bert = model.instance_encoder
	torch.cuda.empty_cache()
	label_bert, inst_bert = accelerator.prepare(label_bert, inst_bert)
	# label embeddings
	label_embeds = np.zeros((len(label_dataloader)*16*8, 512)).astype('float32')
	count = 0
	label_bert.eval()
	with torch.no_grad():
		for i, batch in enumerate(tqdm(label_dataloader, desc='Embedding Labels', disable=not accelerator.is_local_main_process)):
			batch_att_mask = ~(batch.eq(0))
			feature = {'input_ids': batch, 'attention_mask': batch_att_mask}
			embed = label_bert(feature)["sentence_embedding"]
			output = accelerator.gather(embed)
			output = output.data.cpu().numpy().astype('float32')
			num_label = output.shape[0]
			label_embeds[count:count + num_label, :] = output
			count += num_label
	label_num = len(label_list)
	label_embeds = label_embeds[:label_num]
	label_embeds = label_embeds.astype('float32')
	

	# instance embeddings
	inst_embeds = np.zeros((len(instance_dataloader)*128*8, 512)).astype('float32')
	count = 0
	inst_bert.eval()
	with torch.no_grad():
		count = 0
		for i, batch in enumerate(tqdm(instance_dataloader, desc='Embedding Instances', disable=not accelerator.is_local_main_process)):
			batch_att_mask = ~(batch.eq(0))
			feature = {'input_ids': batch, 'attention_mask': batch_att_mask}
			embed = inst_bert(feature)["sentence_embedding"]
			output = accelerator.gather(embed)
			output = output.data.cpu().numpy().astype('float32')
			num_inst = output.shape[0]
			inst_embeds[count:count+num_inst, :] = output
			count += num_inst
	accelerator.print("embedding")
	inst_embeds = inst_embeds.astype('float32')
	inst_embeds = inst_embeds[:instances_num]
	test_inst_embeds = inst_embeds[-len(all_titles):]

	
	accelerator.print("Finish embedding")
	D, I = get_knn(test_inst_embeds, label_embeds, accelerator, bsz=64)
	label_bert.train()
	inst_bert.train()
	del label_embeds

	return D, I, inst_embeds

def get_knn(inst_embeddings, label_embeddings, accelerator, top_k=100, bsz=65536):
	accelerator.print("FAISS")
	# logging.info("FAISS indexer building")
	res = faiss.StandardGpuResources()
	flat_config = faiss.GpuIndexFlatConfig()
	flat_config.useFloat16 = False
	flat_config.device = accelerator.local_process_index
	indexer = faiss.GpuIndexFlatIP(res, inst_embeddings.shape[1], flat_config)
	indexer.add(label_embeddings)
	# logging.info("FAISS indexer searching")
	num_inst = inst_embeddings.shape[0]
	nr_batch = int(math.ceil(num_inst / bsz))
	D_list, I_list = [], []
	accelerator.print("index")
	for bidx in tqdm(range(nr_batch)):
		sidx = bidx * bsz
		eidx = min((bidx + 1) * bsz, num_inst)
		D, I = indexer.search(inst_embeddings[sidx:eidx], top_k)
		D_list.append(D)
		I_list.append(I)
	D = np.concatenate(D_list)
	I = np.concatenate(I_list)
	return D, I


def evaluate(args, step, model, label_dataloader, label_list, 
						instance_dataloader, instances_num, all_titles, accelerator):

	D, I, inst_embeds = perform_eval(model, label_dataloader, label_list, instance_dataloader, instances_num, all_titles, accelerator)
	
	num_inst = len(all_titles)
	results = {pid: {} for pid in all_titles}
	accelerator.print("Results")
	for row_id in range(num_inst):
		inst_id = all_titles[row_id]
		for col_id, score in zip(I[row_id], D[row_id]):
			lid = label_list[col_id]
			results[inst_id][lid] = float(score)
	return results