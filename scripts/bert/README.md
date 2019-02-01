# SQuAD 1.1 Experiments

To reproduce fine-tuenn SQuAD 1.1 experiments, one can run the following:

```bash
python finetune_squad.py --optimizer adam --gpu
```

After training for two epochs, one can get the following results.

```bash
{'exact_match': 81.21097445600756, 'f1': 88.4551346176558}
```

# Latency

To report inference latency on SQuAD 1.1 test data, run the following command. Note that it is optional to use `fp16` or `hybridize`

```bash
export CUDA_VISIBLE_DEVICES=0
python finetune_squad.py \
--optimizer adam \
--gpu \
--only_predict \
--model_parameters output_dir/net_parameters \
--test_batch_size 8 \
--max_examples 1000 \
--context_factor 1 \
--latency_report_file latency_report \
--fp16 \
--hybridize
```  