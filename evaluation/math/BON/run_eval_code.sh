#example

python eval.py --benchmark_type llama_data --bon_size 4 --input_data_path Example_Llama31_GSM8k_testset_bo256.jsonl  --reward_port 8080 --eval_type confidence --dataset_type gsm8k --prm_model_path Lux0926/ASPRM-M &
python eval.py --benchmark_type llama_data --bon_size 8 --input_data_path Example_Llama31_GSM8k_testset_bo256.jsonl  --reward_port 8080 --eval_type confidence --dataset_type gsm8k --prm_model_path Lux0926/ASPRM-M &
python eval.py --benchmark_type llama_data --bon_size 16 --input_data_path Example_Llama31_GSM8k_testset_bo256.jsonl  --reward_port 8080 --eval_type confidence --dataset_type gsm8k --prm_model_path Lux0926/ASPRM-M &
python eval.py --benchmark_type llama_data --bon_size 32 --input_data_path Example_Llama31_GSM8k_testset_bo256.jsonl  --reward_port 8080 --eval_type confidence --dataset_type gsm8k --prm_model_path Lux0926/ASPRM-M &
python eval.py --benchmark_type llama_data --bon_size 64 --input_data_path Example_Llama31_GSM8k_testset_bo256.jsonl  --reward_port 8080 --eval_type confidence --dataset_type gsm8k --prm_model_path Lux0926/ASPRM-M &

wait
