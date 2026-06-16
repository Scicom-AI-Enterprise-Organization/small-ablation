################################################
# Single Turn 
################################################
python benchmark_redteam.py \
--dataset_path RedTeam-LLM-SingleTurn-Customer-Support \
--target_api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
--api_key API_KEY \
--concurrency 10 \
--target_model qwen/qwen3.6-27b \
--single_turn

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-SingleTurn-Customer-Support \
--target_api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
--api_key API_KEY \
--concurrency 10 \
--target_model Qwen/Qwen3.6-35B-A3B \
--single_turn

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-SingleTurn-Customer-Support \
--target_api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
--api_key API_KEY \
--concurrency 10 \
--target_model Qwen/Qwen3.5-122B-A10B \
--single_turn

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-SingleTurn-Customer-Support \
--target_api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
--api_key API_KEY \
--concurrency 10 \
--target_model mistralai/Mistral-Small-4-119B-2603 \
--single_turn


python benchmark_redteam.py \
--dataset_path RedTeam-LLM-SingleTurn-Customer-Support \
--target_api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
--judge_api_url https://tm-vm1-llm-1.cae.scicom.dev/v1/chat/completions \
--api_key API_KEY \
--concurrency 10 \
--target_model MiniMaxAI/MiniMax-M2.7 \
--single_turn

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-SingleTurn-Customer-Support \
--target_api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
--api_key API_KEY \
--concurrency 10 \
--target_model google/gemma-4-31b-it \
--single_turn

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-SingleTurn-Customer-Support \
--target_api_url https://vp67xvkjckxpgu-8000.proxy.runpod.net/v1/chat/completions \
--judge_api_url https://serverlessgpu.aies.scicom.dev/endpoint-rq/v1/chat/completions \
--api_key API_KEY \
--concurrency 50 \
--target_model zai-org/GLM-4.7-Flash \
--single_turn

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-SingleTurn-Customer-Support \
--target_api_url https://qabv3qjna1b4v2-8000.proxy.runpod.net/v1/chat/completions \
--judge_api_url https://serverlessgpu.aies.scicom.dev/endpoint-rq/v1/chat/completions \
--api_key API_KEY \
--concurrency 50 \
--target_model Qwen/Qwen3.5-397B-A17B \
--single_turn

################################################
# Multi Turn
################################################

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-MultiTurn-Customer-Support \
--target_api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
--judge_api_url https://tm-vm1-llm-1.cae.scicom.dev/v1/chat/completions \
--api_key API_KEY \
--concurrency 10 \
--target_model qwen/qwen3.6-27b

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-MultiTurn-Customer-Support \
--target_api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
--judge_api_url https://tm-vm1-llm-1.cae.scicom.dev/v1/chat/completions \
--api_key API_KEY \
--concurrency 10 \
--target_model Qwen/Qwen3.6-35B-A3B

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-MultiTurn-Customer-Support \
--target_api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
--judge_api_url https://tm-vm1-llm-1.cae.scicom.dev/v1/chat/completions \
--api_key API_KEY \
--concurrency 10 \
--target_model Qwen/Qwen3.5-122B-A10B


python benchmark_redteam.py \
--dataset_path RedTeam-LLM-MultiTurn-Customer-Support \
--target_api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
--judge_api_url https://tm-vm1-llm-1.cae.scicom.dev/v1/chat/completions \
--api_key API_KEY \
--concurrency 10 \
--target_model mistralai/Mistral-Small-4-119B-2603

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-MultiTurn-Customer-Support \
--target_api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
--judge_api_url https://tm-vm1-llm-1.cae.scicom.dev/v1/chat/completions \
--api_key API_KEY \
--concurrency 10 \
--target_model MiniMaxAI/MiniMax-M2.7

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-MultiTurn-Customer-Support \
--target_api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
--api_key API_KEY \
--concurrency 10 \
--target_model google/gemma-4-31b-it

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-MultiTurn-Customer-Support \
--target_api_url https://vp67xvkjckxpgu-8000.proxy.runpod.net/v1/chat/completions \
--judge_api_url https://serverlessgpu.aies.scicom.dev/endpoint-rq/v1/chat/completions \
--api_key API_KEY \
--concurrency 50 \
--target_model zai-org/GLM-4.7-Flash

python benchmark_redteam.py \
--dataset_path RedTeam-LLM-MultiTurn-Customer-Support \
--target_api_url https://qabv3qjna1b4v2-8000.proxy.runpod.net/v1/chat/completions \
--judge_api_url https://serverlessgpu.aies.scicom.dev/endpoint-rq/v1/chat/completions \
--api_key API_KEY \
--concurrency 50 \
--target_model Qwen/Qwen3.5-397B-A17B
