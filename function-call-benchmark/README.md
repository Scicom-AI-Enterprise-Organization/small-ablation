  # Function Call Benchmark                                                                                                                             
Evaluates LLM function-calling accuracy on the `Scicom-intl/Function-Call` dataset (Telco multi-function multi-turn) across three language splits:    English, Mandarin, and Malay.                                                                                                                         
                                                                                                                                                        
  ## Setup & Usage

```bash
uv sync
export OPENROUTER_API_KEY="your-key-here"
```

Set the `MODEL` variable in `main.py` to your target OpenRouter model ID, then run:

```bash
uv run main.py
```

Results are saved to `results/` as JSON files per split, plus `results/average.json`.

## How It Works

  - Loads multi-turn conversations with expected tool calls from HuggingFace                                                                            
  - Sends each turn to the model via OpenRouter and compares predicted vs expected tool calls
  - Supports parallel workers, checkpointing, and automatic retries with exponential backoff                                                            
                                                                                                                                                        
  ## Metrics                                                                                                                                            
                                                                                                                                                        
  | Metric | Description |
  |--------|-------------|
  | **Function Accuracy** | % of calls where the correct function was predicted |                                                                       
  | **Parameter Accuracy** | % of individual parameters with correct values |
  | **Turn-Level Perfect** | % of turns where all functions and params were exactly correct |                                                           
  | **Argument Similarity** | String similarity score between predicted and expected arguments |                                                        
                                                                                                   

## Overall Performance Comparison

Metrics averaged across all three language splits


| Model                              | Avg Func Acc | Avg Param Acc | Avg Turn-Level | Avg Arg Sim |
| ---------------------------------- | ------------ | ------------- | -------------- | ----------- |
| z-ai/glm-4.7                       | 90.49%       | 61.27%        | 35.96%         | 70.65%      |
| qwen/qwen3.5-27b                   | 86.74%       | 57.47%        | 29.96%         | 65.24%      |
| z-ai/glm-5                         | 86.73%       | 54.69%        | 29.30%         | 63.66%      |
| qwen/qwen3.5-397b-a17b             | 86.64%       | 58.64%        | 33.97%         | 66.04%      |
| qwen/qwen3.5-35b-a3b               | 85.71%       | 56.15%        | 28.97%         | 61.84%      |
| qwen/qwen3.5-122b-a10b             | 85.57%       | 57.74%        | 32.65%         | 64.56%      |
| qwen/qwen3-235b-a22b-thinking-2507 | 82.26%       | 52.51%        | 25.08%         | 59.29%      |
| qwen/qwen3.5-9b                    | 81.75%       | 51.78%        | 26.18%         | 59.34%      |
| z-ai/glm-4.7-flash                 | 81.48%       | 50.42%        | 23.15%         | 56.32%      |
| qwen/qwen3-30b-a3b-thinking-2507   | 79.41%       | 38.87%        | 18.70%         | 45.35%      |
| nvidia/nemotron-3-nano-30b-a3b     | 77.12%       | 47.59%        | 22.90%         | 50.48%      |
| qwen/qwen3-32b                     | 76.49%       | 43.71%        | 21.40%         | 49.05%      |
| openai/gpt-oss-120b                | 74.27%       | 26.83%        | 14.24%         | 27.42%      |
| nvidia/nemotron-3-super-120b-a12b  | 66.42%       | 42.27%        | 21.51%         | 44.83%      |




## Language-Specific Performance (Function Accuracy)

| Model | English | Mandarin | Malay |
|-------|:-------:|:--------:|:-----:|
| z-ai/glm-4.7 | 90.81% | 89.42% | 91.24% |
| z-ai/glm-5 | 87.28% | 85.43% | 87.46% |
| qwen/qwen3.5-397b-a17b | 87.08% | 86.12% | 86.71% |
| qwen/qwen3.5-27b | 86.51% | 85.54% | 88.16% |
| qwen/qwen3.5-35b-a3b | 85.71% | 82.96% | 88.45% |
| qwen/qwen3.5-122b-a10b | 85.59% | 85.85% | 85.28% |
| qwen/qwen3.5-9b | 84.10% | 79.94% | 81.19% |
| qwen/qwen3-235b-a22b-thinking-2507 | 83.10% | 78.47% | 85.20% |
| z-ai/glm-4.7-flash | 80.13% | 78.96% | 85.37% |
| qwen/qwen3-30b-a3b-thinking-2507 | 78.96% | 76.39% | 82.87% |
| qwen/qwen3-32b | 78.66% | 73.54% | 77.27% |
| nvidia/nemotron-3-nano-30b-a3b | 73.25% | 76.39% | 81.74% |
| openai/gpt-oss-120b | 72.84% | 73.17% | 76.81% |
| nvidia/nemotron-3-super-120b-a12b | 70.03% | 63.90% | 65.32% |


Detailed per-model results with per-language breakdowns, per-function stats, and failure logs are available in [`Function_Benchmark_Results/`](./Function_Benchmark_Results), organized by provider and model.