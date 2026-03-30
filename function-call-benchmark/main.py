import time
import json
import os
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset
from difflib import SequenceMatcher
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from collections import defaultdict
import threading

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "your model"
MAX_WORKERS = 50
MAX_RETRIES = 10
TASK_TIMEOUT = 600

api_failure_lock = threading.Lock()
api_failure_count = 0
api_success_count = 0
api_retry_count = 0


def call_model(client, messages, tools, retries=MAX_RETRIES):
    global api_failure_count, api_success_count, api_retry_count
    
    attempt = 0
    base_delay = 2
    max_delay = 60
    start_time = time.time()
    max_total_time = 300
    
    while attempt < retries:
        if time.time() - start_time > max_total_time:
            with api_failure_lock:
                api_failure_count += 1
            print(f"\nTOTAL TIMEOUT after {max_total_time}s")
            return None
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                timeout=120
            )
            
            with api_failure_lock:
                api_success_count += 1
                if attempt > 0:
                    api_retry_count += attempt
            
            return response
            
        except Exception as e:
            attempt += 1
            error_msg = str(e)
            
            is_rate_limit = "rate" in error_msg.lower() or "429" in error_msg
            
            if attempt < retries:
                if is_rate_limit:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                else:
                    delay = min(base_delay * (1.5 ** attempt), max_delay / 2)
                
                print(f"\nAPI call failed (attempt {attempt}/{retries}): {error_msg}")
                print(f"   Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                with api_failure_lock:
                    api_failure_count += 1
                
                print(f"\nAPI CALL FAILED after {retries} attempts!")
                print(f"   Error: {error_msg}")
                print(f"   Model: {MODEL}")
                print(f"   This turn will be marked as API_FAILURE")
                return None
    
    return None


def calc_similarity(expected, actual):
    if type(expected) != type(actual):
        return 0.0
    
    if isinstance(expected, dict):
        if not expected and not actual:
            return 1.0
        
        keys = set(expected.keys()) | set(actual.keys())
        if not keys:
            return 1.0
        
        score = 0.0
        for k in keys:
            if k in expected and k in actual:
                score += calc_similarity(expected[k], actual[k])
        
        return score / len(keys)
    
    if isinstance(expected, list):
        if not expected and not actual:
            return 1.0
        if len(expected) != len(actual):
            max_len = max(len(expected), len(actual))
            min_len = min(len(expected), len(actual))
            if min_len == 0:
                return 0.0
            score = sum(calc_similarity(expected[i], actual[i]) for i in range(min_len))
            return score / max_len
        
        if not expected:
            return 1.0
        
        total = sum(calc_similarity(e, a) for e, a in zip(expected, actual))
        return total / len(expected)
    
    if expected == actual:
        return 1.0
    
    if isinstance(expected, str) and isinstance(actual, str):
        return SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
    
    return 0.0


def calc_param_accuracy(expected_params, predicted_params):
    if not expected_params:
        return 1.0, 0, 0
    
    correct = 0
    total = len(expected_params)
    
    for key in expected_params:
        if key in predicted_params and expected_params[key] == predicted_params[key]:
            correct += 1
    
    return correct / total, correct, total


def evaluate_multiple_tool_calls(expected_calls, predicted_calls, api_failed=False):
    if api_failed:
        results = []
        for call in expected_calls:
            func_name = call['function']['name']
            try:
                exp_params = json.loads(call['function']['arguments'])
            except:
                exp_params = {}
            
            results.append({
                'type': 'api_failure',
                'function': func_name,
                'func_match': False,
                'param_accuracy': 0.0,
                'correct_params': 0,
                'total_params': len(exp_params),
                'similarity': 0.0,
                'expected_func': func_name,
                'predicted_func': None,
                'expected_params': exp_params,
                'predicted_params': {},
                'api_failed': True
            })
        return results
    
    exp_by_name = defaultdict(list)
    for call in expected_calls:
        func_name = call['function']['name']
        exp_by_name[func_name].append(call)
    
    pred_by_name = defaultdict(list)
    for call in predicted_calls:
        func_name = call.function.name
        pred_by_name[func_name].append(call)
    
    results = []
    all_func_names = set(exp_by_name.keys()) | set(pred_by_name.keys())
    
    for func_name in all_func_names:
        exp_list = exp_by_name.get(func_name, [])
        pred_list = pred_by_name.get(func_name, [])
        
        max_pairs = max(len(exp_list), len(pred_list))
        
        for i in range(max_pairs):
            if i < len(exp_list) and i < len(pred_list):
                exp = exp_list[i]
                pred = pred_list[i]
                
                try:
                    exp_params = json.loads(exp['function']['arguments'])
                except:
                    exp_params = {}
                
                try:
                    pred_params = json.loads(pred.function.arguments)
                except:
                    pred_params = {}
                
                param_acc, correct, total = calc_param_accuracy(exp_params, pred_params)
                similarity = calc_similarity(exp_params, pred_params)
                
                results.append({
                    'type': 'match',
                    'function': func_name,
                    'func_match': True,
                    'param_accuracy': param_acc,
                    'correct_params': correct,
                    'total_params': total,
                    'similarity': similarity,
                    'expected_func': func_name,
                    'predicted_func': func_name,
                    'expected_params': exp_params,
                    'predicted_params': pred_params,
                    'api_failed': False
                })
                
            elif i < len(exp_list):
                exp = exp_list[i]
                try:
                    exp_params = json.loads(exp['function']['arguments'])
                except:
                    exp_params = {}
                
                results.append({
                    'type': 'missing',
                    'function': func_name,
                    'func_match': False,
                    'param_accuracy': 0.0,
                    'correct_params': 0,
                    'total_params': len(exp_params),
                    'similarity': 0.0,
                    'expected_func': func_name,
                    'predicted_func': None,
                    'expected_params': exp_params,
                    'predicted_params': {},
                    'api_failed': False
                })
                
            else:
                pred = pred_list[i]
                try:
                    pred_params = json.loads(pred.function.arguments)
                except:
                    pred_params = {}
                
                results.append({
                    'type': 'extra',
                    'function': func_name,
                    'func_match': False,
                    'param_accuracy': 0.0,
                    'correct_params': 0,
                    'total_params': 0,
                    'similarity': 0.0,
                    'expected_func': None,
                    'predicted_func': func_name,
                    'expected_params': {},
                    'predicted_params': pred_params,
                    'api_failed': False
                })
    
    return results


def load_row(dataset_name, config, split, idx):
    ds = load_dataset(dataset_name, config, split=split)
    row = ds[idx]
    
    messages = json.loads(row['conversation'])['messages']
    funcs = json.loads(row['functions'])
    
    tools = []
    for f in funcs['functions']:
        tools.append({
            "type": "function",
            "function": {
                "name": f['name'],
                "description": f['description'],
                "parameters": f['parameters']
            }
        })
    
    return messages, tools


def eval_row(client, messages, tools, row_idx):
    context = []
    all_call_results = []
    turn_num = 0
    
    for msg in messages:
        if msg['role'] == 'assistant' and msg.get('tool_calls'):
            turn_num += 1
            
            expected_calls = msg['tool_calls']
            
            resp = call_model(client, context, tools)
            
            api_failed = (resp is None)
            
            if not resp or not resp.choices[0].message.tool_calls:
                predicted_calls = []
            else:
                predicted_calls = resp.choices[0].message.tool_calls
            
            call_results = evaluate_multiple_tool_calls(
                expected_calls, 
                predicted_calls,
                api_failed=api_failed
            )
            
            for result in call_results:
                result.update({
                    'row': row_idx,
                    'turn': turn_num,
                    'expected_call_count': len(expected_calls),
                    'predicted_call_count': len(predicted_calls) if not api_failed else 0
                })
                all_call_results.append(result)
        
        context.append(msg)
    
    return all_call_results


def process_row(idx, dataset, config, split, api_key):
    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        messages, tools = load_row(dataset, config, split, idx)
        call_results = eval_row(client, messages, tools, idx)
        return (split, call_results)
    except Exception as e:
        print(f"\nERROR processing row {idx} in {split}: {e}")
        return (split, None)


def load_checkpoint(checkpoint_file):
    if checkpoint_file.exists():
        with checkpoint_file.open('r') as f:
            return json.load(f)
    return {'completed': {}, 'results': {}}


def save_checkpoint(checkpoint_file, completed_tasks, split_results):
    checkpoint = {
        'completed': completed_tasks,
        'results': {split: results for split, results in split_results.items()}
    }
    with checkpoint_file.open('w') as f:
        json.dump(checkpoint, f, indent=2)


def calc_metrics(all_results):
    if not all_results:
        return {
            'total_calls': 0,
            'total_turns': 0,
            'total_params': 0,
            'function_accuracy': 0,
            'parameter_accuracy': 0,
            'turn_level_parameter_accuracy': 0,
            'argument_similarity': 0,
            'call_count_accuracy': 0,
            'exact_match_rate': 0,
            'missing_calls': 0,
            'extra_calls': 0,
            'match_calls': 0,
            'api_failed_calls': 0,
            'perfect_parameter_turns': 0,
            'per_function': {},
            'failures': []
        }
    
    api_failed_results = [r for r in all_results if r.get('api_failed', False)]
    valid_results = [r for r in all_results if not r.get('api_failed', False)]
    
    total_calls = len(valid_results)
    correct_functions = sum(1 for r in valid_results if r['func_match'])
    
    total_params = sum(r['total_params'] for r in valid_results)
    correct_params = sum(r['correct_params'] for r in valid_results)
    
    total_similarity = sum(r['similarity'] for r in valid_results)
    
    missing_count = sum(1 for r in valid_results if r['type'] == 'missing')
    extra_count = sum(1 for r in valid_results if r['type'] == 'extra')
    match_count = sum(1 for r in valid_results if r['type'] == 'match')
    
    exact_matches = sum(1 for r in valid_results if r['type'] == 'match' and r['param_accuracy'] == 1.0)
    
    turn_groups = defaultdict(list)
    for r in valid_results:
        turn_key = (r['row'], r['turn'])
        turn_groups[turn_key].append(r)
    
    perfect_parameter_turns = sum(
        1 for turn_results in turn_groups.values()
        if all(
            r['type'] == 'match' and r['param_accuracy'] == 1.0 
            for r in turn_results
        )
    )
    
    turn_level_parameter_accuracy = (
        perfect_parameter_turns / len(turn_groups) 
        if turn_groups else 0
    )
    
    turns = {}
    for r in valid_results:
        turn_key = (r['row'], r['turn'])
        if turn_key not in turns:
            turns[turn_key] = {
                'expected': r['expected_call_count'],
                'predicted': r['predicted_call_count']
            }
    
    call_count_correct = sum(
        1 for t in turns.values() 
        if t['expected'] == t['predicted']
    )
    call_count_accuracy = call_count_correct / len(turns) if turns else 0
    
    per_function = defaultdict(lambda: {
        'total': 0,
        'correct_func': 0,
        'correct_params': 0,
        'total_params': 0,
        'exact_matches': 0
    })
    
    for r in valid_results:
        func = r['function']
        per_function[func]['total'] += 1
        if r['func_match']:
            per_function[func]['correct_func'] += 1
        per_function[func]['correct_params'] += r['correct_params']
        per_function[func]['total_params'] += r['total_params']
        if r['type'] == 'match' and r['param_accuracy'] == 1.0:
            per_function[func]['exact_matches'] += 1
    
    for func in per_function:
        stats = per_function[func]
        stats['function_accuracy'] = stats['correct_func'] / stats['total'] if stats['total'] > 0 else 0
        stats['parameter_accuracy'] = stats['correct_params'] / stats['total_params'] if stats['total_params'] > 0 else 0
        stats['exact_match_rate'] = stats['exact_matches'] / stats['total'] if stats['total'] > 0 else 0
    
    failures = [
        {
            'row': r['row'],
            'turn': r['turn'],
            'type': r['type'],
            'function': r['function'],
            'expected_func': r['expected_func'],
            'predicted_func': r['predicted_func'],
            'expected_params': r['expected_params'],
            'predicted_params': r['predicted_params'],
            'param_accuracy': r['param_accuracy'],
            'similarity': r['similarity']
        }
        for r in valid_results 
        if not r['func_match'] or r['param_accuracy'] < 1.0
    ]
    
    return {
        'total_calls': total_calls,
        'total_calls_including_failures': len(all_results),
        'total_turns': len(turns),
        'total_params': total_params,
        'function_accuracy': correct_functions / total_calls if total_calls > 0 else 0,
        'parameter_accuracy': correct_params / total_params if total_params > 0 else 0,
        'turn_level_parameter_accuracy': turn_level_parameter_accuracy,
        'perfect_parameter_turns': perfect_parameter_turns,
        'argument_similarity': total_similarity / total_calls if total_calls > 0 else 0,
        'call_count_accuracy': call_count_accuracy,
        'exact_match_rate': exact_matches / match_count if match_count > 0 else 0,
        'missing_calls': missing_count,
        'extra_calls': extra_count,
        'match_calls': match_count,
        'api_failed_calls': len(api_failed_results),
        'per_function': dict(per_function),
        'failures': failures
    }


def print_metrics(config_name, metrics):
    print(f"\n{config_name}")
    print("=" * 60)
    print(f"Total Turns: {metrics['total_turns']}")
    print(f"Total Params: {metrics['total_params']}")
    
    if metrics['api_failed_calls'] > 0:
        print(f"API Failed Calls: {metrics['api_failed_calls']} (excluded from metrics)")
    
    print(f"Function Accuracy: {metrics['function_accuracy']:.2%}")
    print(f"Parameter Accuracy (per-param): {metrics['parameter_accuracy']:.2%}")
    print(f"Turn-Level Parameter Accuracy: {metrics['turn_level_parameter_accuracy']:.2%} ({metrics['perfect_parameter_turns']}/{metrics['total_turns']} perfect turns)")
    print(f"Argument Similarity: {metrics['argument_similarity']:.2%}")


def print_api_stats():
    global api_success_count, api_failure_count, api_retry_count
    
    total_attempts = api_success_count + api_failure_count
    if total_attempts == 0:
        return
    
    print(f"\n{'='*60}")
    print("API CALL STATISTICS")
    print(f"{'='*60}")
    print(f"Successful calls: {api_success_count}")
    print(f"Failed calls: {api_failure_count}")
    print(f"Total retries: {api_retry_count}")
    print(f"Success rate: {api_success_count / total_attempts:.2%}")
    if api_success_count > 0:
        print(f"Avg retries per successful call: {api_retry_count / api_success_count:.2f}")
    print(f"{'='*60}")


def run_benchmark(dataset, config, splits, workers=MAX_WORKERS, limit_rows=None):
    global api_success_count, api_failure_count, api_retry_count
    
    api_success_count = 0
    api_failure_count = 0
    api_retry_count = 0
    
    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)
    
    checkpoint_file = result_dir / f"checkpoint_{config}.json"
    
    checkpoint = load_checkpoint(checkpoint_file)
    completed_tasks = checkpoint.get('completed', {})
    split_results = defaultdict(list, checkpoint.get('results', {}))
    
    split_sizes = {}
    for split in splits:
        ds = load_dataset(dataset, config, split=split)
        size = len(ds) if limit_rows is None else min(limit_rows, len(ds))
        split_sizes[split] = size
    
    total_rows = sum(split_sizes.values())
    completed_count = len(completed_tasks)
    
    print(f"\n{'='*60}")
    print(f"STARTING BENCHMARK")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"Dataset: {dataset}/{config}")
    print(f"Splits: {splits}")
    print(f"Total Rows: {total_rows}")
    print(f"Already Completed: {completed_count}")
    print(f"Remaining: {total_rows - completed_count}")
    print(f"Workers: {workers}")
    print(f"Max Retries: {MAX_RETRIES}")
    print(f"Task Timeout: {TASK_TIMEOUT}s")
    print(f"{'='*60}\n")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        
        for split in splits:
            for idx in range(split_sizes[split]):
                task_key = f"{split}:{idx}"
                
                if task_key in completed_tasks:
                    continue
                
                future = executor.submit(
                    process_row, idx, dataset, config, split, OPENROUTER_API_KEY
                )
                futures[future] = (split, idx, task_key)
        
        if not futures:
            print("All tasks already completed!")
        else:
            save_interval = 10
            tasks_since_save = 0
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc=f"Processing {config}"):
                split, idx, task_key = futures[future]
                
                try:
                    split_name, call_results = future.result(timeout=TASK_TIMEOUT)
                    
                    if call_results:
                        split_results[split_name].extend(call_results)
                    
                    completed_tasks[task_key] = True
                    tasks_since_save += 1
                    
                    if tasks_since_save >= save_interval:
                        save_checkpoint(checkpoint_file, completed_tasks, split_results)
                        tasks_since_save = 0
                    
                except TimeoutError:
                    tqdm.write(f"\nTIMEOUT: {split} row {idx} exceeded {TASK_TIMEOUT}s - skipping")
                    completed_tasks[task_key] = 'timeout'
                    
                except Exception as e:
                    tqdm.write(f"\nCritical error in {split} row {idx}: {e}")
                    completed_tasks[task_key] = f'error: {str(e)}'
            
            save_checkpoint(checkpoint_file, completed_tasks, split_results)
    
    print_api_stats()
    
    all_split_metrics = {}
    
    for split in splits:
        metrics = calc_metrics(split_results[split])
        all_split_metrics[split] = metrics
        
        split_file = result_dir / f"{split}.json"
        with split_file.open("w") as f:
            json.dump({
                'model': MODEL,
                'dataset': f"{dataset}/{config}",
                'split': split,
                **metrics
            }, f, indent=2)
        
        print_metrics(split, metrics)
    
    if len(splits) > 1:
        avg_func = sum(m['function_accuracy'] for m in all_split_metrics.values()) / len(all_split_metrics)
        avg_params = sum(m['parameter_accuracy'] for m in all_split_metrics.values()) / len(all_split_metrics)
        avg_turn_params = sum(m['turn_level_parameter_accuracy'] for m in all_split_metrics.values()) / len(all_split_metrics)
        avg_sim = sum(m['argument_similarity'] for m in all_split_metrics.values()) / len(all_split_metrics)
        total_turns = sum(m['total_turns'] for m in all_split_metrics.values())
        total_params = sum(m['total_params'] for m in all_split_metrics.values())
        total_perfect_turns = sum(m['perfect_parameter_turns'] for m in all_split_metrics.values())
        total_api_failed = sum(m['api_failed_calls'] for m in all_split_metrics.values())
        
        print_metrics(f"AVERAGE - {config}", {
            'total_turns': total_turns,
            'total_params': total_params,
            'api_failed_calls': total_api_failed,
            'function_accuracy': avg_func,
            'parameter_accuracy': avg_params,
            'turn_level_parameter_accuracy': avg_turn_params,
            'perfect_parameter_turns': total_perfect_turns,
            'argument_similarity': avg_sim
        })
        
        average_metrics = {
            'model': MODEL,
            'dataset': f"{dataset}/{config}",
            'splits': splits,
            'total_turns': total_turns,
            'total_params': total_params,
            'perfect_parameter_turns': total_perfect_turns,
            'api_failed_calls': total_api_failed,
            'average_function_accuracy': avg_func,
            'average_parameter_accuracy': avg_params,
            'average_turn_level_parameter_accuracy': avg_turn_params,
            'average_argument_similarity': avg_sim,
            'api_statistics': {
                'successful_calls': api_success_count,
                'failed_calls': api_failure_count,
                'total_retries': api_retry_count,
                'success_rate': api_success_count / (api_success_count + api_failure_count) if (api_success_count + api_failure_count) > 0 else 0
            },
            'individual_results': {
                split: {
                    'total_turns': m['total_turns'],
                    'total_params': m['total_params'],
                    'perfect_parameter_turns': m['perfect_parameter_turns'],
                    'api_failed_calls': m['api_failed_calls'],
                    'function_accuracy': m['function_accuracy'],
                    'parameter_accuracy': m['parameter_accuracy'],
                    'turn_level_parameter_accuracy': m['turn_level_parameter_accuracy'],
                    'argument_similarity': m['argument_similarity']
                }
                for split, m in all_split_metrics.items()
            }
        }
        
        with (result_dir / "average.json").open("w") as f:
            json.dump(average_metrics, f, indent=2)
        
        return average_metrics
    else:
        return all_split_metrics[splits[0]]


if __name__ == "__main__":
    run_benchmark(
        dataset="Scicom-intl/Function-Call",
        config="telco_multifunctions_premium_multiturn",
        splits=["test", "test_zh", "test_ms"],
        workers=50,
        limit_rows=None
    )