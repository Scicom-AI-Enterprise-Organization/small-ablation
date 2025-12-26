import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from queue import Queue
from io import StringIO
from IPython.core.magic import register_cell_magic
import ast
import threading
import uuid
import sys
import traceback
import logging

_PROCESS_POOL = []
_COMMAND_QUEUES = []
_OUTPUT_QUEUES = []
_RESULT_QUEUES = []
_INITIALIZED = False
_WORLD_SIZE = 0
_OUTPUT_THREADS = []
_STOP_OUTPUT_THREADS = False
PRINT_ON_RANK = -1

def wrap_last_expr_with_print(code):
    """
    Thanks to ChatGPT.
    """
    try:
        tree = ast.parse(code)
        if not tree.body:
            return code

        last_stmt = tree.body[-1]
        if isinstance(last_stmt, ast.Expr) and not isinstance(last_stmt.value, ast.Call):
            start_line = last_stmt.lineno - 1
            end_line = last_stmt.end_lineno - 1 if hasattr(last_stmt, "end_lineno") else start_line

            code_lines = code.split('\n')

            expr_lines = code_lines[start_line:end_line + 1]
            indent = len(expr_lines[0]) - len(expr_lines[0].lstrip())
            indent_str = ' ' * indent
            wrapped_expr = f"{indent_str}print({''.join(expr_lines).strip()})"

            return "\n".join(code_lines[:start_line] + [wrapped_expr] + code_lines[end_line + 1:])
        else:
            return code
    except Exception as e:
        return code
        
def _output_monitor(rank, output_queue, callback):
    """
    Monitor for initialization messages only.
    Regular execution outputs are handled synchronously in execute_on_gpus.
    """
    global _STOP_OUTPUT_THREADS
    
    while not _STOP_OUTPUT_THREADS:
        try:
            try:
                output = output_queue.get(block=True, timeout=0.1)
                if callback and isinstance(output, dict) and output.get('type') == 'init':
                    # Only handle initialization messages
                    callback(rank, output.get('message', ''), output.get('status', 'success'))
            except Exception as e:
                pass
        except:
            pass

def _worker_process(rank, world_size, command_queue, output_queue, result_queue, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Send initialization message through output queue
    output_queue.put({
        'type': 'init',
        'message': f"Worker {rank}/{world_size} initialized on GPU {rank} on {master_addr}:{master_port}",
        'status': 'success'
    })
    result_queue.put({"status": "init", "message": f"Worker {rank}/{world_size} initialized"})
    
    local_namespace = {"rank": rank, "world_size": world_size}
    exec("import torch", local_namespace)
    exec("import torch.nn as nn", local_namespace)
    exec("import torch.distributed as dist", local_namespace)
    exec("import torch.multiprocessing as mp", local_namespace)
    exec("from torch.nn.parallel import DistributedDataParallel as DDP", local_namespace)

    # Buffer for collecting print outputs during command execution
    print_buffer = []
    
    def custom_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        message = sep.join(str(arg) for arg in args) + end
        print_buffer.append(message)

    local_namespace["print"] = custom_print
    
    while True:
        try:
            cmd, cmd_id = command_queue.get()
            
            if cmd == "EXIT":
                break

            # Clear print buffer for new command
            print_buffer.clear()
            
            output_buffer = StringIO()
            original_stdout = sys.stdout
            sys.stdout = output_buffer

            cmd = wrap_last_expr_with_print(cmd)
            
            try:
                exec(cmd, local_namespace)
                stdout_content = output_buffer.getvalue()
                all_output = ''.join(print_buffer) + stdout_content
                status = "success"

                result = {
                    "status": status,
                    "output": all_output, 
                    "cmd_id": cmd_id,
                    "rank": rank
                }
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                stdout_content = output_buffer.getvalue()
                all_output = ''.join(print_buffer) + stdout_content
                status = "error"

                result = {
                    "status": status,
                    "error": str(e),
                    "traceback": tb_str,
                    "output": all_output,
                    "cmd_id": cmd_id,
                    "rank": rank
                }
            finally:
                sys.stdout = original_stdout
                
            result_queue.put(result)
            
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

            result_queue.put({
                "status": "error", 
                "error": f"Worker error: {str(e)}",
                "traceback": tb_str,
                "cmd_id": cmd_id if 'cmd_id' in locals() else None,
                "rank": rank
            })
    
    dist.destroy_process_group()
    result_queue.put({"status": "exit", "message": f"Worker {rank} exiting"})

def _print_output(rank, message, status="success"):
    """Only used for initialization messages"""
    if PRINT_ON_RANK > -1 and rank != PRINT_ON_RANK:
        return

    if status == 'error':
        logging.error(f"[GPU {rank}] {message}")
    else:
        sys.stdout.write(f"[GPU {rank}] {message}")
        sys.stdout.flush()

def init_multigpus_repl(
    num_gpus: int = None, 
    master_addr: str = "localhost", 
    master_port: str = "12355",
    print_on_rank: int = -1,
):
    """
    Initializes a multi-GPU interactive REPL environment, typically within a Jupyter notebook.
    
    Parameters:
    ----------
    num_gpus : int, optional
        Number of GPUs to initialize across. If None, uses all available GPUs.

    master_addr : str, optional
        The master node's address used for setting up the process group (default: "localhost").
        Required in multi-node setups.

    master_port : str, optional
        Port used for initializing the torch.distributed process group (default: "12355").
        Must be free on the master node.

    print_on_rank : int, optional
        Specifies which rank should handle printing to stdout.
        - If set to -1, all ranks print.
        - If set to a specific rank (e.g., 0), only that rank will output to stdout.
    """

    global _PROCESS_POOL, _COMMAND_QUEUES, _OUTPUT_QUEUES, _RESULT_QUEUES
    global _INITIALIZED, _WORLD_SIZE, _OUTPUT_THREADS, _STOP_OUTPUT_THREADS
    global PRINT_ON_RANK
    
    if _INITIALIZED:
        logging.error("Multi-GPU REPL already initialized")
        return
    
    _STOP_OUTPUT_THREADS = False
    PRINT_ON_RANK = print_on_rank
    
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 0:
        raise ValueError("No GPUs available")
    
    logging.info(f"Initializing multi-GPU REPL with {num_gpus} GPUs...")
    _WORLD_SIZE = num_gpus
    
    for _ in range(num_gpus):
        _COMMAND_QUEUES.append(mp.Queue())
        _OUTPUT_QUEUES.append(mp.Queue())
        _RESULT_QUEUES.append(mp.Queue())
    
    for rank in range(num_gpus):
        p = mp.Process(
            target=_worker_process,
            args=(
                rank, 
                num_gpus, 
                _COMMAND_QUEUES[rank], 
                _OUTPUT_QUEUES[rank],
                _RESULT_QUEUES[rank],
                master_addr,
                master_port,
            )
        )
        p.start()
        _PROCESS_POOL.append(p)
    
    for rank in range(num_gpus):
        thread = threading.Thread(
            target=_output_monitor, 
            args=(rank, _OUTPUT_QUEUES[rank], _print_output),
            daemon=True
        )
        thread.start()
        _OUTPUT_THREADS.append(thread)
    
    for _ in range(num_gpus):
        result = _RESULT_QUEUES[_].get()
        if result["status"] == "init":
            logging.info(result["message"])
    
    _INITIALIZED = True
    logging.info("Multi-GPU REPL environment ready")

def execute_on_gpus(code):
    global _INITIALIZED, _WORLD_SIZE
    
    if not _INITIALIZED:
        raise RuntimeError("Multi-GPU REPL not initialized. Call init_multigpus_repl() first.")
    
    cmd_id = str(uuid.uuid4())
    
    for rank in range(_WORLD_SIZE):
        _COMMAND_QUEUES[rank].put((code, cmd_id))
    
    results = []
    for rank in range(_WORLD_SIZE):
        result = _RESULT_QUEUES[rank].get()
        results.append(result)
    
    results.sort(key=lambda r: r.get("rank", 0))
    
    # Print outputs synchronously in the current cell context
    for result in results:
        rank = result.get('rank', 0)
        output = result.get('output', '')
        status = result.get('status', 'success')
        
        if output and (PRINT_ON_RANK == -1 or rank == PRINT_ON_RANK):
            print(f"[GPU {rank}] {output}", end='')
        
        # Print errors/tracebacks
        if status == 'error' and (PRINT_ON_RANK == -1 or rank == PRINT_ON_RANK):
            traceback_str = result.get('traceback', result.get('error', ''))
            if traceback_str:
                print(f"[GPU {rank}] ERROR:\n{traceback_str}", file=sys.stderr)

@register_cell_magic
def multigpus(line, cell):
    if not _INITIALIZED:
        num_gpus = int(line.strip()) if line.strip() else None
        init_multigpus_repl(num_gpus)
    
    execute_on_gpus(cell)