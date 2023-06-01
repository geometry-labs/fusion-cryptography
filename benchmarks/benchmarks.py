import pickle
import random
import statistics
import string
import time
from pathlib import Path
from typing import Callable, List, Dict, Any, Tuple
from loguru import logger

from memory_profiler import memory_usage

from fusion.fusion import fusion_setup, keygen, sign, aggregate, verify, OneTimeKeyTuple

# Prepare the input data
SEC_PARAMS: List[int] = [128, 256]
NUM_SUB_SAMPLES: int = 1
MIN_SUB_SAMPLE_SIZE: int = 2
MAX_SUB_SAMPLE_SIZE: int = 2**5

# Dictionaries for storing benchmark data
time_data: Dict[int, Dict[str, List[float]]] = {secpar: {} for secpar in SEC_PARAMS}
mem_data: Dict[int, Dict[str, List[float]]] = {secpar: {} for secpar in SEC_PARAMS}


def benchmark_func(func: Callable[..., Any], *args: Any) -> Tuple[Any, float, float]:
    start_mem: float = memory_usage()[0]
    start_time: float = time.time()

    result = func(*args)

    end_time: float = time.time()
    end_mem: float = memory_usage()[0]

    return result, end_time - start_time, end_mem - start_mem


def benchmark_all() -> None:
    for secpar in SEC_PARAMS:
        sub_sample_size: int = MIN_SUB_SAMPLE_SIZE
        while sub_sample_size <= MAX_SUB_SAMPLE_SIZE:
            logger.info(
                f"\nRunning benchmark for security parameter {secpar} and sub-sample size {sub_sample_size}"
            )
            seed: int = random.randint(0, 1000000)
            print(".", end="")
            # Initialize function-specific lists in the dictionaries
            for func_name in ["fusion_setup", "keygen", "sign", "aggregate", "verify"]:
                if func_name not in time_data[secpar]:
                    time_data[secpar][func_name] = []
                    mem_data[secpar][func_name] = []

            # Call fusion_setup and measure time and memory
            a, time_taken, mem_taken = benchmark_func(fusion_setup, secpar, seed)
            time_data[secpar]["fusion_setup"].append(time_taken)
            mem_data[secpar]["fusion_setup"].append(mem_taken)

            # Call keygen and measure time and memory
            keys: List[OneTimeKeyTuple] = []
            for _ in range(sub_sample_size * NUM_SUB_SAMPLES):
                print(".", end="")
                key, time_taken, mem_taken = benchmark_func(keygen, a, seed)
                keys.append(key)
                time_data[secpar]["keygen"].append(time_taken)
                mem_data[secpar]["keygen"].append(mem_taken)
            vks: List[str] = [key[1] for key in keys]

            # Construct string messages
            messages: List[str] = [
                "".join(random.choices(string.ascii_letters + string.digits, k=20))
                for _ in range(sub_sample_size * NUM_SUB_SAMPLES)
            ]

            # Call sign and measure time and memory
            sigs: List[str] = []
            for key, message in zip(keys, messages):
                print(".", end="")
                sig, time_taken, mem_taken = benchmark_func(sign, a, key, message)
                sigs.append(sig)
                time_data[secpar]["sign"].append(time_taken)
                mem_data[secpar]["sign"].append(mem_taken)

            # Call aggregate and measure time and memory
            agg_sigs: List[str] = []
            for i in range(NUM_SUB_SAMPLES):
                print(".", end="")
                agg_sig, time_taken, mem_taken = benchmark_func(
                    aggregate,
                    a,
                    vks[i * sub_sample_size : (i + 1) * sub_sample_size],
                    messages[i * sub_sample_size : (i + 1) * sub_sample_size],
                    sigs[i * sub_sample_size : (i + 1) * sub_sample_size],
                )
                agg_sigs.append(agg_sig)
                time_data[secpar]["aggregate"].append(time_taken)
                mem_data[secpar]["aggregate"].append(mem_taken)

            # Call verify and measure time and memory
            for i in range(NUM_SUB_SAMPLES):
                print(".", end="")
                try:
                    _, time_taken, mem_taken = benchmark_func(
                        verify,
                        a,
                        vks[i * sub_sample_size : (i + 1) * sub_sample_size],
                        messages[i * sub_sample_size : (i + 1) * sub_sample_size],
                        agg_sigs[i],
                    )
                except:
                    logger.info(" - exception - ")
                    _, time_taken, mem_taken = benchmark_func(
                        verify,
                        a,
                        vks[i * sub_sample_size : (i + 1) * sub_sample_size],
                        messages[i * sub_sample_size : (i + 1) * sub_sample_size],
                        agg_sigs[i],
                    )
                assert verify(
                    a,
                    vks[i * sub_sample_size : (i + 1) * sub_sample_size],  # noqa
                    messages[i * sub_sample_size : (i + 1) * sub_sample_size],
                    agg_sigs[i],  # noqa
                )
                time_data[secpar]["verify"].append(time_taken)
                mem_data[secpar]["verify"].append(mem_taken)

            sub_sample_size *= 2

        # After every SUB_SAMPLE_SIZE iteration, check if any mean time has exceeded MAX_RUNTIME
        for func_name in ["fusion_setup", "keygen", "sign", "aggregate", "verify"]:
            time_data[secpar][f"{func_name}_mean"] = statistics.mean(
                time_data[secpar][func_name]
            )
            time_data[secpar][f"{func_name}_median"] = statistics.median(
                time_data[secpar][func_name]
            )
            mem_data[secpar][f"{func_name}_mean"] = statistics.mean(
                mem_data[secpar][func_name]
            )
            mem_data[secpar][f"{func_name}_median"] = statistics.median(
                mem_data[secpar][func_name]
            )


def write_summary(filename: str) -> None:
    summary: str = ""
    for secpar in SEC_PARAMS:
        summary += f"Security Parameter: {secpar}\n"
        summary += "-" * 40 + "\n"
        for func_name in ["fusion_setup", "keygen", "sign", "aggregate", "verify"]:
            if (
                secpar in time_data
                and secpar in mem_data
                and f"{func_name}_mean" in time_data[secpar]
                and f"{func_name}_median" in time_data[secpar]
                and f"{func_name}_mean" in mem_data[secpar]
                and f"{func_name}_median" in mem_data[secpar]
            ):
                summary += f"Function: {func_name}\n"
                summary += (
                    f"Mean Time: {time_data[secpar][f'{func_name}_mean']} seconds\n"
                )
                summary += (
                    f"Median Time: {time_data[secpar][f'{func_name}_median']} seconds\n"
                )
                summary += f"Mean Memory: {mem_data[secpar][f'{func_name}_mean']} MB\n"
                summary += (
                    f"Median Memory: {mem_data[secpar][f'{func_name}_median']} MB\n"
                )
                summary += "-" * 40 + "\n"

    # Handle saving

    output_dir: Path = Path.cwd() / "benchmarks_output"
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / filename, "w") as f:
        f.write(summary)

    with open(output_dir / "time_data.pickle", "wb") as f:
        pickle.dump(time_data, f)
        logger.info(f"Wrote pickle file to {output_dir / 'time_data.pickle'}")


if __name__ == "__main__":
    benchmark_all()
    write_summary("benchmark_summary.txt")
    logger.info("Done")
