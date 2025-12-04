import ollama
import time
import os

MODEL = "llama3.1"

def test_gpu_inference():
    print(f"--- 1. Connecting to Ollama ({MODEL}) ---")
    
    # Check if we can run nvidia-smi to see baseline
    print("\n[Baseline GPU Status]")
    if os.system("nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader") != 0:
        print("Could not run nvidia-smi (ignore if on CPU-only setup)")

    print(f"\n--- 2. Sending request to {MODEL}... ---")
    print("   (Watch your 'ollama serve' terminal window for 'offloaded layers' log!)")
    
    start_time = time.time()
    
    # We ask for a longer response to keep the GPU busy for a few seconds
    response = ollama.chat(
        model=MODEL,
        messages=[
            {'role': 'user', 'content': 'Write a 200-word story about a robot discovering a GPU.'},
        ],
        options={
            "num_gpu": 99,  # Force layer offloading
            "num_ctx": 4096
        }
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Calculate stats
    eval_count = response.get('eval_count', 0)
    eval_duration_ns = response.get('eval_duration', 1) # nanoseconds
    eval_duration_sec = eval_duration_ns / 1_000_000_000
    
    tokens_per_sec = eval_count / eval_duration_sec if eval_duration_sec > 0 else 0

    print(f"\n--- 3. Inference Complete ---")
    print(f"Total Time:       {duration:.2f} seconds")
    print(f"Tokens Generated: {eval_count}")
    print(f"Speed:            {tokens_per_sec:.2f} tokens/sec")

    # Interpretation
    print("\n--- 4. Verdict ---")
    if tokens_per_sec > 30:
        print("✅ FAST (>30 t/s). Definitely running on GPU.")
    elif tokens_per_sec > 10:
        print("✅ MODERATE (10-30 t/s). Likely GPU (or very fast CPU).")
    else:
        print("⚠️ SLOW (<10 t/s). Suspected CPU usage. Check logs.")

    print("\n[Post-Inference GPU Status (Model should be loaded in VRAM)]")
    os.system("nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader")

if __name__ == "__main__":
    try:
        test_gpu_inference()
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure 'ollama serve' is running in a separate terminal!")