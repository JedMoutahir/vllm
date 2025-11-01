import argparse, asyncio, aiohttp, time, json, os
from pathlib import Path
from tqdm import tqdm

SYSTEM_PROMPT = "You are a helpful assistant."
USER_PLACEHOLDER = "{prompt}"

async def worker(name, queue, session, url, headers, payload_base, results_path):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        prompt = item
        payload = dict(payload_base)
        payload["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PLACEHOLDER.format(prompt=prompt)}
        ]
        t0 = time.perf_counter()
        status = 0
        try:
            async with session.post(f"{url}/chat/completions", json=payload, timeout=600) as resp:
                status = resp.status
                data = await resp.json()
        except Exception as e:
            data = {"error": str(e)}
        t1 = time.perf_counter()
        rec = {
            "ts": time.time(),
            "latency_s": t1 - t0,
            "status": status,
            "prompt_chars": len(prompt),
            "response_chars": len(json.dumps(data)) if isinstance(data, dict) else 0,
            "raw": data
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        queue.task_done()

async def main_async(args):
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    # Load prompts
    prompts = []
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    if not prompts:
        prompts = ["Summarize the benefits of unit testing in one paragraph."] * args.requests

    # replicate prompts to match requests
    expanded = (prompts * ((args.requests + len(prompts) - 1) // len(prompts)))[:args.requests]

    queue = asyncio.Queue()
    for p in expanded:
        await queue.put(p)

    payload_base = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens
    }
    headers = {"Content-Type": "application/json"}

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=None)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        workers = [asyncio.create_task(worker(f"w{i}", queue, session, args.endpoint, headers, payload_base, results_path)) for i in range(args.concurrency)]
        # progress bar
        pbar = tqdm(total=args.requests, desc="Requests")
        done = 0
        while done < args.requests:
            await asyncio.sleep(0.1)
            # count lines written so far
            if results_path.exists():
                with open(results_path, "r", encoding="utf-8") as f:
                    cnt = sum(1 for _ in f)
                pbar.n = cnt
                pbar.refresh()
                done = cnt
        # signal shutdown
        for _ in workers:
            await queue.put(None)
        await queue.join()
        for w in workers:
            w.cancel()
        pbar.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="dummy")  # vLLM ignores this; accepts any string
    ap.add_argument("--prompts-file", default=None)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out", default="runs/vllm_bench")
    args = ap.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
