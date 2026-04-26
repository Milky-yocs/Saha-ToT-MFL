import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOT_AGENT_PATH = os.path.join(ROOT, "server", "tot_agent.py")


def _load_tot_agent():
    spec = importlib.util.spec_from_file_location("mmqs_tot_agent", TOT_AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load tot_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TOTAPIAgent, module.TOTAPIError


def build_args():
    parser = argparse.ArgumentParser(description="Test MMQS ToT API connectivity.")
    parser.add_argument("--api_url", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--api_key_env", type=str, default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--Q", type=int, default=5)
    parser.add_argument("--probe_first", type=int, default=1)
    parser.add_argument("--probe_timeout", type=float, default=60.0)
    parser.add_argument("--prefer_stream", type=int, default=0)
    parser.add_argument("--force_non_stream", type=int, default=0)
    parser.add_argument("--stream_connect_timeout", type=float, default=10.0)
    parser.add_argument("--stream_read_timeout", type=float, default=180.0)
    parser.add_argument("--probe_stream_first_chunk", type=int, default=0)
    return parser.parse_args()


def _print_transient_hint(err_text: str):
    text = str(err_text or "").lower()
    if "http 504" in text or "gateway time-out" in text or "timed out" in text:
        print("HINT=upstream model or gateway timed out; run minimal probe then retry")


def main():
    args = build_args()
    if len(str(args.api_url).strip()) <= 0:
        print("TOT_API_OK=0")
        print("ERROR=api_url is empty; please pass your own --api_url")
        return 1
    if len(str(args.model).strip()) <= 0:
        print("TOT_API_OK=0")
        print("ERROR=model is empty; please pass your own --model")
        return 1
    if len(str(args.api_key_env).strip()) <= 0:
        print("TOT_API_OK=0")
        print("ERROR=api_key_env is empty; please pass your own --api_key_env")
        return 1
    TOTAPIAgent, TOTAPIError = _load_tot_agent()
    agent = TOTAPIAgent(
        api_url=args.api_url,
        model=args.model,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        retry=args.retry,
        prefer_stream=bool(int(args.prefer_stream)),
        force_non_stream=bool(int(args.force_non_stream)),
        stream_connect_timeout=args.stream_connect_timeout,
        stream_read_timeout=args.stream_read_timeout,
    )

    mock_state = {
        "round": 12,
        "round_total": 400,
        "stage_ratio": 0.03,
        "regional_accuracy": 0.71,
        "delay_mean": 0.41,
        "delay_std": 0.12,
        "delay_cv": 0.29,
        "fairness_gap": 0.36,
        "Q": 64,
        "component_mean": {
            "s_data": 0.57,
            "s_perf": 0.60,
            "s_res": 0.71,
            "s_cool": 0.66,
            "s_fair": 0.54,
            "s_modal": 0.95,
        },
        "component_std": {
            "s_data": 0.19,
            "s_perf": 0.17,
            "s_res": 0.15,
            "s_cool": 0.20,
            "s_fair": 0.21,
            "s_modal": 0.05,
        },
        "memory_top": [],
    }

    try:
        if int(args.probe_first) == 1:
            if int(args.probe_stream_first_chunk) == 1:
                probe_meta = agent.probe_connectivity_stream_first_chunk(prompt="ok", timeout=args.probe_timeout)
            else:
                probe_meta = agent.probe_connectivity(prompt="ok", timeout=args.probe_timeout)
            print("PROBE_OK=1")
            print("PROBE_META={}".format(json.dumps(probe_meta, ensure_ascii=False)))

        candidates, meta = agent.propose_weights(mock_state, Q=args.Q)
        serializable = []
        for item in candidates:
            serializable.append(
                {
                    "name": item.get("name", ""),
                    "weights": [float(v) for v in item.get("weights", [])],
                    "reason": item.get("reason", ""),
                    "score": float(item.get("score", 0.0)),
                }
            )
        print("TOT_API_OK=1")
        print("URL_REPR={}".format(repr(args.api_url)))
        print("MODEL={}".format(args.model))
        print("META={}".format(json.dumps(meta, ensure_ascii=False)))
        print("CANDIDATES={}".format(json.dumps(serializable, ensure_ascii=False)))
        return 0
    except TOTAPIError as exc:
        print("TOT_API_OK=0")
        print("URL_REPR={}".format(repr(args.api_url)))
        print("ERROR={}".format(str(exc)))
        _print_transient_hint(str(exc))
        return 1
    except Exception as exc:                                
        print("TOT_API_OK=0")
        print("URL_REPR={}".format(repr(args.api_url)))
        print("ERROR={}".format(str(exc)))
        _print_transient_hint(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
