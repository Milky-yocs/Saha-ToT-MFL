import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

import numpy as np
import requests

try:
    import websocket                
except Exception:                                
    websocket = None


class TOTAPIError(RuntimeError):
    """ToT API call or parsing failed."""


class TOTAPIAgent(object):
    """External LLM ToT branch generator for MMQS dynamic weights."""

    WEIGHT_KEYS = ("data", "modal", "perf", "res", "cool", "fair")

    def __init__(
        self,
        api_url: str,
        model: str,
        api_key_env: str = "",
        timeout: float = 120.0,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.7,
        retry: int = 1,
        prefer_stream: bool = False,
        force_non_stream: bool = False,
        enable_reformat_repair: bool = False,
        stream_connect_timeout: float = 10.0,
        stream_read_timeout: Optional[float] = None,
    ):
        self.api_url = self._sanitize_url(api_url)
        self.model = self._normalize_model_name(model)
        self.api_key_env = str(api_key_env).strip()
        self.timeout = max(1.0, float(timeout))
        self.max_tokens = max(32, int(max_tokens))
        self.temperature = float(np.clip(float(temperature), 0.0, 2.0))
        self.top_p = float(np.clip(float(top_p), 0.0, 1.0))
        self.retry = max(0, int(retry))
        self.prefer_stream = bool(prefer_stream)
        self.force_non_stream = bool(force_non_stream)
        self.enable_reformat_repair = bool(enable_reformat_repair)
        self.stream_connect_timeout = max(1.0, float(stream_connect_timeout))
        default_read_timeout = self.timeout if stream_read_timeout is None else float(stream_read_timeout)
        self.stream_read_timeout = max(1.0, default_read_timeout)
        self.backoff_base_sec = 1.5
        self.backoff_max_sec = 12.0

    def _build_payload(
        self,
        state: Dict[str, Any],
        Q: int,
        max_tokens: int = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        token_budget = self.max_tokens if max_tokens is None else max(32, int(max_tokens))
        system_prompt, user_prompt = self._build_propose_prompts(state, Q)
        return self._build_chat_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=token_budget,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=stream,
        )

    def _build_probe_payload(self, prompt: str = "ok", max_tokens: int = 16, stream: bool = False) -> Dict[str, Any]:
        return self._build_chat_payload(
            system_prompt="",
            user_prompt=str(prompt or "ok"),
            max_tokens=max(8, int(max_tokens)),
            temperature=0.0,
            top_p=1.0,
            stream=stream,
        )

    def _build_prune_payload(
        self,
        state: Dict[str, Any],
        evaluated_branches: List[Dict[str, Any]],
        max_tokens: int = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        token_budget = self.max_tokens if max_tokens is None else max(64, int(max_tokens))
        system_prompt, user_prompt = self._build_prune_prompts(state, evaluated_branches)
        return self._build_chat_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=token_budget,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=stream,
        )

    @staticmethod
    def _sanitize_url(url: str) -> str:
        clean = str(url or "").strip().replace("\r", "").replace("\n", "")
        if not clean.startswith("http"):
            raise TOTAPIError("invalid api_url: {}".format(repr(clean)))
        return clean

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        raw = str(model or "").strip()
        low = raw.lower()
        alias_map = {
            "glm-5": "zai-org/GLM-5",
            "glm5": "zai-org/GLM-5",
        }
        return alias_map.get(low, raw)

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.environ.get(name, None)
        if raw is None:
            return bool(default)
        text = str(raw).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        return bool(default)

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        proxy_url = str(os.environ.get("MMQS_TOT_API_PROXY", "")).strip()
        use_system_proxy = TOTAPIAgent._env_flag("MMQS_TOT_API_USE_SYSTEM_PROXY", True)

        if proxy_url:
            session.trust_env = False
            session.proxies = {"http": proxy_url, "https": proxy_url}
            return session

        if use_system_proxy:
            session.trust_env = True
            return session

        session.trust_env = False
        session.proxies = {}
        return session

    def _is_atomgit_v5(self) -> bool:
        lower_url = self.api_url.lower()
        return ("api.atomgit.com" in lower_url) and ("/api/v5/" in lower_url)

    def _use_single_user_role_message(self) -> bool:
        if self._env_flag("MMQS_TOT_API_SINGLE_USER_ROLE", False):
            return True
        model_name = str(self.model or "").strip().lower()
        if model_name.startswith("models/"):
            model_name = model_name.split("/", 1)[1]
        return model_name.startswith("gemma-")

    def _build_chat_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool = False,
    ) -> Dict[str, Any]:
        if self._is_atomgit_v5():
            merged_prompt = str(user_prompt or "").strip()
            sys_text = str(system_prompt or "").strip()
            if sys_text:
                merged_prompt = "{}\n\n{}".format(sys_text, merged_prompt)
            return {
                "model": self.model,
                "messages": [merged_prompt],
                "maxTokens": max(8, int(max_tokens)),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": 0,
                "frequency_penalty": 0,
            }

        if self._use_single_user_role_message():
            merged_prompt = str(user_prompt or "").strip()
            sys_text = str(system_prompt or "").strip()
            if sys_text:
                merged_prompt = "{}\n\n{}".format(sys_text, merged_prompt)
            return {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": merged_prompt,
                    },
                ],
                "stream": bool(stream),
                "max_tokens": max(8, int(max_tokens)),
                "temperature": float(temperature),
                "top_p": float(top_p),
            }

        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": str(system_prompt or ""),
                },
                {
                    "role": "user",
                    "content": str(user_prompt or ""),
                },
            ],
            "stream": bool(stream),
            "max_tokens": max(8, int(max_tokens)),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }

    def _build_propose_prompts(self, state: Dict[str, Any], Q: int) -> Tuple[str, str]:
        if self._use_bidi_live_api():
            stage_ratio = self._safe_float(state.get("stage_ratio", 0.0), 0.0)
            regional_accuracy = self._safe_float(state.get("regional_accuracy", 0.0), 0.0)
            fairness_gap = self._safe_float(state.get("fairness_gap", 0.0), 0.0)
            delay_cv = self._safe_float(state.get("delay_cv", 0.0), 0.0)
            system_prompt = (
                "You are an MMQS 6D weight planner for hierarchical federated learning. "
                "Output must be exactly one line in this format only: "
                "W=[w1,w2,w3,w4,w5,w6]. "
                "No markdown, no explanation, no extra tokens."
            )
            user_prompt = (
                "Task: choose MMQS 6D weights from regional signals only.\n"
                "State: stage_ratio={:.4f}, regional_accuracy={:.4f}, fairness_gap={:.4f}, delay_cv={:.4f}.\n"
                "Weight order is fixed: [w_data,w_modal,w_perf,w_res,w_cool,w_fair].\n"
                "Parameter meanings:\n"
                "- regional_accuracy: higher is better.\n"
                "- stage_ratio: progress (early<0.33, mid 0.33-0.66, late>0.66).\n"
                "- fairness_gap: higher means less fair.\n"
                "- delay_cv: higher means less stable latency.\n"
                "Action guidance:\n"
                "- If fairness_gap is high, increase w_fair.\n"
                "- If delay_cv is high, increase w_res and/or w_cool.\n"
                "- If regional_accuracy is low in early stage, increase w_data and w_perf.\n"
                "Hard constraints:\n"
                "1) 6 numbers only.\n"
                "2) each weight in [0,1], non-negative.\n"
                "3) sum(weights)=1.\n"
                "4) Output exactly one line: W=[w1,w2,w3,w4,w5,w6]."
            ).format(stage_ratio, regional_accuracy, fairness_gap, delay_cv)
            return system_prompt, user_prompt

        system_prompt = (
            "You are an MMQS 6D weight planner for hierarchical federated learning. "
            "Task: generate candidate weight vectors for client selection scoring. "
            "Weight layout is fixed: [w_data, w_modal, w_perf, w_res, w_cool, w_fair]. "
            "Primary objective: improve regional-model validation performance. "
            "Secondary objectives: reduce fairness_gap and delay instability (delay_cv), "
            "while keeping robust exploration. "
            "Regional Feedback Only: use only regional signals from the prompt/state JSON. "
            "Never use or infer any global-model metrics. "
            "Return strict JSON only, no markdown, no chain-of-thought."
        )
        user_prompt = self._build_prompt(state, Q)
        return system_prompt, user_prompt

    def _build_prune_prompts(
        self,
        state: Dict[str, Any],
        evaluated_branches: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        if self._use_bidi_live_api():
            system_prompt = (
                "You are a ToT branch pruning optimizer for MMQS. "
                "Output strict JSON only. No markdown. No extra text."
            )
            user_prompt = (
                "State JSON:\n{}\n"
                "Evaluated branches JSON:\n{}\n"
                "Pick winner and one pruned branch using regional_accuracy only.\n"
                "Return JSON only:\n"
                "{{\"pruned_branch_id\":\"...\",\"winner_branch_id\":\"...\","
                "\"next_exploration_direction\":\"...\",\"next_seed_weights\":[w1,w2,w3,w4,w5,w6]}}.\n"
                "Constraints: 6 weights, each in [0,1], sum=1."
            ).format(
                json.dumps(state, ensure_ascii=False, separators=(",", ":")),
                json.dumps(evaluated_branches, ensure_ascii=False, separators=(",", ":")),
            )
            return system_prompt, user_prompt

        system_prompt = (
            "You are a Tree-of-Thought pruning optimizer for MMQS 6D weights in hierarchical federated learning. "
            "Use regional feedback only to prune and evolve branches. "
            "Never use or infer any global-model metrics. "
            "Output strict JSON only, no markdown, no chain-of-thought."
        )
        user_prompt = self._build_prune_prompt(state, evaluated_branches)
        return system_prompt, user_prompt

    def _model_name_core(self) -> str:
        model_name = str(self.model or "").strip().lower()
        if model_name.startswith("models/"):
            model_name = model_name.split("/", 1)[1]
        return model_name

    def _use_bidi_live_api(self) -> bool:
        if self._env_flag("MMQS_TOT_API_DISABLE_BIDI", False):
            return False
        if self._env_flag("MMQS_TOT_API_FORCE_BIDI", False):
            return True
        model_name = self._model_name_core()
        return "native-audio" in model_name

    def _resolve_ws_proxy(self) -> Dict[str, Any]:
        proxy_url = str(os.environ.get("MMQS_TOT_API_PROXY", "")).strip()
        if not proxy_url and self._env_flag("MMQS_TOT_API_USE_SYSTEM_PROXY", True):
            proxy_url = str(
                os.environ.get("HTTPS_PROXY")
                or os.environ.get("https_proxy")
                or os.environ.get("HTTP_PROXY")
                or os.environ.get("http_proxy")
                or ""
            ).strip()
        if not proxy_url:
            return {}
        if "://" not in proxy_url:
            proxy_url = "http://{}".format(proxy_url)
        parsed = urlparse(proxy_url)
        if not parsed.hostname:
            return {}
        proxy_cfg = {
            "http_proxy_host": parsed.hostname,
            "proxy_type": "http",
        }
        if parsed.port:
            proxy_cfg["http_proxy_port"] = int(parsed.port)
        if parsed.username:
            proxy_cfg["http_proxy_auth"] = (
                parsed.username,
                parsed.password or "",
            )
        return proxy_cfg

    def _build_bidi_ws_urls(self, api_key: str) -> List[str]:
        host = urlparse(self.api_url).hostname or ""
        if not host:
            raise TOTAPIError("cannot derive websocket host from api_url")
        key_q = quote_plus(str(api_key))
        return [
            "wss://{}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={}".format(
                host, key_q
            ),
            "wss://{}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={}".format(
                host, key_q
            ),
        ]

    def _extract_bidi_error(self, event: Dict[str, Any]) -> str:
        if not isinstance(event, dict):
            return ""
        err = event.get("error", None)
        if isinstance(err, dict):
            msg = str(err.get("message", "")).strip()
            if msg:
                return msg
            return json.dumps(err, ensure_ascii=False)
        if isinstance(err, str) and err.strip():
            return err.strip()
        go_away = event.get("goAway", event.get("go_away", None))
        if isinstance(go_away, dict):
            msg = str(go_away.get("reason", go_away.get("message", ""))).strip()
            if msg:
                return msg
        return ""

    def _extract_bidi_text(self, event: Dict[str, Any]) -> str:
        if not isinstance(event, dict):
            return ""
        server_content = event.get("serverContent", event.get("server_content", {}))
        if not isinstance(server_content, dict):
            return ""
        output_trans = server_content.get("outputTranscription", server_content.get("output_transcription", {}))
        if isinstance(output_trans, dict):
            text_out = output_trans.get("text", "")
            if isinstance(text_out, str) and text_out:
                return text_out
        model_turn = server_content.get("modelTurn", server_content.get("model_turn", {}))
        if not isinstance(model_turn, dict):
            return ""
        parts = model_turn.get("parts", [])
        if not isinstance(parts, list):
            return ""
        chunks = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text", "")
            if isinstance(text, str) and text:
                chunks.append(text)
        return "".join(chunks)

    def _is_bidi_turn_complete(self, event: Dict[str, Any]) -> bool:
        if not isinstance(event, dict):
            return False
        server_content = event.get("serverContent", event.get("server_content", {}))
        if not isinstance(server_content, dict):
            return False
        if bool(server_content.get("turnComplete", False)):
            return True
        if bool(server_content.get("turn_complete", False)):
            return True
        return False

    def _request_content_bidi_once(
        self,
        ws_url: str,
        prompt_text: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        timeout: float,
        use_snake_case: bool = False,
    ) -> str:
        if websocket is None:
            raise TOTAPIError("websocket-client not installed; pip install websocket-client==1.5.1")
        conn_timeout = max(1.0, min(float(timeout), float(self.stream_connect_timeout)))
        ws_kwargs: Dict[str, Any] = {
            "timeout": conn_timeout,
            "enable_multithread": False,
        }
        ws_proxy = self._resolve_ws_proxy()
        if len(ws_proxy) > 0:
            ws_kwargs.update(ws_proxy)

        ws = websocket.create_connection(ws_url, **ws_kwargs)
        try:
            ws.settimeout(max(1.0, float(timeout)))
            is_native_audio = "native-audio" in self._model_name_core()
            voice_name = str(os.environ.get("MMQS_TOT_API_BIDI_VOICE", "Kore")).strip() or "Kore"
            enable_audio_trans = self._env_flag("MMQS_TOT_API_BIDI_OUTPUT_TRANSCRIPTION", True)
            if is_native_audio:
                cfg_camel = {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": voice_name,
                            }
                        }
                    },
                }
                cfg_snake = {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": voice_name,
                            }
                        }
                    },
                }
            else:
                cfg_camel = {
                    "responseModalities": ["TEXT"],
                    "temperature": float(temperature),
                    "topP": float(top_p),
                    "maxOutputTokens": max(8, int(max_tokens)),
                }
                cfg_snake = {
                    "response_modalities": ["TEXT"],
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "max_output_tokens": max(8, int(max_tokens)),
                }
            if use_snake_case:
                setup_payload = {
                    "setup": {
                        "model": self.model,
                        "generation_config": cfg_snake,
                    }
                }
                if is_native_audio and enable_audio_trans:
                    setup_payload["setup"]["output_audio_transcription"] = {}
                client_payload = {
                    "client_content": {
                        "turns": [{"role": "user", "parts": [{"text": str(prompt_text)}]}],
                        "turn_complete": True,
                    }
                }
            else:
                setup_payload = {
                    "setup": {
                        "model": self.model,
                        "generationConfig": cfg_camel,
                    }
                }
                if is_native_audio and enable_audio_trans:
                    setup_payload["setup"]["outputAudioTranscription"] = {}
                client_payload = {
                    "clientContent": {
                        "turns": [{"role": "user", "parts": [{"text": str(prompt_text)}]}],
                        "turnComplete": True,
                    }
                }

            ws.send(json.dumps(setup_payload, ensure_ascii=False))
            setup_deadline = time.time() + max(3.0, min(15.0, float(timeout)))
            while time.time() < setup_deadline:
                raw = ws.recv()
                if raw is None:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "ignore")
                raw = str(raw).strip()
                if not raw:
                    continue
                evt = json.loads(raw)
                err = self._extract_bidi_error(evt)
                if err:
                    raise TOTAPIError("bidi setup error: {}".format(err))
                if evt.get("setupComplete") is not None or evt.get("setup_complete") is not None:
                    break

            ws.send(json.dumps(client_payload, ensure_ascii=False))
            chunks = []
            deadline = time.time() + max(3.0, float(timeout))
            while time.time() < deadline:
                raw = ws.recv()
                if raw is None:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "ignore")
                raw = str(raw).strip()
                if not raw:
                    continue
                evt = json.loads(raw)
                err = self._extract_bidi_error(evt)
                if err:
                    raise TOTAPIError("bidi runtime error: {}".format(err))
                text = self._extract_bidi_text(evt)
                if text:
                    chunks.append(text)
                if self._is_bidi_turn_complete(evt):
                    if len(chunks) > 0:
                        return "".join(chunks)
                    break

            if len(chunks) > 0:
                return "".join(chunks)
            raise TOTAPIError("empty content in bidi response")
        finally:
            try:
                ws.close()
            except Exception:                                
                pass

    def _request_content_bidi(
        self,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        timeout: float,
    ) -> str:
        prompt_text = str(user_prompt or "").strip()
        sys_text = str(system_prompt or "").strip()
        if sys_text and self._env_flag("MMQS_TOT_API_BIDI_INCLUDE_SYSTEM", False):
            prompt_text = "{}\n\n{}".format(sys_text, prompt_text)
        ws_urls = self._build_bidi_ws_urls(api_key)
        last_error = None
        for ws_url in ws_urls:
            for use_snake_case in (False, True):
                try:
                    return self._request_content_bidi_once(
                        ws_url=ws_url,
                        prompt_text=prompt_text,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                        use_snake_case=use_snake_case,
                    )
                except Exception as exc:                                
                    last_error = exc
                    continue
        raise TOTAPIError(str(last_error))

    def propose_weights(self, state: Dict[str, Any], Q: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.api_url:
            raise TOTAPIError("ToT API URL is empty")
        if not self.model:
            raise TOTAPIError("ToT API model is empty")

        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise TOTAPIError("missing api key env: {}".format(self.api_key_env))

        Q = max(1, int(Q))
        if self._use_bidi_live_api():
            system_prompt, user_prompt = self._build_propose_prompts(state, Q)
            start = time.time()
            last_error = None
            for attempt in range(self.retry + 1):
                try:
                    token_budget = max(64, int(self.max_tokens / (2 ** attempt)))
                    content = self._request_content_bidi(
                        api_key=api_key,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=token_budget,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        timeout=self.timeout,
                    )
                    raw_tail = str(content)[-1200:]
                    candidates = []
                    json_parse_error = None
                    try:
                        candidates = self._parse_candidates(content, Q)
                    except TOTAPIError as exc:
                        json_parse_error = exc
                    except Exception as exc:                                
                        json_parse_error = exc
                    if len(candidates) <= 0:
                        candidates = self._parse_candidates_from_text_fallback(content, Q)
                    if len(candidates) <= 0:
                        candidates = self._parse_candidates_from_named_weights(content, Q)
                    if len(candidates) > 0:
                        elapsed = time.time() - start
                        return candidates, {"elapsed_sec": float(elapsed), "attempt": int(attempt + 1)}
                    tail_safe = raw_tail.replace("\r", " ").replace("\n", " ")
                    if json_parse_error is not None:
                        raise TOTAPIError(
                            "no valid candidates parsed; json_parse_error={}; raw_tail={}".format(
                                str(json_parse_error), tail_safe[:500]
                            )
                        )
                    raise TOTAPIError(
                        "no valid candidates parsed from model output; raw_tail={}".format(tail_safe[:500])
                    )
                except Exception as exc:                                
                    last_error = exc
                    if attempt < self.retry and self._is_transient_error(exc):
                        self._sleep_backoff(attempt)
                        continue
            raise TOTAPIError(str(last_error))

        headers = {
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type": "application/json",
        }

        start = time.time()
        last_error = None
        for attempt in range(self.retry + 1):
            try:
                raw_tail = ""
                token_budget = max(64, int(self.max_tokens / (2 ** attempt)))
                session = self._build_session()
                parse_error = None
                mode_errors = []
                for stream_mode in self._iter_stream_modes():
                    try:
                        payload = self._build_payload(
                            state=state,
                            Q=Q,
                            max_tokens=token_budget,
                            stream=stream_mode,
                        )
                        content = self._request_content(
                            session=session,
                            headers=headers,
                            payload=payload,
                            stream=stream_mode,
                            timeout=self._resolve_timeout(stream_mode),
                        )
                        raw_tail = str(content)[-1200:]
                        candidates = []
                        json_parse_error = None
                        try:
                            candidates = self._parse_candidates(content, Q)
                        except TOTAPIError as exc:
                            json_parse_error = exc
                        except Exception as exc:                                
                            json_parse_error = exc
                        if len(candidates) <= 0:
                            candidates = self._parse_candidates_from_text_fallback(content, Q)
                        if len(candidates) <= 0:
                            candidates = self._parse_candidates_from_named_weights(content, Q)
                        if len(candidates) <= 0 and self.enable_reformat_repair:
                            candidates = self._repair_candidates_via_reformat(
                                session=session,
                                headers=headers,
                                raw_content=content,
                                Q=Q,
                            )
                        if len(candidates) > 0:
                            elapsed = time.time() - start
                            return candidates, {"elapsed_sec": float(elapsed), "attempt": int(attempt + 1)}
                        tail_safe = raw_tail.replace("\r", " ").replace("\n", " ")
                        if json_parse_error is not None:
                            parse_error = TOTAPIError(
                                "no valid candidates parsed; json_parse_error={}; raw_tail={}".format(
                                    str(json_parse_error), tail_safe[:500]
                                )
                            )
                        else:
                            parse_error = TOTAPIError(
                                "no valid candidates parsed from model output; raw_tail={}".format(tail_safe[:500])
                            )
                        mode_errors.append(
                            "{}: {}".format("stream" if stream_mode else "non_stream", str(parse_error))
                        )
                    except Exception as mode_exc:                                
                        mode_errors.append(
                            "{}: {}".format("stream" if stream_mode else "non_stream", str(mode_exc)[:300])
                        )
                        continue
                if parse_error is not None:
                    if len(mode_errors) > 0:
                        parse_error = TOTAPIError("{}; mode_errors={}".format(
                            str(parse_error), " | ".join(mode_errors[:3])
                        ))
                    raise parse_error
                if len(mode_errors) > 0:
                    raise TOTAPIError("no stream mode succeeded for propose_weights; mode_errors={}".format(
                        " | ".join(mode_errors[:3])
                    ))
                raise TOTAPIError("no stream mode succeeded for propose_weights")
            except Exception as exc:                                
                last_error = exc
                if attempt < self.retry:
                    if self._is_transient_error(exc):
                        self._sleep_backoff(attempt)
                    continue
        raise TOTAPIError(str(last_error))

    def prune_and_evolve(
        self,
        state: Dict[str, Any],
        evaluated_branches: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self.api_url:
            raise TOTAPIError("ToT API URL is empty")
        if not self.model:
            raise TOTAPIError("ToT API model is empty")

        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise TOTAPIError("missing api key env: {}".format(self.api_key_env))

        branches = self._sanitize_evaluated_branches(evaluated_branches)
        if len(branches) <= 0:
            raise TOTAPIError("empty evaluated branches for pruning")

        if self._use_bidi_live_api():
            system_prompt, user_prompt = self._build_prune_prompts(state, branches)
            start = time.time()
            last_error = None
            for attempt in range(self.retry + 1):
                try:
                    token_budget = max(128, int(self.max_tokens / (2 ** attempt)))
                    content = self._request_content_bidi(
                        api_key=api_key,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=token_budget,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        timeout=self.timeout,
                    )
                    prune_result = self._parse_prune_result(content, branches)
                    elapsed = time.time() - start
                    return prune_result, {"elapsed_sec": float(elapsed), "attempt": int(attempt + 1)}
                except Exception as exc:                                
                    last_error = exc
                    if attempt < self.retry and self._is_transient_error(exc):
                        self._sleep_backoff(attempt)
                        continue
            raise TOTAPIError(str(last_error))

        headers = {
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type": "application/json",
        }

        start = time.time()
        last_error = None
        for attempt in range(self.retry + 1):
            try:
                token_budget = max(128, int(self.max_tokens / (2 ** attempt)))
                session = self._build_session()
                parse_error = None
                mode_errors = []
                for stream_mode in self._iter_stream_modes():
                    try:
                        payload = self._build_prune_payload(
                            state=state,
                            evaluated_branches=branches,
                            max_tokens=token_budget,
                            stream=stream_mode,
                        )
                        content = self._request_content(
                            session=session,
                            headers=headers,
                            payload=payload,
                            stream=stream_mode,
                            timeout=self._resolve_timeout(stream_mode),
                        )
                        prune_result = self._parse_prune_result(content, branches)
                        elapsed = time.time() - start
                        return prune_result, {"elapsed_sec": float(elapsed), "attempt": int(attempt + 1)}
                    except Exception as exc:                                
                        parse_error = exc
                        mode_errors.append(
                            "{}: {}".format("stream" if stream_mode else "non_stream", str(exc)[:300])
                        )
                        continue
                if parse_error is not None:
                    if len(mode_errors) > 0:
                        parse_error = TOTAPIError("{}; mode_errors={}".format(
                            str(parse_error), " | ".join(mode_errors[:3])
                        ))
                    raise parse_error
                if len(mode_errors) > 0:
                    raise TOTAPIError("no stream mode succeeded for prune_and_evolve; mode_errors={}".format(
                        " | ".join(mode_errors[:3])
                    ))
                raise TOTAPIError("no stream mode succeeded for prune_and_evolve")
            except Exception as exc:                                
                last_error = exc
                if attempt < self.retry:
                    if self._is_transient_error(exc):
                        self._sleep_backoff(attempt)
                    continue
        raise TOTAPIError(str(last_error))

    def _sanitize_evaluated_branches(self, evaluated_branches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean = []
        for idx, item in enumerate(evaluated_branches):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", item.get("branch_id", "branch_{}".format(idx)))).strip()
            if not name:
                name = "branch_{}".format(idx)
            weights = self._coerce_weights(item.get("weights"))
            if weights is None:
                continue
            acc_raw = item.get("regional_accuracy", None)
            acc = self._safe_float(acc_raw, np.nan)
            if np.isfinite(acc):
                regional_accuracy = float(acc)
            else:
                regional_accuracy = None
            clean.append(
                {
                    "name": name,
                    "weights": [float(v) for v in weights.tolist()],
                    "reason": str(item.get("reason", "")).strip(),
                    "regional_accuracy": regional_accuracy,
                    "proxy_reward": self._safe_float(item.get("proxy_reward", item.get("score", 0.0)), 0.0),
                }
            )
        return clean

    def probe_connectivity(self, prompt: str = "ok", timeout: float = None) -> Dict[str, Any]:
        if not self.api_url:
            raise TOTAPIError("ToT API URL is empty")
        if not self.model:
            raise TOTAPIError("ToT API model is empty")
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise TOTAPIError("missing api key env: {}".format(self.api_key_env))

        if self._use_bidi_live_api():
            request_timeout = self.timeout if timeout is None else max(1.0, float(timeout))
            content = self._request_content_bidi(
                api_key=api_key,
                system_prompt="",
                user_prompt=str(prompt or "ok"),
                max_tokens=16,
                temperature=0.0,
                top_p=1.0,
                timeout=request_timeout,
            )
            return {
                "ok": True,
                "mode": "bidi",
                "content_preview": str(content)[:120],
            }

        headers = {
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type": "application/json",
        }
        request_timeout = self.timeout if timeout is None else max(1.0, float(timeout))
        session = self._build_session()
        last_error = None
        for stream_mode in self._iter_stream_modes():
            try:
                payload = self._build_probe_payload(prompt=prompt, max_tokens=16, stream=stream_mode)
                content = self._request_content(
                    session=session,
                    headers=headers,
                    payload=payload,
                    stream=stream_mode,
                    timeout=self._resolve_timeout(stream_mode, fallback=request_timeout),
                )
                return {
                    "ok": True,
                    "mode": "stream" if stream_mode else "non_stream",
                    "content_preview": str(content)[:120],
                }
            except Exception as exc:                                
                last_error = exc
                continue
        raise TOTAPIError(str(last_error))

    def probe_connectivity_stream_first_chunk(self, prompt: str = "ok", timeout: float = None) -> Dict[str, Any]:
        if not self.api_url:
            raise TOTAPIError("ToT API URL is empty")
        if not self.model:
            raise TOTAPIError("ToT API model is empty")
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise TOTAPIError("missing api key env: {}".format(self.api_key_env))

        if self._use_bidi_live_api():
            request_timeout = self.timeout if timeout is None else max(1.0, float(timeout))
            start = time.time()
            content = self._request_content_bidi(
                api_key=api_key,
                system_prompt="",
                user_prompt=str(prompt or "ok"),
                max_tokens=16,
                temperature=0.0,
                top_p=1.0,
                timeout=request_timeout,
            )
            return {
                "ok": True,
                "mode": "bidi_first_chunk",
                "http_status": 101,
                "first_data_preview": str(content)[:200],
                "first_delta": str(content)[:200],
                "elapsed_sec": float(time.time() - start),
            }

        headers = {
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = self._build_probe_payload(prompt=prompt, max_tokens=16, stream=True)
        request_timeout = self._resolve_timeout(
            True,
            fallback=(self.timeout if timeout is None else max(1.0, float(timeout))),
        )
        session = self._build_session()
        start = time.time()
        response = session.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=request_timeout,
            stream=True,
        )
        if response.status_code != 200:
            msg = response.text[:500] if response.text else ""
            raise TOTAPIError("HTTP {} {}".format(response.status_code, msg))

        first_data = ""
        first_delta = ""
        for line in response.iter_lines():
            if line is None:
                continue
            raw_line = line.strip()
            if not raw_line:
                continue
            if raw_line.startswith(b"data:"):
                body = raw_line[5:].strip()
            else:
                body = raw_line
            if not body or body == b"[DONE]":
                continue
            decoded = body.decode("utf-8", "ignore")
            first_data = decoded[:200]
            try:
                event_json = json.loads(decoded)
                first_delta = self._extract_chunk_content(event_json)
            except Exception:
                first_delta = ""
            break

        return {
            "ok": True,
            "mode": "stream_first_chunk",
            "http_status": int(response.status_code),
            "first_data_preview": first_data,
            "first_delta": first_delta,
            "elapsed_sec": float(time.time() - start),
        }

    def _is_transient_error(self, exc: Exception) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if websocket is not None:
            ws_timeout = getattr(websocket, "WebSocketTimeoutException", None)
            ws_exception = getattr(websocket, "WebSocketException", None)
            if ws_timeout is not None and isinstance(exc, ws_timeout):
                return True
            if ws_exception is not None and isinstance(exc, ws_exception):
                text_ws = str(exc).lower()
                if ("timed out" in text_ws) or ("timeout" in text_ws) or ("429" in text_ws):
                    return True
        text = str(exc).lower()
        transient_marks = (
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "timed out",
            "connection aborted",
            "temporary",
            "temporarily unavailable",
            "gateway time-out",
            "gateway timeout",
            "request overloaded",
            "please try again later",
            "service busy",
            "system busy",
            "resource exhausted",
            "server overloaded",
            "overloaded",
            "quota",
            "try again",
            "rate limit",
            "too many requests",
            "handshake status 429",
        )
        for mark in transient_marks:
            if mark in text:
                return True
        return False

    def _sleep_backoff(self, attempt: int):
        wait = min(self.backoff_max_sec, self.backoff_base_sec * (2 ** int(attempt)))
        jitter = min(0.5, wait * 0.1)
        time.sleep(wait + random.uniform(0.0, jitter))

    def _iter_stream_modes(self) -> List[bool]:
        if self.force_non_stream:
            return [False]
        if self._is_atomgit_v5():
            return [False]
        if self.prefer_stream:
            return [True, False]
        return [False, True]

    def _resolve_timeout(self, stream_mode: bool, fallback: Optional[float] = None):
        if not stream_mode:
            if fallback is None:
                return self.timeout
            return max(1.0, float(fallback))
        connect_timeout = max(1.0, float(self.stream_connect_timeout))
        read_default = self.stream_read_timeout if fallback is None else max(1.0, float(fallback))
        read_timeout = max(1.0, float(read_default))
        return (connect_timeout, read_timeout)

    def _request_content(
        self,
        session: requests.Session,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        stream: bool,
        timeout,
    ) -> str:
        if stream and (not self._is_atomgit_v5()):
            return self._request_content_stream(session=session, headers=headers, payload=payload, timeout=timeout)
        return self._request_content_non_stream(session=session, headers=headers, payload=payload, timeout=timeout)

    def _request_content_non_stream(
        self,
        session: requests.Session,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout,
    ) -> str:
        response = session.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        if response.status_code != 200:
            msg = response.text[:500] if response.text else ""
            raise TOTAPIError("HTTP {} {}".format(response.status_code, msg))
        response_json = self._parse_response_json(response)
        return self._extract_content(response_json)

    def _request_content_stream(
        self,
        session: requests.Session,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout,
    ) -> str:
        stream_headers = dict(headers)
        stream_headers["Accept"] = "text/event-stream"
        response = session.post(
            self.api_url,
            headers=stream_headers,
            json=payload,
            timeout=timeout,
            stream=True,
        )
        if response.status_code != 200:
            msg = response.text[:500] if response.text else ""
            raise TOTAPIError("HTTP {} {}".format(response.status_code, msg))

        chunks = []
        last_event = None
        for line in response.iter_lines():
            if line is None:
                continue
            raw_line = line.strip()
            if not raw_line:
                continue
            if raw_line.startswith(b"data:"):
                body = raw_line[5:].strip()
            else:
                body = raw_line
            if not body:
                continue
            if body == b"[DONE]":
                break
            try:
                event_json = json.loads(body.decode("utf-8", "ignore"))
            except Exception:
                continue
            last_event = event_json
            chunk = self._extract_chunk_content(event_json)
            if chunk:
                chunks.append(chunk)

        if len(chunks) > 0:
            return "".join(chunks)
        if isinstance(last_event, dict):
            try:
                return self._extract_content(last_event)
            except Exception:
                pass
        raise TOTAPIError("empty streamed content in API response")

    def _content_obj_to_text(self, content_obj: Any) -> str:
        if isinstance(content_obj, str):
            return content_obj
        if isinstance(content_obj, dict):
            keys = ("text", "content", "reasoning_content", "output_text", "reasoning")
            parts = []
            for key in keys:
                val = content_obj.get(key, "")
                if isinstance(val, str) and val:
                    parts.append(val)
                elif isinstance(val, list):
                    nested = self._content_obj_to_text(val)
                    if nested:
                        parts.append(nested)
            return "".join(parts)
        if isinstance(content_obj, list):
            parts = []
            for item in content_obj:
                if isinstance(item, dict):
                    nested = self._content_obj_to_text(item)
                    if nested:
                        parts.append(nested)
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return ""

    def _extract_chunk_content(self, event_json: Dict[str, Any]) -> str:
        if not isinstance(event_json, dict):
            return ""
        choices = event_json.get("choices", [])
        if not isinstance(choices, list) or len(choices) <= 0:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta", {})
        if isinstance(delta, dict):
            text = self._content_obj_to_text(delta.get("content", ""))
            if text:
                return text
            text = self._content_obj_to_text(delta.get("reasoning_content", ""))
            if text:
                return text
            text = self._content_obj_to_text(delta.get("reasoning", ""))
            if text:
                return text
            text = self._content_obj_to_text(delta.get("output_text", ""))
            if text:
                return text
        message = first.get("message", {})
        if isinstance(message, dict):
            text = self._content_obj_to_text(message.get("content", ""))
            if text:
                return text
            text = self._content_obj_to_text(message.get("reasoning_content", ""))
            if text:
                return text
            text = self._content_obj_to_text(message.get("reasoning", ""))
            if text:
                return text
            text = self._content_obj_to_text(message.get("output_text", ""))
            if text:
                return text
        if isinstance(first.get("text"), str):
            return str(first.get("text"))
        text = self._content_obj_to_text(first.get("reasoning", ""))
        if text:
            return text
        text = self._content_obj_to_text(first.get("reasoning_content", ""))
        if text:
            return text
        text = self._content_obj_to_text(first.get("output_text", ""))
        if text:
            return text
        return ""

    def _extract_content(self, response_json: Dict[str, Any]) -> str:
        choices = response_json.get("choices", [])
        if len(choices) <= 0:
            raise TOTAPIError("missing choices in API response")
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {}) if isinstance(first, dict) else {}
        if isinstance(message, str):
            content = message
        else:
            content = self._content_obj_to_text(message.get("content", ""))
            if not content and isinstance(message, dict):
                content = self._content_obj_to_text(message)
        if not content:
            delta = first.get("delta", {}) if isinstance(first, dict) else {}
            content = self._content_obj_to_text(delta.get("content", ""))
        if not content:
            delta = first.get("delta", {}) if isinstance(first, dict) else {}
            content = self._content_obj_to_text(delta)
        if not content and isinstance(first, dict):
            content = first.get("text", "")
        if not content and isinstance(first, dict):
            content = self._content_obj_to_text(first.get("reasoning_content", ""))
        if not content and isinstance(first, dict):
            content = self._content_obj_to_text(first.get("reasoning", ""))
        if not content and isinstance(first, dict):
            content = self._content_obj_to_text(first.get("output_text", ""))
        if not content and isinstance(response_json, dict):
            direct_text = response_json.get("text", "")
            if isinstance(direct_text, str) and direct_text.strip():
                content = direct_text
        if not content and isinstance(response_json, dict):
            content = self._content_obj_to_text(response_json.get("reasoning_content", ""))
        if not content and isinstance(response_json, dict):
            content = self._content_obj_to_text(response_json.get("reasoning", ""))
        if not content and isinstance(response_json, dict):
            output_text = response_json.get("output_text", "")
            if isinstance(output_text, str) and output_text.strip():
                content = output_text
        if not isinstance(content, str) or len(content.strip()) <= 0:
            raise TOTAPIError("empty content in API response")
        return content

    def _parse_response_json(self, response: requests.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except Exception:
            raw = response.content or b""
            text = ""
            if isinstance(raw, (bytes, bytearray)) and len(raw) > 0:
                text = raw.decode("utf-8", "ignore").strip()
            if not text:
                text = (response.text or "").strip()
            if not text:
                raise TOTAPIError("empty HTTP response body")
            if text.startswith("data:"):
                events = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == "[DONE]":
                        continue
                    try:
                        events.append(json.loads(body))
                    except Exception:
                        continue
                if len(events) <= 0:
                    raise TOTAPIError("invalid SSE payload: {}".format(text[:300]))
                                 
                if len(events) > 1:
                    acc = []
                    for evt in events:
                        choices = evt.get("choices", [])
                        if not choices:
                            continue
                        first = choices[0] if isinstance(choices[0], dict) else {}
                        delta = first.get("delta", {}) if isinstance(first, dict) else {}
                        chunk = self._content_obj_to_text(delta.get("content", ""))
                        if not chunk:
                            chunk = self._content_obj_to_text(delta.get("reasoning_content", ""))
                        if not chunk:
                            chunk = self._content_obj_to_text(delta.get("reasoning", ""))
                        if not chunk:
                            msg = first.get("message", {}) if isinstance(first, dict) else {}
                            chunk = self._content_obj_to_text(msg)
                        if chunk:
                            acc.append(str(chunk))
                    if len(acc) > 0:
                        return {"choices": [{"message": {"content": "".join(acc)}}]}
                                                                                       
                if isinstance(events[-1], dict):
                    return events[-1]
            raise TOTAPIError("response body is not JSON: {}".format(text[:300]))

    def _repair_candidates_via_reformat(
        self,
        session: requests.Session,
        headers: Dict[str, str],
        raw_content: str,
        Q: int,
    ) -> List[Dict[str, Any]]:
        raw_trim = (raw_content or "").strip()
        if len(raw_trim) <= 0:
            return []
        user_prompt = (
            "Convert the following text into strict JSON only.\n"
            "Return exactly this format:\n"
            "{{\"candidates\":[{{\"name\":\"branch_1\",\"weights\":{{\"data\":0.0,\"modal\":0.0,\"perf\":0.0,\"res\":0.0,\"cool\":0.0,\"fair\":0.0}},\"reason\":\"...\"}}]}}\n"
            "Rules:\n"
            "1) Return exactly {} candidates.\n"
            "2) weights has six non-negative values in [0,1], sum=1.\n"
            "3) No markdown, no thinking process, no extra text.\n\n"
            "Source text:\n{}"
        ).format(int(Q), raw_trim[:6000])
        payload = self._build_chat_payload(
            system_prompt="You are a strict JSON formatter. Output JSON only.",
            user_prompt=user_prompt,
            max_tokens=max(self.max_tokens, 1024),
            temperature=0.0,
            top_p=1.0,
        )
        response = session.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
        if response.status_code != 200:
            return []
        response_json = self._parse_response_json(response)
        content = self._extract_content(response_json)
        parsed = self._parse_candidates(content, Q)
        if len(parsed) <= 0:
            parsed = self._parse_candidates_from_text_fallback(content, Q)
        if len(parsed) <= 0:
            parsed = self._parse_candidates_from_named_weights(content, Q)
        return parsed

    def _build_prompt(self, state: Dict[str, Any], Q: int) -> str:
        state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        return (
            "Generate ToT candidate branches for MMQS 6D scoring from current FL state.\n"
            "State JSON:\n{}\n"
            "Weight semantics (fixed order):\n"
            "- w1=w_data, w2=w_modal, w3=w_perf, w4=w_res, w5=w_cool, w6=w_fair\n"
            "State parameter meanings and effects:\n"
            "- regional_accuracy: current regional validation accuracy (higher is better).\n"
            "- stage_ratio: training progress in [0,1+] (early<0.33, mid 0.33-0.66, late>0.66).\n"
            "- fairness_gap: participation/quality gap across clients (higher means less fair).\n"
            "- delay_cv: delay coefficient of variation (higher means less stable latency).\n"
            "- component_mean.s_data/s_modal/s_perf/s_res/s_cool/s_fair: current average component scores.\n"
            "Weight action guide:\n"
            "- If fairness_gap is high, increase w_fair.\n"
            "- If delay_cv is high, increase w_res and/or w_cool.\n"
            "- If regional_accuracy is low in early stage, prioritize w_data and w_perf.\n"
            "- Keep exploration but avoid extreme single-dimension dominance.\n"
            "Output contract (strict JSON only):\n"
            "{{\"branches\":[{{\"branch_id\":\"b1\",\"reasoning\":\"short reason\",\"weights\":[w1,w2,w3,w4,w5,w6]}}]}}\n"
            "Hard constraints:\n"
            "1) Return exactly {} branches.\n"
            "2) branch_id must be unique.\n"
            "3) Every weights list has exactly 6 numbers.\n"
            "4) Each weight is in [0,1], non-negative.\n"
            "5) Sum(weights)=1 (normalize if needed).\n"
            "6) Use only regional signals from State JSON; never use global metrics.\n"
            "7) No markdown, no prose outside JSON, no chain-of-thought.".format(state_json, int(Q))
        )

    def _build_prune_prompt(self, state: Dict[str, Any], evaluated_branches: List[Dict[str, Any]]) -> str:
        state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        branches_json = json.dumps(evaluated_branches, ensure_ascii=False, separators=(",", ":"))
        return (
            "Perform feedback-driven active pruning for ToT branches of MMQS 6D weights.\n"
            "State JSON:\n{}\n"
            "Evaluated branches JSON:\n{}\n"
            "Rules:\n"
            "1) Use regional_accuracy as the only empirical reward signal.\n"
            "2) Never use or infer global metrics.\n"
            "3) If a branch has regional_accuracy=null, treat it as not executed; do not hallucinate reward.\n"
            "4) Prefer the winner with higher regional_accuracy; if tie, prefer better fairness/latency tradeoff inferred from state.\n"
            "5) next_seed_weights must contain 6 values in [0,1] and sum to 1.\n"
            "6) Output strict JSON only, no markdown, no extra text.\n"
            "Return exactly this format:\n"
            "{{\"pruned_branch_id\":\"...\",\"winner_branch_id\":\"...\",\"next_exploration_direction\":\"...\","
            "\"next_seed_weights\":[w1,w2,w3,w4,w5,w6]}}"
        ).format(state_json, branches_json)

    def _parse_candidates(self, content: str, Q: int) -> List[Dict[str, Any]]:
        obj = self._load_json_payload(content)
        if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
            candidate_items = obj.get("candidates", [])
        elif isinstance(obj, dict) and isinstance(obj.get("branches"), list):
            candidate_items = obj.get("branches", [])
        elif isinstance(obj, list):
            candidate_items = obj
        elif isinstance(obj, dict):
            candidate_items = [obj]
        else:
            candidate_items = []

        parsed = []
        for idx, item in enumerate(candidate_items):
            if not isinstance(item, dict):
                continue
            weights = self._coerce_weights(item.get("weights"))
            if weights is None:
                continue
            branch_name = item.get("name", item.get("branch_id", "branch_{}".format(idx)))
            branch_reason = item.get("reason", item.get("reasoning", ""))
            parsed.append(
                {
                    "name": str(branch_name).strip() or "branch_{}".format(idx),
                    "weights": weights,
                    "reason": str(branch_reason).strip(),
                    "score": self._safe_float(item.get("score", 0.0), 0.0),
                    "parse_source": "strict_json",
                }
            )
            if len(parsed) >= Q:
                break
        return parsed

    def _parse_prune_result(
        self,
        content: str,
        evaluated_branches: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        obj = self._load_json_payload(content)
        if isinstance(obj, list):
            obj = obj[0] if len(obj) > 0 else {}
        if not isinstance(obj, dict):
            raise TOTAPIError("prune response is not a JSON object")

        branch_map = {str(item.get("name", "")).strip(): item for item in evaluated_branches}
        if len(branch_map) <= 0:
            raise TOTAPIError("empty branch map in prune parser")

        winner = str(
            obj.get(
                "winner_branch_id",
                obj.get("winner_branch", obj.get("winner", "")),
            )
        ).strip()
        if winner not in branch_map:
            winner = self._pick_best_branch_name(evaluated_branches)

        pruned = str(
            obj.get(
                "pruned_branch_id",
                obj.get("pruned_branch", obj.get("drop_branch_id", "")),
            )
        ).strip()
        if pruned not in branch_map or pruned == winner:
            pruned = self._pick_worst_branch_name(evaluated_branches, exclude=winner)

        direction = str(
            obj.get("next_exploration_direction", obj.get("direction", ""))
        ).strip()
        if not direction:
            direction = "regional_feedback_refine"

        seed_weights = self._coerce_weights(obj.get("next_seed_weights", obj.get("seed_weights")))
        if seed_weights is None and winner in branch_map:
            seed_weights = self._coerce_weights(branch_map[winner].get("weights"))
        if seed_weights is None:
            seed_weights = self._coerce_weights(evaluated_branches[0].get("weights"))
        if seed_weights is None:
            raise TOTAPIError("failed to parse next seed weights")

        return {
            "pruned_branch_id": pruned,
            "winner_branch_id": winner,
            "next_exploration_direction": direction,
            "next_seed_weights": [float(v) for v in seed_weights.tolist()],
        }

    def _pick_best_branch_name(self, evaluated_branches: List[Dict[str, Any]]) -> str:
        best_name = str(evaluated_branches[0].get("name", "branch_0"))
        best_score = -1e30
        for item in evaluated_branches:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            acc = item.get("regional_accuracy", None)
            acc_val = self._safe_float(acc, np.nan)
            if np.isfinite(acc_val):
                score = 1000.0 + float(acc_val)
            else:
                score = self._safe_float(item.get("proxy_reward", item.get("score", 0.0)), 0.0)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name

    def _pick_worst_branch_name(self, evaluated_branches: List[Dict[str, Any]], exclude: str = "") -> str:
        worst_name = ""
        worst_score = 1e30
        for item in evaluated_branches:
            name = str(item.get("name", "")).strip()
            if not name or name == exclude:
                continue
            acc = item.get("regional_accuracy", None)
            acc_val = self._safe_float(acc, np.nan)
            if np.isfinite(acc_val):
                score = float(acc_val)
            else:
                score = self._safe_float(item.get("proxy_reward", item.get("score", 0.0)), 0.0)
            if score < worst_score:
                worst_score = score
                worst_name = name
        if worst_name:
            return worst_name
                                          
        for item in evaluated_branches:
            name = str(item.get("name", "")).strip()
            if name and name != exclude:
                return name
        return exclude

    def _coerce_weights(self, weights_obj: Any):
        if isinstance(weights_obj, dict):
            key_alias = {
                "data": ("data", "s_data", "w_data"),
                "modal": ("modal", "s_modal", "w_modal"),
                "perf": ("perf", "s_perf", "w_perf"),
                "res": ("res", "s_res", "w_res"),
                "cool": ("cool", "s_cool", "w_cool"),
                "fair": ("fair", "s_fair", "w_fair"),
            }
            vec = []
            for key in self.WEIGHT_KEYS:
                value = None
                for alias in key_alias[key]:
                    if alias in weights_obj:
                        value = weights_obj.get(alias)
                        break
                vec.append(self._safe_float(value, 0.0))
            arr = np.array(vec, dtype=float)
        elif isinstance(weights_obj, (list, tuple, np.ndarray)) and len(weights_obj) >= len(self.WEIGHT_KEYS):
            arr = np.array([self._safe_float(v, 0.0) for v in weights_obj[: len(self.WEIGHT_KEYS)]], dtype=float)
        else:
            return None

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.maximum(arr, 0.0)
        total = float(np.sum(arr))
        if total <= 1e-12:
            return None
        return arr / total

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(val):
            return float(default)
        return float(val)

    def _load_json_payload(self, content: str) -> Any:
        text = str(content).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```"):
                text = "\n".join(lines[1:-1]).strip()
                if text.startswith("json"):
                    text = text[4:].strip()
        try:
            return json.loads(text)
        except Exception:                                
            pass

        decoder = json.JSONDecoder()
        parsed_items = []
        for idx, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                obj, end = decoder.raw_decode(text[idx:])
                parsed_items.append((idx, idx + end, obj))
            except Exception:
                continue

        if len(parsed_items) > 0:
                                                                                          
            for _, _, obj in reversed(parsed_items):
                if isinstance(obj, dict) and (
                    isinstance(obj.get("candidates"), list) or isinstance(obj.get("branches"), list)
                ):
                    return obj
            return parsed_items[-1][2]
        raise TOTAPIError("model output is not valid JSON")

    def _parse_candidates_from_text_fallback(self, content: str, Q: int) -> List[Dict[str, Any]]:
        text = str(content or "")
        if len(text.strip()) <= 0:
            return []

        def _coerce_from_number_seq(nums_seq):
            vals = [max(0.0, self._safe_float(v, 0.0)) for v in list(nums_seq)[: len(self.WEIGHT_KEYS)]]
            if len(vals) < len(self.WEIGHT_KEYS):
                remain = max(0.0, 1.0 - float(sum(vals)))
                if remain > 0 and len(vals) < len(self.WEIGHT_KEYS):
                    vals.append(remain)
                while len(vals) < len(self.WEIGHT_KEYS):
                    vals.append(0.0)
            return self._coerce_weights(vals)

        parsed = []
                                                                                   
        template_matches = re.findall(r'(?im)^\s*W\s*=\s*\[([^\]]+)\]\s*$', text)
        if len(template_matches) <= 0:
            template_matches = re.findall(r'(?im)^\s*weights\s*=\s*\[([^\]]+)\]\s*$', text)
        for idx, block in enumerate(template_matches):
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", block)
            if len(nums) < len(self.WEIGHT_KEYS):
                continue
            weights = _coerce_from_number_seq(nums)
            if weights is None:
                continue
            parsed.append(
                {
                    "name": "template_line_{}".format(idx),
                    "weights": weights,
                    "reason": "strict template parse from W=[...] line",
                    "score": 0.0,
                    "parse_source": "strict_template",
                }
            )
            if len(parsed) >= Q:
                break
        if len(parsed) > 0:
            return parsed

                                                            
        list_matches = re.findall(r'weights"\s*:\s*\[([^\]]+)\]', text, flags=re.IGNORECASE)
                                                                         
        if len(list_matches) <= 0:
            list_matches = re.findall(r"\[([0-9eE\+\-\.,\s]{8,})\]", text)

        for idx, block in enumerate(list_matches):
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", block)
            if len(nums) < len(self.WEIGHT_KEYS):
                continue
            weights = _coerce_from_number_seq(nums)
            if weights is None:
                continue
            parsed.append(
                {
                    "name": "text_fallback_{}".format(idx),
                    "weights": weights,
                    "reason": "fallback parse from raw text",
                    "score": 0.0,
                    "parse_source": "text_fallback_list",
                }
            )
            if len(parsed) >= Q:
                break
        if len(parsed) > 0:
            return parsed

                                                                                                           
        partial_blocks = re.findall(r'(?i)weights\s*"\s*:\s*\[([^\]\n\r]*)', text)
        if len(partial_blocks) <= 0:
            partial_blocks = re.findall(r"(?i)weights\s*:\s*\[([^\]\n\r]*)", text)
        for idx, block in enumerate(partial_blocks):
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", block)
            if len(nums) <= 0:
                continue
            weights = _coerce_from_number_seq(nums)
            if weights is None:
                continue
            parsed.append(
                {
                    "name": "text_partial_weights_{}".format(idx),
                    "weights": weights,
                    "reason": "fallback parse from truncated weights list",
                    "score": 0.0,
                    "parse_source": "text_partial_weights",
                }
            )
            if len(parsed) >= Q:
                break
        if len(parsed) > 0:
            return parsed

                                                                        
                                                             
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if len(nums) >= len(self.WEIGHT_KEYS):
            nums_float = [self._safe_float(v, 0.0) for v in nums]
            nums_01 = [v for v in nums_float if 0.0 <= v <= 1.0]
            candidate_nums = nums_01[-len(self.WEIGHT_KEYS):] if len(nums_01) >= len(self.WEIGHT_KEYS) else nums_float[-len(self.WEIGHT_KEYS):]
            weights = _coerce_from_number_seq(candidate_nums)
            if weights is not None:
                reason = "fallback parse from numeric tail in raw text"
                weights_arr = np.array(weights, dtype=float)
                                                                                                             
                if float(np.max(weights_arr)) > 0.85:
                    weights_arr = 0.8 * weights_arr + 0.2 * (np.ones((len(self.WEIGHT_KEYS),), dtype=float) / float(len(self.WEIGHT_KEYS)))
                    weights_arr = weights_arr / float(np.sum(weights_arr))
                    weights = weights_arr
                    reason = "fallback parse from numeric tail in raw text (smoothed)"
                parsed.append(
                    {
                        "name": "numeric_tail_fallback_0",
                        "weights": weights,
                        "reason": reason,
                        "score": 0.0,
                        "parse_source": "numeric_tail_fallback",
                    }
                )
        return parsed

    def _parse_candidates_from_named_weights(self, content: str, Q: int) -> List[Dict[str, Any]]:
        text = str(content or "")
        if len(text.strip()) <= 0:
            return []

        alias_map = {
            "data": ["data", "s_data", "w_data", "mmqs_w_data"],
            "modal": ["modal", "s_modal", "w_modal", "mmqs_w_modal"],
            "perf": ["perf", "s_perf", "w_perf", "mmqs_w_perf"],
            "res": ["res", "s_res", "w_res", "mmqs_w_res"],
            "cool": ["cool", "s_cool", "w_cool", "mmqs_w_cool"],
            "fair": ["fair", "s_fair", "w_fair", "mmqs_w_fair"],
        }
        number_pat = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

        key_to_value = {}
        for key in self.WEIGHT_KEYS:
            aliases = alias_map.get(key, [key])
            for alias in aliases:
                escaped = re.escape(alias)
                patterns = [
                    r'(?i)(?:[`"\']?\b{}\b[`"\']?)\s*[:=]\s*{}'.format(escaped, number_pat),
                    r'(?i)(?:[`"\']?\b{}\b[`"\']?)\s+{}'.format(escaped, number_pat),
                ]
                hit = None
                for pattern in patterns:
                    m = re.search(pattern, text)
                    if m:
                        hit = m
                        break
                if hit:
                    key_to_value[key] = self._safe_float(hit.group(1), 0.0)
                    break

        if len(key_to_value) <= 0:
            return []

                                                                    
        default_map = {
            "data": 0.35,
            "modal": 0.00,
            "perf": 0.35,
            "res": 0.10,
            "cool": 0.10,
            "fair": 0.10,
        }
        vec = [self._safe_float(key_to_value.get(key, default_map[key]), default_map[key]) for key in self.WEIGHT_KEYS]
        weights = self._coerce_weights(vec)
        if weights is None:
            return []

        return [
            {
                "name": "named_weights_fallback",
                "weights": weights,
                "reason": "fallback parse from named weights in raw text",
                "score": 0.0,
                "parse_source": "named_weights_fallback",
            }
        ][: max(1, int(Q))]
