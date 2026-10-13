#!/usr/bin/env python3
"""Reusable Vast.ai manager for the video-subtitle-remover API deployment.

Reads the API key from the VAST_API_KEY environment variable.
"""

import argparse
import datetime
import json
import os
import sys
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Set

import requests

BASE_URL = "https://console.vast.ai/api/v0"
BASE_URL_V1 = "https://console.vast.ai/api/v1"
DOCKER_REPO = "liudunxu/video-subtitle-remover"
DOCKER_TAG = "vast-gpu"
LABEL_PREFIX = "video-subtitle-remover-api-"
DEFAULT_LABEL = "video-subtitle-remover-api-mvp"
DEFAULT_PORT = 6006
DEFAULT_DISK = 70
DEFAULT_GPU = "RTX 3090"
# GPU 型号偏好顺序（对齐 OmniVoice 的逐级降级模式）：未显式 --gpu 时
# 按此顺序搜索，当前型号没有可用单卡 offer 才降级到下一个。
GPU_PREFERENCE = ["RTX 3090 Ti", "RTX 3090", "RTX 4090"]
DEFAULT_MIN_VRAM_MB = 12000
KNOWN_GOOD_FILE = os.path.join(os.path.dirname(__file__), "vast_known_good.json")

# Preferred CPU families for the current image stack (PyTorch 2.8 + PaddlePaddle 3.0).
# These CPUs are known to support AVX2/FMA and avoid the Illegal instruction crashes
# seen on very old Intel Xeon E5 v2/v3 and generic Common KVM processor labels.
_PREFERRED_CPU_SUBSTRINGS = [
    "AMD Ryzen",
    "AMD EPYC 7",
    "AMD EPYC 72",
    "AMD EPYC 73",
    "AMD EPYC 74",
    "AMD EPYC 75",
    "AMD EPYC 76",
    "AMD EPYC 77",
    "Xeon E5-2673 v4",
    "Xeon E5-2686 v4",
    "Xeon E5-2690 v4",
    "Xeon E5-2695 v4",
    "Xeon E5-2696 v4",
    "Xeon E5-2697 v4",
    "Xeon W-",
    "Xeon Gold",
    "Xeon Silver",
    "Core i5-",
    "Core i7-",
    "Core i9-",
]

# Region priority for video-subtitle-remover-api deployments (aligned with
# OmniVoice): Southeast Asia first, then East Asia (KR/TW/JP/HK), then other
# Asia, then Americas. A previous Oman (West Asia) pick downloaded at ~6KB/s.
_SOUTHEAST_ASIA_COUNTRY_CODES = {"SG", "TH", "VN", "MY", "ID", "PH", "KH", "MM", "LA", "BN", "TL"}
_EAST_ASIA_COUNTRY_CODES = {"KR", "TW", "JP", "HK", "MO"}
_OTHER_ASIA_COUNTRY_CODES = {"CN", "IN", "TR", "AE", "SA", "IL", "QA", "KW", "BH", "OM"}
_AMERICAS_COUNTRY_CODES = {"US", "CA", "MX", "BR", "AR", "CL", "CO", "PE"}

REQUIRED_ENV = {
    f"-p {DEFAULT_PORT}:{DEFAULT_PORT}": "1",
    "HOST": "0.0.0.0",
    "PORT": str(DEFAULT_PORT),
    "VSR_REQUIRE_GPU": "1",
    "VSR_API_WORK_DIR": "/workspace/video_subtitle_remover_api",
    "XDG_CACHE_HOME": "/workspace/.cache",
    "MPLCONFIGDIR": "/workspace/.cache/matplotlib",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
}


def _api_key() -> str:
    key = os.environ.get("VAST_API_KEY")
    if not key:
        raise SystemExit("Error: VAST_API_KEY environment variable is required.")
    return key


def _headers(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def latest_digest() -> str:
    """Resolve the current linux/amd64 digest for the configured Docker tag."""
    url = f"https://hub.docker.com/v2/repositories/{DOCKER_REPO}/tags/{DOCKER_TAG}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    digest = data.get("digest")
    if not digest and "images" in data:
        for img in data["images"]:
            if img.get("architecture") == "amd64":
                digest = img.get("digest")
                break
        if not digest:
            digest = data["images"][0].get("digest")
    if not digest:
        _die("Could not resolve Docker image digest from Docker Hub.")
    return digest


def list_instances(key: str, cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """List instances using the v1 endpoint (v0 /instances/ is deprecated)."""
    select_cols = urllib.parse.quote(json.dumps(cols or ["id", "label", "actual_status", "cur_state", "status_msg", "ports", "public_ipaddr"]))
    url = f"{BASE_URL_V1}/instances/?select_cols={select_cols}&limit=50"
    resp = requests.get(url, headers=_headers(key), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("instances", []) or []


def matching_instances(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [i for i in instances if (i.get("label") or "").startswith(LABEL_PREFIX)]


def destroy_instance(key: str, instance_id: int) -> Dict[str, Any]:
    resp = requests.delete(
        f"{BASE_URL}/instances/{instance_id}/", headers=_headers(key), timeout=60
    )
    resp.raise_for_status()
    return resp.json()


def search_offers(
    key: str,
    gpu_name: str = DEFAULT_GPU,
    min_vram_mb: int = DEFAULT_MIN_VRAM_MB,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    payload = {
        "gpu_name": {"in": [gpu_name]},
        # 只选单卡机：本服务只用一张 GPU，多卡整机属于白花钱
        # （此前是 {"gte": 1}，曾因此选中一台 4x3090）。
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": min_vram_mb},
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "type": "ondemand",
        "limit": limit,
    }
    resp = requests.post(
        f"{BASE_URL}/bundles/", headers=_headers(key), json=payload, timeout=60
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("offers", []) or []


def create_instance(
    key: str,
    offer_id: int,
    image: str,
    label: str = DEFAULT_LABEL,
    disk: int = DEFAULT_DISK,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "image": image,
        "label": label,
        "disk": disk,
        "runtype": "args",
        "target_state": "running",
        "env": env or {},
    }
    resp = requests.put(
        f"{BASE_URL}/asks/{offer_id}/", headers=_headers(key), json=body, timeout=60
    )
    resp.raise_for_status()
    return resp.json()


def get_instance(key: str, instance_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single instance via the v1 endpoint using select_filters."""
    filters = urllib.parse.quote(json.dumps({"id": {"eq": instance_id}}))
    cols = urllib.parse.quote(json.dumps([
        "id", "label", "actual_status", "cur_state", "status_msg", "ports",
        "public_ipaddr", "dph_total", "gpu_name", "disk_usage"
    ]))
    url = f"{BASE_URL_V1}/instances/?select_filters={filters}&select_cols={cols}&limit=5"
    resp = requests.get(url, headers=_headers(key), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    instances = data.get("instances") or []
    return instances[0] if instances else None


def instance_logs(key: str, instance_id: int, tail: str = "800") -> str:
    """Request instance logs and fetch them from the returned S3 URL.

    Vast.ai logs are async: the PUT returns a result_url that must be
    polled briefly before it becomes available.
    """
    resp = requests.put(
        f"{BASE_URL}/instances/request_logs/{instance_id}",
        headers=_headers(key),
        json={"tail": tail},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    result_url = data.get("result_url")
    if not result_url:
        raise RuntimeError(f"No result_url in log response: {data}")

    deadline = time.time() + 30
    last_error: Optional[Exception] = None
    while time.time() < deadline:
        try:
            log_resp = requests.get(result_url, timeout=30)
            log_resp.raise_for_status()
            return log_resp.text
        except requests.RequestException as e:
            last_error = e
            time.sleep(2)
    raise RuntimeError(f"Could not fetch logs from {result_url}: {last_error}")


def _public_url(instance: Dict[str, Any]) -> Optional[str]:
    ports = instance.get("ports", {}) or {}
    public_ip = instance.get("public_ipaddr") or instance.get("public_ip")
    if not public_ip:
        return None

    # Vast.ai exposes ports as {"6006/tcp": [{"HostIp": "...", "HostPort": "..."}]}
    for key, mappings in ports.items():
        if key in (f"{DEFAULT_PORT}/tcp", str(DEFAULT_PORT)):
            if isinstance(mappings, list) and mappings:
                host_port = mappings[0].get("HostPort")
                if host_port:
                    return f"http://{public_ip}:{host_port}"
    return None


def load_known_good(path: str = KNOWN_GOOD_FILE) -> List[Dict[str, Any]]:
    """Load the known-good instance ledger from JSON."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("instances", []) or []
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: could not read known-good file {path}: {e}", file=sys.stderr)
        return []


def save_known_good(instances: List[Dict[str, Any]], path: str = KNOWN_GOOD_FILE) -> None:
    """Persist the known-good instance ledger to JSON."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"instances": instances}, f, indent=2, ensure_ascii=False)
        print(f"Updated known-good instances in {path}")
    except OSError as e:
        print(f"Warning: could not write known-good file {path}: {e}", file=sys.stderr)


def _build_known_good_entry(instance_id: int, offer: Dict[str, Any], image: str) -> Dict[str, Any]:
    """Build a ledger entry for a successfully started instance."""
    return {
        "instance_id": instance_id,
        "machine_id": offer.get("machine_id"),
        "gpu_name": offer.get("gpu_name"),
        "cpu_name": offer.get("cpu_name"),
        "country_code": offer.get("country_code"),
        "region": _offer_region(offer),
        "image": image,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "notes": "GPU runtime check passed",
    }


def _offer_region(offer: Dict[str, Any]) -> str:
    country = (offer.get("country_code") or "").upper()
    if not country:
        # vast 的 geolocation 形如 "City, CC"，取逗号后的国家码精确匹配，
        # 避免子串误伤（如 "in" 命中 "Singapore"）。
        geo = (offer.get("geolocation") or "")
        country = geo.rsplit(",", 1)[-1].strip().upper() if geo else ""
    if country in _SOUTHEAST_ASIA_COUNTRY_CODES:
        return "southeast_asia"
    if country in _EAST_ASIA_COUNTRY_CODES:
        return "east_asia"
    if country in _OTHER_ASIA_COUNTRY_CODES:
        return "other_asia"
    if country in _AMERICAS_COUNTRY_CODES:
        return "americas"
    return "other"


def _cpu_score(offer: Dict[str, Any]) -> int:
    """Prefer known-good modern CPUs; penalise generic/unknown/old CPUs."""
    cpu = str(offer.get("cpu_name") or "").lower()
    if not cpu or cpu == "common kvm processor":
        return 100
    if any(sub.lower() in cpu for sub in _PREFERRED_CPU_SUBSTRINGS):
        return 0
    return 50


def _score_offer(offer: Dict[str, Any], known_good_machine_ids: Optional[Set[int]] = None) -> tuple:
    # Prefer known-good machines, then SEA > East Asia > other Asia > Americas,
    # then known-good CPUs, then reliability, then lower price; fewer CPU cores
    # as the final tie-break (同等条件下少核机器，少占资源碎片).
    machine_id = offer.get("machine_id")
    is_known_good = bool(known_good_machine_ids and machine_id in known_good_machine_ids)
    region_rank = {
        "southeast_asia": 0,
        "east_asia": 1,
        "other_asia": 2,
        "americas": 3,
        "other": 4,
    }.get(_offer_region(offer), 4)
    cpu_rank = _cpu_score(offer)
    reliability = offer.get("reliability", 0) or 0
    dph = offer.get("dph_total", float("inf")) or float("inf")
    cpu_cores = offer.get("cpu_cores_effective") or offer.get("cpu_cores") or 0
    return (0 if is_known_good else 1, region_rank, cpu_rank, -reliability, dph, cpu_cores)


def cmd_status(key: str, args: argparse.Namespace) -> int:
    instances = list_instances(key)
    matches = matching_instances(instances)
    if not matches:
        print(f"No instances with label prefix '{LABEL_PREFIX}' found.")
        return 0

    print(f"{'ID':>8}  {'Label':<36}  {'Status':<12}  {'Public URL'}")
    for i in matches:
        iid = i.get("id") or "?"
        url = _public_url(i) or "-"
        print(
            f"{iid:>8}  "
            f"{i.get('label', ''):<36}  "
            f"{i.get('actual_status', '?'):<12}  "
            f"{url}"
        )
    return 0


def cmd_stop(key: str, args: argparse.Namespace) -> int:
    instances = list_instances(key)
    matches = matching_instances(instances)
    if not matches:
        print(f"No instances with label prefix '{LABEL_PREFIX}' to delete.")
        return 0

    print(f"Deleting {len(matches)} matching instance(s)...")
    for i in matches:
        iid = i["id"]
        label = i.get("label", "")
        status = i.get("actual_status", "?")
        print(f"  Deleting {iid} ({label}, status={status}) ...")
        try:
            destroy_instance(key, iid)
            print(f"  Deleted {iid}.")
        except requests.HTTPError as e:
            print(f"  Failed to delete {iid}: {e}", file=sys.stderr)

    # Verify
    remaining = matching_instances(list_instances(key))
    if remaining:
        print("WARNING: Some matching instances remain:", file=sys.stderr)
        for i in remaining:
            print(f"  {i['id']} {i.get('label')}", file=sys.stderr)
        return 1
    print("All matching instances deleted.")
    return 0


def cmd_start(key: str, args: argparse.Namespace) -> int:
    image = args.image
    if not image:
        digest = latest_digest()
        image = f"{DOCKER_REPO}@{digest}"
        print(f"Resolved latest image: {image}")

    # 1. Check existing
    instances = list_instances(key)
    matches = matching_instances(instances)
    running = [i for i in matches if i.get("actual_status") == "running"]
    if running:
        print("Found existing running instance(s); reusing the first one.")
        for i in running:
            url = _public_url(i)
            print(f"  {i['id']} {i.get('label')} -> {url or 'no public URL yet'}")
        return 0

    if matches:
        print(
            f"WARNING: Found {len(matches)} non-running matching instance(s). "
            "Continuing to create a new one."
        )

    # 2. Load known-good history and search offers
    known_good = load_known_good(args.known_good_file)
    known_good_machine_ids = {i.get("machine_id") for i in known_good if i.get("machine_id")}
    if known_good_machine_ids:
        print(
            f"Loaded {len(known_good)} known-good instance(s); "
            f"preferring machine_ids: {sorted(known_good_machine_ids)}"
        )

    # GPU 型号偏好（未显式 --gpu 时按 GPU_PREFERENCE 顺序逐级降级）
    gpu_candidates = [args.gpu] if args.gpu else GPU_PREFERENCE
    offers = []
    for gpu_name in gpu_candidates:
        print(
            f"Searching on-demand single-GPU offers for {gpu_name} with >= {args.min_vram_mb} MB VRAM ..."
        )
        offers = search_offers(key, gpu_name=gpu_name, min_vram_mb=args.min_vram_mb)
        if offers:
            break
        print(f"  No {gpu_name} offers; falling back to next GPU model.")
    if not offers:
        _die("No suitable offers found.")

    # Filter for direct port capability (direct_port_count >= 1)
    direct_offers = [
        o
        for o in offers
        if o.get("direct_port_count", 0) >= 1 or o.get("direct_port_count") is None
    ]
    if not direct_offers:
        _die("No offers with direct ports found.")

    direct_offers.sort(key=lambda o: _score_offer(o, known_good_machine_ids))
    print(f"Found {len(direct_offers)} offer(s) with direct ports.")

    # 3. Try creating from best offers
    created_id: Optional[int] = None
    chosen_offer: Optional[Dict[str, Any]] = None
    for offer in direct_offers[:5]:
        offer_id = offer.get("id")
        region = _offer_region(offer)
        cpu = offer.get("cpu_name", "unknown")
        print(
            f"Trying offer {offer_id} ({region}, cpu={cpu}, "
            f"reliability={offer.get('reliability')}, "
            f"dph=${offer.get('dph_total')}) ..."
        )
        try:
            result = create_instance(
                key,
                offer_id,
                image,
                label=args.label,
                disk=args.disk,
                env=REQUIRED_ENV,
            )
            created_id = result.get("new_contract")
            if created_id:
                chosen_offer = offer
                print(f"Created instance {created_id} from offer {offer_id}.")
                break
        except requests.HTTPError as e:
            print(f"  Offer {offer_id} failed: {e}", file=sys.stderr)
            continue

    if not created_id:
        _die("Could not create instance from any suitable offer.")

    # 4. Poll until running and port assigned
    print("Polling instance status...")
    deadline = time.time() + args.timeout
    public_url: Optional[str] = None
    while time.time() < deadline:
        instance = get_instance(key, created_id)
        if not instance:
            time.sleep(args.interval)
            continue

        status = instance.get("actual_status", "?")
        public_url = _public_url(instance)
        print(f"  status={status}, url={public_url or 'pending'}")

        if status == "running" and public_url:
            break
        time.sleep(args.interval)
    else:
        _die(f"Timed out waiting for instance {created_id} to become reachable.")

    print(f"Instance {created_id} is running at {public_url}")

    # 5. Health check
    print("Waiting for /health ...")
    health_ok = False
    deadline = time.time() + args.health_timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{public_url}/health", timeout=10)
            print(f"  /health status={resp.status_code} body={resp.text[:200]}")
            if resp.status_code == 200:
                health_ok = True
                break
        except Exception as e:
            print(f"  /health error: {e}")
        time.sleep(args.interval)

    # 6. Logs
    print("Fetching recent logs for GPU runtime check ...")
    gpu_runtime_ok = False
    try:
        logs = instance_logs(key, created_id)
        if "GPU runtime check passed" in logs:
            gpu_runtime_ok = True
            print("  Confirmed: GPU runtime check passed.")
        else:
            print("  GPU runtime check message not found in recent logs.")
        if args.show_logs:
            print("--- logs ---")
            print(logs[-4000:])
    except requests.HTTPError as e:
        print(f"  Could not fetch logs: {e}", file=sys.stderr)

    if not health_ok:
        _die("Health check did not succeed within timeout.")

    # Persist known-good entry
    if gpu_runtime_ok and not args.no_save_known_good and chosen_offer is not None:
        entry = _build_known_good_entry(created_id, chosen_offer, image)
        known_good.append(entry)
        save_known_good(known_good, args.known_good_file)

    print(f"\nInstance ready: {created_id}")
    print(f"Public URL: {public_url}")
    return 0


def cmd_logs(key: str, args: argparse.Namespace) -> int:
    try:
        logs = instance_logs(key, int(args.instance_id))
        print(logs)
    except requests.HTTPError as e:
        _die(f"Failed to fetch logs: {e}")
    return 0


def cmd_health(key: str, args: argparse.Namespace) -> int:
    """OmniVoice 风格健康探测：GET <url>/health，要求 200 且 {"ok": true}。"""
    url = args.url.rstrip("/")
    try:
        resp = requests.get(f"{url}/health", timeout=10)
    except Exception as e:
        _die(f"health check error: {e}")
    print(f"/health status={resp.status_code} body={resp.text[:300]}")
    ok = False
    try:
        ok = resp.status_code == 200 and bool(resp.json().get("ok"))
    except Exception:
        ok = False
    if not ok:
        _die("health check failed (expect HTTP 200 with {\"ok\": true})")
    return 0


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manage Vast.ai instances for video-subtitle-remover API."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VAST_API_KEY"),
        help="Vast.ai API key (defaults to VAST_API_KEY env var)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="List matching instances")
    sub.add_parser("list", help="Alias of status (OmniVoice-style verb)")

    p_stop = sub.add_parser("stop", help="Delete all matching instances")
    sub.add_parser("destroy-all", help="Alias of stop (OmniVoice-style verb)")

    p_health = sub.add_parser("health", help="Check <url>/health (no API key needed)")
    p_health.add_argument("url", help="Public base URL of the instance")

    p_start = sub.add_parser("start", help="Start a new instance")
    p_start.add_argument("--image", default="", help="Docker image (defaults to latest digest)")
    p_start.add_argument("--label", default=DEFAULT_LABEL, help="Instance label")
    p_start.add_argument("--disk", type=int, default=DEFAULT_DISK, help="Disk size in GB")
    p_start.add_argument("--gpu", default="", help="GPU name filter (default: preference order 3090 Ti -> 3090 -> 4090)")
    p_start.add_argument("--min-vram-mb", type=int, default=DEFAULT_MIN_VRAM_MB, help="Minimum VRAM in MB")
    p_start.add_argument("--timeout", type=int, default=600, help="Seconds to wait for running")
    p_start.add_argument("--health-timeout", type=int, default=300, help="Seconds to wait for /health")
    p_start.add_argument("--interval", type=int, default=10, help="Polling interval seconds")
    p_start.add_argument("--show-logs", action="store_true", help="Print logs after startup")
    p_start.add_argument(
        "--known-good-file",
        default=KNOWN_GOOD_FILE,
        help="Path to known-good instances JSON ledger",
    )
    p_start.add_argument(
        "--no-save-known-good",
        action="store_true",
        help="Do not append a successful instance to the known-good ledger",
    )

    p_logs = sub.add_parser("logs", help="Fetch logs for an instance")
    p_logs.add_argument("instance_id", help="Instance ID")

    args = parser.parse_args(argv)
    # health 不需要 Vast API key；其余命令必须解析出 key
    key = None if args.command == "health" else (args.api_key or _api_key())

    commands = {
        "status": cmd_status,
        "list": cmd_status,
        "stop": cmd_stop,
        "destroy-all": cmd_stop,
        "start": cmd_start,
        "logs": cmd_logs,
        "health": cmd_health,
    }
    return commands[args.command](key, args)


if __name__ == "__main__":
    sys.exit(main())
