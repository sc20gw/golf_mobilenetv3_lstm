# src/collect_env.py
import sys, platform, torch, json, os, subprocess
os.makedirs("logs", exist_ok=True)
env = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}
print(json.dumps(env, indent=2))
with open("logs/env_info.json","w",encoding="utf-8") as f:
    json.dump(env,f,ensure_ascii=False,indent=2)
try:
    req = subprocess.check_output([sys.executable,"-m","pip","freeze"], text=True, timeout=120)
    with open("logs/requirements_lock.txt","w",encoding="utf-8") as f:
        f.write(req)
except Exception as e:
    print("pip freeze failed:", e)
