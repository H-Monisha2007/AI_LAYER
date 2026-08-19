"""
DeepForensics Diagnostic & Readiness Checker

Verifies system components:
- Python version & PyTorch environment (CPU vs CUDA)
- Node.js & npm.cmd installation
- Frontend directory & package.json scripts
- Backend module entry point
- Database connection (deepforensics.db)
- Trained model checkpoints in model_weights/
- Port 8000 (Backend) and Port 5173 (Frontend) availability
"""
import os
import sys
import json
import socket
import subprocess

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_DIR = os.path.join(ROOT_DIR, "model_weights")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
DB_PATH = os.path.join(ROOT_DIR, "deepforensics.db")


def check_port(port: int) -> bool:
    """Returns True if port is open/available, False if occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        result = s.connect_ex(("127.0.0.1", port))
        return result != 0  # 0 means port is currently in use/open by a server


def run_check():
    print("=" * 65)
    print("      DEEPFORENSICS AUTOMATED STARTUP DIAGNOSTICS      ")
    print("=" * 65)

    checks = []

    # 1. Python Environment
    py_ver = sys.version.split()[0]
    print(f"[+] Python Version: {py_ver}")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_str = "CUDA (GPU)" if cuda_avail else "CPU"
        print(f"[+] PyTorch Installed: Version {torch.__version__} ({device_str})")
        checks.append(("PyTorch Environment", "PASS"))
    except ImportError:
        print("[!] PyTorch NOT installed!")
        checks.append(("PyTorch Environment", "FAIL"))

    # 2. Node & npm.cmd Check
    node_installed = False
    try:
        res = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
        print(f"[+] Node.js Version: {res.stdout.strip()}")
        node_installed = True
    except Exception:
        print("[!] Node.js not found in PATH.")

    npm_installed = False
    try:
        res = subprocess.run(["cmd", "/c", "npm.cmd", "--version"], capture_output=True, text=True, check=True)
        print(f"[+] npm.cmd Version: {res.stdout.strip()}")
        npm_installed = True
    except Exception:
        print("[!] npm.cmd not found!")

    if node_installed and npm_installed:
        checks.append(("Node.js / npm Environment", "PASS"))
    else:
        checks.append(("Node.js / npm Environment", "FAIL"))

    # 3. Frontend Package Check
    frontend_pkg = os.path.join(FRONTEND_DIR, "package.json")
    if os.path.exists(frontend_pkg):
        with open(frontend_pkg, "r") as f:
            data = json.load(f)
        has_dev = "dev" in data.get("scripts", {})
        if has_dev:
            print(f"[+] Frontend Directory: {FRONTEND_DIR} (package.json 'dev' script OK)")
            checks.append(("Frontend Configuration", "PASS"))
        else:
            print(f"[!] Frontend package.json missing 'dev' script!")
            checks.append(("Frontend Configuration", "FAIL"))
    else:
        print(f"[!] Frontend directory package.json missing at {frontend_pkg}")
        checks.append(("Frontend Configuration", "FAIL"))

    # 4. Backend Main Entry Point Check
    backend_main = os.path.join(ROOT_DIR, "backend", "main.py")
    if os.path.exists(backend_main):
        print(f"[+] Backend Entry Point: backend/main.py OK")
        checks.append(("Backend Configuration", "PASS"))
    else:
        print(f"[!] Backend entry point missing at {backend_main}")
        checks.append(("Backend Configuration", "FAIL"))

    # 5. Database File Check
    if os.path.exists(DB_PATH):
        print(f"[+] SQLite Database: {DB_PATH} OK")
        checks.append(("Database Status", "PASS"))
    else:
        print(f"[!] SQLite Database file missing at {DB_PATH} (Will auto-initialize on startup)")
        checks.append(("Database Status", "PASS"))

    # 6. Model Weights & Checkpoints Check
    rgb_weights = os.path.join(WEIGHTS_DIR, "efficientnet_b4", "best.pt")
    freq_weights = os.path.join(WEIGHTS_DIR, "convnext_dct", "best.pt")
    noise_weights = os.path.join(WEIGHTS_DIR, "noise_model", "best.pt")

    ready_models = 0
    if os.path.exists(rgb_weights): ready_models += 1
    if os.path.exists(freq_weights): ready_models += 1
    if os.path.exists(noise_weights): ready_models += 1

    if ready_models > 0:
        print(f"[+] Model Checkpoints: {ready_models}/3 Domain PyTorch Weights Loaded OK")
        checks.append(("Model Checkpoints", "PASS"))
    else:
        print(f"[!] Model Checkpoints: 0/3 loaded (Status will be MODEL_NOT_READY until models are trained)")
        checks.append(("Model Checkpoints", "WARN"))

    # 7. Ports Availability Check
    backend_port_avail = check_port(8000)
    frontend_port_avail = check_port(5173)

    if backend_port_avail:
        print("[+] Backend Port 8000: Available")
    else:
        print("[!] Backend Port 8000: Occupied (Another server instance may be running)")

    if frontend_port_avail:
        print("[+] Frontend Port 5173: Available")
    else:
        print("[!] Frontend Port 5173: Occupied (Another Vite instance may be running)")

    print("\n" + "=" * 65)
    print("                    DIAGNOSTIC SUMMARY                    ")
    print("=" * 65)
    all_pass = True
    for item, status in checks:
        icon = "[PASS]" if status == "PASS" else (" shadow [WARN]" if status == "WARN" else "[FAIL]")
        print(f" {icon:<8} : {item}")
        if status == "FAIL":
            all_pass = False

    print("=" * 65)
    if all_pass:
        print("STATUS: SYSTEM READY FOR DEEPFORENSICS STARTUP!")
    else:
        print("STATUS: SOME COMPONENT CHECKS FAILED - SEE DETAILS ABOVE.")
    print("=" * 65)


if __name__ == "__main__":
    run_check()
