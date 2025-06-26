import os
import time
import pickle
import sys
import signal

if len(sys.argv) != 3:
    print("Usage: python watchdog.py <pid_file_path> <timeout_seconds>", flush=True)
    sys.exit(1)

pid_file_path = sys.argv[1]
timeout_seconds = int(sys.argv[2])
start_time = time.time()
stop_file = f"{pid_file_path}.stop"  # Special file to stop watchdog

# Load stored PIDs and PGIDs
try:
    with open(pid_file_path, "rb") as f:
        pid_list = pickle.load(f)
except FileNotFoundError:
    print("Watchdog: No process list found, exiting.", flush=True)
    sys.exit(0)

print(f"Watchdog: Monitoring {len(pid_list)} processes...", flush=True)

while True:
    if os.path.exists(stop_file):
        print("Watchdog: STOP signal detected. Exiting cleanly.", flush=True)
        os.remove(stop_file)
        sys.exit(0)

    if time.time() - start_time > timeout_seconds:
        break

    time.sleep(5)  # Check every 5 seconds

# **Timeout reached – kill processes**
print("Watchdog: Timeout reached! Killing processes.", flush=True)

for rank, pid, pgid in pid_list:
    try:
        print(f"Watchdog: Sending SIGKILL to process {pid} (PGID: {pgid})", flush=True)
        os.kill(pid, signal.SIGKILL)  # Kill individual process
        os.system(f"kill -9 {pid}")  # Extra force kill
        os.system(f"kill -9 -{pgid}")  # Kill process group

    except ProcessLookupError:
        print(f"Watchdog: Process {pid} already terminated.", flush=True)
    except PermissionError:
        print(f"Watchdog: No permission to kill process {pid}.", flush=True)
    except Exception as e:
        print(f"Watchdog: Error killing process {pid}: {e}", flush=True)

print("Watchdog finished cleanup.", flush=True)
sys.exit(0)
