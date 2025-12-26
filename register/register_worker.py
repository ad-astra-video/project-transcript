import os
import logging
import sys
import time
import requests
import json
import threading
import signal
from typing import Optional, Dict, Any

#set where to send registration request
ORCH_URL = os.environ.get("ORCH_URL", "")
ORCH_SECRET = os.environ.get("ORCH_SECRET","")
# static configuration (these rarely change at runtime)
CAPABILITY_NAME = os.environ.get("CAPABILITY_NAME", "")
CAPABILITY_URL = os.environ.get("CAPABILITY_URL", "http://localhost:9876")
CAPABILITY_DESCRIPTION = os.environ.get("CAPABILITY_DESCRIPTION", "")


# Get the logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def register_to_orchestrator():
    # Build request payload from environment variables each time so
    # updates to capacity/price are picked up without restarting the worker
    capability_capacity = int(os.environ.get("CAPABILITY_CAPACITY", "1"))
    capability_price_per_unit = int(os.environ.get("CAPABILITY_PRICE_PER_UNIT", "0"))
    capability_price_scaling = int(os.environ.get("CAPABILITY_PRICE_SCALING", "1"))
    capability_price_currency = os.environ.get("CAPABILITY_PRICE_CURRENCY", "WEI")

    register_req: Dict[str, Any] = {
        "url": CAPABILITY_URL,
        "name": CAPABILITY_NAME,
        "description": CAPABILITY_DESCRIPTION,
        "capacity": capability_capacity,
        "price_per_unit": capability_price_per_unit,
        "price_scaling": capability_price_scaling,
        "currency": capability_price_currency,
    }
    headers = {
        "Authorization": ORCH_SECRET,
        "Content-Type": "application/json",
    }
    #do the registration
    max_retries = 10
    delay = 2  # seconds
    logger.info("registering: " + json.dumps(register_req))
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                ORCH_URL + "/capability/register",
                json=register_req,
                headers=headers,
                timeout=5,
                verify=False,
            )  # You can change to POST or other method

            if response.status_code == 200:
                logger.info("Capability registered")
                return True
            elif response.status_code == 400:
                logger.error("orch secret incorrect")
                return False
            else:
                logger.warning(
                    f"Attempt {attempt} returned status {response.status_code}: {response.text}"
                )
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt} failed with error: {e}")

        if attempt == max_retries:
            logger.error("All retries failed.")
            break
        time.sleep(delay)

    return False


def re_register_loop(stop_event: threading.Event, interval_seconds: int = 60):
    """Background loop that re-registers periodically until stop_event is set."""
    logger.info("Starting re-register loop (every %s seconds)", interval_seconds)
    # Do an initial registration immediately
    try:
        register_to_orchestrator()
    except Exception as e:
        logger.exception("Initial registration failed: %s", e)

    while not stop_event.wait(interval_seconds):
        try:
            logger.debug("Periodic re-registration triggered")
            register_to_orchestrator()
        except Exception:
            logger.exception("Error during periodic re-registration")

if __name__ == "__main__":
    # Start background re-register thread that updates capacity/price from env each minute
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=re_register_loop, args=(stop_event, 60), daemon=True)
    worker_thread.start()

    print("worker registration background thread started")

    # Graceful shutdown handling
    def _signal_handler(signum, frame):
        logger.info("Received shutdown signal (%s), stopping re-register loop", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        # Keep main thread alive until stop_event is set
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()

    # Wait briefly for thread to exit
    worker_thread.join(timeout=5)
    logger.info("register_worker exiting")
    