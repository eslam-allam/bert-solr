import signal
import logging

logger = logging.getLogger(__name__)

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, *args):
    if self.kill_now:
      logger.info("Received shutdown signal twice. Raising interrupt.")
      raise KeyboardInterrupt()
    logger.info("Received shut down signal. Exit switch flipped.")
    self.kill_now = True