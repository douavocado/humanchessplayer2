import os
import datetime

class Logger:
    def __init__(self, log_file: str = None):
        if log_file is None:
            log_dir = os.path.join(os.getcwd(), 'Engine_logs')
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, str(datetime.datetime.now()).replace(" ", "").replace(":", "_") + '.txt')
        else:
            self.log_file = log_file
        self.log_buffer = "" # Use a buffer to accumulate log messages

    def add_log(self, message: str):
        """Adds a message to the log buffer."""
        self.log_buffer += message

    def write_log(self):
        """Writes the buffered log messages to the log file and clears the buffer."""
        if not self.log_buffer:
            return # Don't write empty logs
        try:
            with open(self.log_file, 'a') as log_f:
                log_f.write(self.log_buffer)
            self.log_buffer = "" # Clear buffer after successful write
        except IOError as e:
            print(f"Error writing to log file {self.log_file}: {e}")
            # Optionally, handle the error differently, e.g., retry or log to console