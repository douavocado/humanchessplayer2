"""
Engine-specific logger that integrates with the unified logging system.

This module provides backwards-compatible logging for engine components
while routing all output through the central SessionLogger.
"""

import os
import datetime
from typing import Optional

# Import the unified logging system
from common.logging import get_logger, LogLevel


class Logger:
    """
    Engine logger that integrates with the unified SessionLogger.
    
    Maintains backwards compatibility with the existing add_log/write_log
    interface while using the new logging infrastructure.
    """
    
    def __init__(self, log_file: str = None):
        """
        Initialise the engine logger.
        
        Args:
            log_file: Legacy parameter, kept for backwards compatibility.
                      Ignored when SessionLogger is available.
        """
        self._legacy_log_file = log_file
        self._buffer: list = []  # Use list for O(1) appends
        self._context: Optional[str] = None
        
        # Fallback to legacy file-based logging if SessionLogger not initialised
        if get_logger() is None and log_file is None:
            log_dir = os.path.join(os.getcwd(), 'Engine_logs')
            os.makedirs(log_dir, exist_ok=True)
            self._legacy_log_file = os.path.join(
                log_dir, 
                str(datetime.datetime.now()).replace(" ", "").replace(":", "_") + '.txt'
            )
    
    def set_context(self, context: str) -> None:
        """Set a context identifier for subsequent log messages."""
        self._context = context
    
    def clear_context(self) -> None:
        """Clear the context identifier."""
        self._context = None
    
    def add_log(self, message: str) -> None:
        """
        Add a message to the log buffer.
        
        Args:
            message: The message to log
        """
        self._buffer.append(message)
    
    def _detect_level(self, message: str) -> LogLevel:
        """Detect log level from message content."""
        msg_upper = message.upper()
        if msg_upper.startswith("ERROR") or "ERROR:" in msg_upper:
            return LogLevel.ERROR
        elif msg_upper.startswith("WARNING") or "WARNING:" in msg_upper:
            return LogLevel.WARN
        elif msg_upper.startswith("CRITICAL"):
            return LogLevel.CRITICAL
        elif "[PERF]" in message:
            return LogLevel.PERF
        elif msg_upper.startswith("DEBUG"):
            return LogLevel.DEBUG
        return LogLevel.INFO
    
    def write_log(self) -> None:
        """Write the buffered log messages and clear the buffer."""
        if not self._buffer:
            return
        
        logger = get_logger()
        
        if logger is not None:
            # Use the unified logging system
            for msg in self._buffer:
                # Parse multiline messages
                for line in msg.strip().split('\n'):
                    if not line.strip():
                        continue
                    
                    level = self._detect_level(line)
                    # Clean up the message if it starts with level prefix
                    clean_line = line
                    for prefix in ["ERROR:", "WARNING:", "DEBUG:", "[PERF]"]:
                        if clean_line.upper().startswith(prefix):
                            clean_line = clean_line[len(prefix):].strip()
                            break
                    
                    logger.engine(level, clean_line, self._context)
        else:
            # Fallback to legacy file-based logging
            if self._legacy_log_file:
                try:
                    with open(self._legacy_log_file, 'a') as log_f:
                        log_f.write(''.join(self._buffer))
                except IOError as e:
                    print(f"Error writing to log file {self._legacy_log_file}: {e}")
        
        self._buffer.clear()
    
    def flush(self) -> None:
        """Write any remaining messages in the buffer to the log file."""
        self.write_log()
    
    # =========================================================================
    # Convenience methods for direct logging at specific levels
    # =========================================================================
    
    def debug(self, message: str) -> None:
        """Log a DEBUG level message."""
        logger = get_logger()
        if logger:
            logger.engine_debug(message, self._context)
        else:
            self.add_log(f"DEBUG: {message}\n")
    
    def info(self, message: str) -> None:
        """Log an INFO level message."""
        logger = get_logger()
        if logger:
            logger.engine_info(message, self._context)
        else:
            self.add_log(f"{message}\n")
    
    def perf(self, message: str) -> None:
        """Log a PERF level message."""
        logger = get_logger()
        if logger:
            logger.engine_perf(message, self._context)
        else:
            self.add_log(f"[PERF] {message}\n")
    
    def warn(self, message: str) -> None:
        """Log a WARN level message."""
        logger = get_logger()
        if logger:
            logger.engine_warn(message, self._context)
        else:
            self.add_log(f"WARNING: {message}\n")
    
    def error(self, message: str) -> None:
        """Log an ERROR level message."""
        logger = get_logger()
        if logger:
            logger.engine_error(message, self._context)
        else:
            self.add_log(f"ERROR: {message}\n")