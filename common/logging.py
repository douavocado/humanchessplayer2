#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified logging system for the Human Chess Player project.

Provides a centralised, low-overhead logging solution with:
- Session-based directory structure
- Log levels with filtering
- List-based buffering for performance
- Formatted, readable output
- Separate channels for client, engine, and errors

@author: james
"""

import os
import time
import threading
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil


class LogLevel(IntEnum):
    """Log levels in order of verbosity (lower = more verbose)."""
    DEBUG = 0    # Detailed debugging information
    PERF = 1     # Performance metrics and timing
    INFO = 2     # General information about program flow
    WARN = 3     # Warnings that don't prevent operation
    ERROR = 4    # Errors that may affect operation
    CRITICAL = 5 # Critical errors that prevent operation


# Level display names with consistent width for alignment
LEVEL_NAMES = {
    LogLevel.DEBUG: "DEBUG",
    LogLevel.PERF: "PERF ",
    LogLevel.INFO: "INFO ",
    LogLevel.WARN: "WARN ",
    LogLevel.ERROR: "ERROR",
    LogLevel.CRITICAL: "CRIT ",
}

# ANSI colour codes for terminal output (optional)
LEVEL_COLOURS = {
    LogLevel.DEBUG: "\033[90m",    # Grey
    LogLevel.PERF: "\033[36m",     # Cyan
    LogLevel.INFO: "\033[0m",      # Default
    LogLevel.WARN: "\033[33m",     # Yellow
    LogLevel.ERROR: "\033[31m",    # Red
    LogLevel.CRITICAL: "\033[91m", # Bright red
}
RESET_COLOUR = "\033[0m"


class LogChannel:
    """
    A single log channel with buffered writing.
    
    Uses list-based buffering for O(1) appends and batched disk writes.
    """
    
    def __init__(self, filepath: Path, min_level: LogLevel = LogLevel.INFO,
                 buffer_size: int = 50, auto_flush: bool = True):
        """
        Initialise a log channel.
        
        Args:
            filepath: Path to the log file
            min_level: Minimum log level to record
            buffer_size: Number of messages before auto-flush
            auto_flush: Whether to auto-flush when buffer is full
        """
        self.filepath = filepath
        self.min_level = min_level
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush
        
        self._buffer: List[str] = []
        self._lock = threading.Lock()
        self._start_time = time.time()
        
        # Ensure parent directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write header
        self._write_header()
    
    def _write_header(self) -> None:
        """Write a header to the log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = [
            "=" * 80,
            f"  Log started: {timestamp}",
            f"  Min level: {LEVEL_NAMES[self.min_level]}",
            "=" * 80,
            "",
        ]
        with open(self.filepath, 'w') as f:
            f.write('\n'.join(header) + '\n')
    
    def log(self, level: LogLevel, message: str, context: Optional[str] = None) -> None:
        """
        Add a log message to the buffer.
        
        Args:
            level: Log level
            message: The message to log
            context: Optional context identifier (e.g., function name)
        """
        if level < self.min_level:
            return
        
        # Format the message
        elapsed = time.time() - self._start_time
        timestamp = f"{elapsed:8.3f}s"
        level_str = LEVEL_NAMES[level]
        
        if context:
            formatted = f"[{timestamp}] [{level_str}] [{context}] {message}"
        else:
            formatted = f"[{timestamp}] [{level_str}] {message}"
        
        # Add to buffer (thread-safe)
        with self._lock:
            self._buffer.append(formatted)
            
            if self.auto_flush and len(self._buffer) >= self.buffer_size:
                self._flush_unlocked()
    
    def _flush_unlocked(self) -> None:
        """Flush buffer to disk (must hold lock)."""
        if not self._buffer:
            return
        
        try:
            with open(self.filepath, 'a') as f:
                f.write('\n'.join(self._buffer) + '\n')
            self._buffer.clear()
        except IOError as e:
            print(f"Error writing to log file {self.filepath}: {e}")
    
    def flush(self) -> None:
        """Flush buffer to disk (thread-safe)."""
        with self._lock:
            self._flush_unlocked()
    
    def close(self) -> None:
        """Flush and close the log channel."""
        self.flush()
        
        # Write footer
        elapsed = time.time() - self._start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer = [
            "",
            "=" * 80,
            f"  Log ended: {timestamp}",
            f"  Session duration: {elapsed:.1f}s",
            "=" * 80,
        ]
        try:
            with open(self.filepath, 'a') as f:
                f.write('\n'.join(footer) + '\n')
        except IOError:
            pass


class SessionLogger:
    """
    Unified session-based logger for the entire application.
    
    Manages multiple log channels (client, engine, errors) within a single
    session directory, providing a consistent interface across all components.
    
    Directory structure:
        logs/
        ├── sessions/
        │   └── YYYY-MM-DD_HH-MM-SS/
        │       ├── client.log
        │       ├── engine.log
        │       └── errors/
        └── latest -> sessions/YYYY-MM-DD_HH-MM-SS/
    """
    
    _instance: Optional['SessionLogger'] = None
    _lock = threading.Lock()
    
    def __init__(self, base_dir: Optional[Path] = None, 
                 client_level: LogLevel = LogLevel.INFO,
                 engine_level: LogLevel = LogLevel.INFO,
                 console_output: bool = False):
        """
        Initialise the session logger.
        
        Args:
            base_dir: Base directory for logs (default: ./logs)
            client_level: Minimum log level for client channel
            engine_level: Minimum log level for engine channel
            console_output: Whether to also print to console
        """
        if base_dir is None:
            base_dir = Path(os.getcwd()) / "logs"
        
        self.base_dir = Path(base_dir)
        self.console_output = console_output
        
        # Create session directory with timestamp
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self.base_dir / "sessions" / self.session_id
        self.errors_dir = self.session_dir / "errors"
        
        # Create directories
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.errors_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log channels
        self._client = LogChannel(
            self.session_dir / "client.log",
            min_level=client_level,
            buffer_size=50
        )
        self._engine = LogChannel(
            self.session_dir / "engine.log", 
            min_level=engine_level,
            buffer_size=100  # Engine logs more frequently
        )
        
        # Update 'latest' symlink
        self._update_latest_symlink()
        
        # Write session info
        self._write_session_info()
        
        # Track game count for sub-session organisation
        self._game_count = 0
    
    def _update_latest_symlink(self) -> None:
        """Update the 'latest' symlink to point to current session."""
        latest_link = self.base_dir / "latest"
        try:
            if latest_link.is_symlink():
                latest_link.unlink()
            elif latest_link.exists():
                shutil.rmtree(latest_link)
            latest_link.symlink_to(self.session_dir, target_is_directory=True)
        except OSError:
            pass  # Symlinks may not work on all platforms
    
    def _write_session_info(self) -> None:
        """Write session metadata to a separate file."""
        info_file = self.session_dir / "session_info.txt"
        info = [
            f"Session ID: {self.session_id}",
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Working Directory: {os.getcwd()}",
            "",
        ]
        with open(info_file, 'w') as f:
            f.write('\n'.join(info))
    
    @classmethod
    def get_instance(cls) -> Optional['SessionLogger']:
        """Get the global SessionLogger instance (if initialised)."""
        return cls._instance
    
    @classmethod
    def initialise(cls, **kwargs) -> 'SessionLogger':
        """
        Initialise the global SessionLogger instance.
        
        Args:
            **kwargs: Arguments to pass to SessionLogger.__init__
        
        Returns:
            The initialised SessionLogger instance
        """
        with cls._lock:
            if cls._instance is not None:
                # Close existing instance
                cls._instance.close()
            cls._instance = cls(**kwargs)
            return cls._instance
    
    # =========================================================================
    # Client Logging Methods
    # =========================================================================
    
    def client(self, level: LogLevel, message: str, context: Optional[str] = None) -> None:
        """Log a message to the client channel."""
        self._client.log(level, message, context)
        if self.console_output and level >= LogLevel.INFO:
            self._print_coloured(level, message, "CLIENT", context)
    
    def client_debug(self, message: str, context: Optional[str] = None) -> None:
        """Log a DEBUG message to client channel."""
        self.client(LogLevel.DEBUG, message, context)
    
    def client_info(self, message: str, context: Optional[str] = None) -> None:
        """Log an INFO message to client channel."""
        self.client(LogLevel.INFO, message, context)
    
    def client_perf(self, message: str, context: Optional[str] = None) -> None:
        """Log a PERF message to client channel."""
        self.client(LogLevel.PERF, message, context)
    
    def client_warn(self, message: str, context: Optional[str] = None) -> None:
        """Log a WARN message to client channel."""
        self.client(LogLevel.WARN, message, context)
    
    def client_error(self, message: str, context: Optional[str] = None) -> None:
        """Log an ERROR message to client channel."""
        self.client(LogLevel.ERROR, message, context)
    
    # =========================================================================
    # Engine Logging Methods
    # =========================================================================
    
    def engine(self, level: LogLevel, message: str, context: Optional[str] = None) -> None:
        """Log a message to the engine channel."""
        self._engine.log(level, message, context)
        if self.console_output and level >= LogLevel.WARN:
            self._print_coloured(level, message, "ENGINE", context)
    
    def engine_debug(self, message: str, context: Optional[str] = None) -> None:
        """Log a DEBUG message to engine channel."""
        self.engine(LogLevel.DEBUG, message, context)
    
    def engine_info(self, message: str, context: Optional[str] = None) -> None:
        """Log an INFO message to engine channel."""
        self.engine(LogLevel.INFO, message, context)
    
    def engine_perf(self, message: str, context: Optional[str] = None) -> None:
        """Log a PERF message to engine channel."""
        self.engine(LogLevel.PERF, message, context)
    
    def engine_warn(self, message: str, context: Optional[str] = None) -> None:
        """Log a WARN message to engine channel."""
        self.engine(LogLevel.WARN, message, context)
    
    def engine_error(self, message: str, context: Optional[str] = None) -> None:
        """Log an ERROR message to engine channel."""
        self.engine(LogLevel.ERROR, message, context)
    
    # =========================================================================
    # Error File Handling
    # =========================================================================
    
    def save_error_file(self, prefix: str, extension: str, content: bytes) -> Path:
        """
        Save an error-related file (screenshot, dump, etc.).
        
        Args:
            prefix: Descriptive prefix for the file
            extension: File extension (e.g., 'png', 'txt')
            content: Binary content to write
        
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%H-%M-%S_%f")[:-3]  # HH-MM-SS_mmm
        filename = f"{prefix}_{timestamp}.{extension}"
        filepath = self.errors_dir / filename
        
        with open(filepath, 'wb') as f:
            f.write(content)
        
        return filepath
    
    def save_error_image(self, prefix: str, image) -> Optional[Path]:
        """
        Save an error-related image using OpenCV.
        
        Args:
            prefix: Descriptive prefix for the file
            image: OpenCV image (numpy array)
        
        Returns:
            Path to the saved file, or None if failed
        """
        try:
            import cv2
            import numpy as np
            
            if image is None or not isinstance(image, np.ndarray):
                return None
            
            timestamp = datetime.now().strftime("%H-%M-%S_%f")[:-3]
            filename = f"{prefix}_{timestamp}.png"
            filepath = self.errors_dir / filename
            
            cv2.imwrite(str(filepath), image)
            return filepath
        except Exception as e:
            self.client_error(f"Failed to save error image: {e}")
            return None
    
    def save_error_context(self, prefix: str, context: Dict[str, Any]) -> Path:
        """
        Save error context information as a text file.
        
        Args:
            prefix: Descriptive prefix for the file
            context: Dictionary of context information
        
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%H-%M-%S_%f")[:-3]
        filename = f"{prefix}_{timestamp}_context.txt"
        filepath = self.errors_dir / filename
        
        lines = [f"Error Context: {prefix}", "=" * 60, ""]
        for key, value in context.items():
            lines.append(f"{key}: {value}")
        lines.append("")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        return filepath
    
    # =========================================================================
    # Game Session Management
    # =========================================================================
    
    def start_game(self, game_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark the start of a new game within the session.
        
        Adds a visual separator in logs for easy navigation.
        """
        self._game_count += 1
        separator = "═" * 70
        
        game_header = [
            "",
            separator,
            f"  GAME {self._game_count} STARTED",
        ]
        
        if game_info:
            for key, value in game_info.items():
                game_header.append(f"  {key}: {value}")
        
        game_header.extend([
            f"  Time: {datetime.now().strftime('%H:%M:%S')}",
            separator,
            "",
        ])
        
        header_str = '\n'.join(game_header)
        self._client.log(LogLevel.INFO, header_str)
        self._engine.log(LogLevel.INFO, header_str)
    
    def end_game(self, result: Optional[str] = None) -> None:
        """
        Mark the end of a game.
        
        Flushes logs and adds a visual separator.
        """
        separator = "─" * 70
        
        game_footer = [
            "",
            separator,
            f"  GAME {self._game_count} ENDED",
        ]
        
        if result:
            game_footer.append(f"  Result: {result}")
        
        game_footer.extend([
            f"  Time: {datetime.now().strftime('%H:%M:%S')}",
            separator,
            "",
        ])
        
        footer_str = '\n'.join(game_footer)
        self._client.log(LogLevel.INFO, footer_str)
        self._engine.log(LogLevel.INFO, footer_str)
        
        # Flush logs after each game
        self.flush()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _print_coloured(self, level: LogLevel, message: str, 
                        channel: str, context: Optional[str]) -> None:
        """Print a coloured message to console."""
        colour = LEVEL_COLOURS.get(level, "")
        level_str = LEVEL_NAMES[level]
        
        if context:
            print(f"{colour}[{channel}] [{level_str}] [{context}] {message}{RESET_COLOUR}")
        else:
            print(f"{colour}[{channel}] [{level_str}] {message}{RESET_COLOUR}")
    
    def flush(self) -> None:
        """Flush all log channels to disk."""
        self._client.flush()
        self._engine.flush()
    
    def close(self) -> None:
        """Close all log channels and finalise the session."""
        self._client.close()
        self._engine.close()
        
        # Update session info with end time
        info_file = self.session_dir / "session_info.txt"
        try:
            with open(info_file, 'a') as f:
                f.write(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Games played: {self._game_count}\n")
        except IOError:
            pass
    
    def __del__(self):
        """Ensure logs are flushed on garbage collection."""
        try:
            self.flush()
        except:
            pass


# =============================================================================
# Convenience Functions for Global Access
# =============================================================================

def get_logger() -> Optional[SessionLogger]:
    """Get the global SessionLogger instance."""
    return SessionLogger.get_instance()


def init_logging(client_level: LogLevel = LogLevel.INFO,
                 engine_level: LogLevel = LogLevel.INFO,
                 console_output: bool = False,
                 base_dir: Optional[Path] = None) -> SessionLogger:
    """
    Initialise the global logging system.
    
    Should be called once at application startup (e.g., in main.py).
    
    Args:
        client_level: Minimum log level for client logs
        engine_level: Minimum log level for engine logs
        console_output: Whether to print to console
        base_dir: Base directory for logs
    
    Returns:
        The initialised SessionLogger
    """
    return SessionLogger.initialise(
        base_dir=base_dir,
        client_level=client_level,
        engine_level=engine_level,
        console_output=console_output
    )


# =============================================================================
# Legacy Compatibility Layer
# =============================================================================

class LegacyLoggerAdapter:
    """
    Adapter to provide backwards compatibility with the old logging interface.
    
    Allows gradual migration from the old `LOG += "message"` pattern.
    """
    
    def __init__(self, channel: str = "client"):
        """
        Create a legacy adapter.
        
        Args:
            channel: Which channel to log to ("client" or "engine")
        """
        self.channel = channel
        self._pending = ""
    
    def add(self, message: str) -> None:
        """Add a message (mimics LOG += message)."""
        self._pending += message
    
    def write(self) -> None:
        """Write pending messages (mimics write_log())."""
        logger = get_logger()
        if logger is None or not self._pending:
            return
        
        # Parse the pending messages and log appropriately
        for line in self._pending.strip().split('\n'):
            if not line:
                continue
            
            # Detect log level from content
            level = LogLevel.INFO
            if line.startswith("ERROR"):
                level = LogLevel.ERROR
            elif line.startswith("[PERF]"):
                level = LogLevel.PERF
                line = line[6:].strip()  # Remove prefix
            elif line.startswith("WARNING"):
                level = LogLevel.WARN
            
            if self.channel == "client":
                logger.client(level, line)
            else:
                logger.engine(level, line)
        
        self._pending = ""
    
    def __iadd__(self, message: str) -> 'LegacyLoggerAdapter':
        """Support LOG += message syntax."""
        self.add(message)
        return self
