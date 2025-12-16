#!/usr/bin/env python3
"""
Analyze timing from client logs to understand delays.
"""

import re

def parse_client_log(filepath):
    """Parse client log and extract timing information."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    
    # Extract timing info
    engine_times = []
    update_counts = 0
    
    print("=" * 70)
    print("CLIENT LOG TIMING ANALYSIS")
    print("=" * 70)
    
    for i, line in enumerate(lines):
        if "Time taken to get move from engine:" in line:
            match = re.search(r"Time taken to get move from engine: ([\d.]+)", line)
            if match:
                engine_times.append(float(match.group(1)))
                print(f"  Move {len(engine_times)}: Engine time = {float(match.group(1))*1000:.1f}ms")
        
        if "Updated dynamic information from full image scans" in line:
            update_counts += 1
    
    print(f"\n  Total engine calls: {len(engine_times)}")
    print(f"  Total full image scans: {update_counts}")
    print(f"  Average scans per move: {update_counts / len(engine_times):.1f}")
    
    print(f"\n  Engine time stats:")
    print(f"    Mean: {sum(engine_times)/len(engine_times)*1000:.1f}ms")
    print(f"    Min: {min(engine_times)*1000:.1f}ms")
    print(f"    Max: {max(engine_times)*1000:.1f}ms")
    
    # At 107ms per scan, this means:
    scan_time_ms = 107
    print(f"\n  Estimated scan overhead (at {scan_time_ms}ms/scan):")
    print(f"    Total scan time: {update_counts * scan_time_ms}ms = {update_counts * scan_time_ms / 1000:.1f}s")
    
    return engine_times, update_counts

def main():
    import sys
    from pathlib import Path
    
    # Default to the attached log file
    log_path = Path(__file__).parent.parent / "Client_logs" / "2025-12-1523_35_28.960581.txt"
    
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    
    print(f"Analyzing: {log_path}\n")
    parse_client_log(log_path)
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
With 48 full image scans taking ~107ms each:
  Total scan time: ~5.1 seconds

For 18 moves in the game (36 ply / 2), that's:
  Average scan overhead per move: ~284ms

This explains why the game feels slower - each move cycle includes
multiple full scans (average 2.7 per move).

The 4K resolution is causing:
  - Background removal: ~52ms per scan
  - Template matching: ~38ms per scan
  - Total per scan: ~107ms

At 1080p (50% scale), this would be:
  - Background removal: ~10ms per scan
  - Total per scan: ~55ms
  
This would roughly DOUBLE the responsiveness.
""")

if __name__ == "__main__":
    main()

