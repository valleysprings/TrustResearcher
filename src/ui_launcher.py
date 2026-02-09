#!/usr/bin/env python3
"""
TrustResearcher - UI Launcher

Lightweight launcher for research process visualization interface:
1. Research process visualization interface
2. Interactive research management

Usage:
    python -m src.ui_launcher --process-ui     # Launch research process visualization
"""

import argparse
import sys

from .utils.web_ui import ProcessVisualizerUI
from .utils.config import load_config


def launch_process_ui(config_dir: str = 'configs/',
                     port: int = 7860, share: bool = False, host: str = 'localhost'):
    """Launch the research process visualization UI (host configurable)"""
    print(f"🚀 Starting Research Process Visualization UI...")
    print(f"💡 Tip: UI will automatically find available port if {port} is occupied")

    try:
        try:
            config = load_config(config_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load config: {e}")
            config = {}
        ui = ProcessVisualizerUI(config=config, port=port, share=share, host=host)
        ui.launch()
    except KeyboardInterrupt:
        print("\n👋 Process UI stopped")
    except Exception as e:
        print(f"❌ Failed to start process UI: {e}")
        if "port" in str(e).lower():
            print("💡 Try running: lsof -ti:7860 | xargs kill -9")
            print("💡 Or use a different port: --process-port 7861")
        return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="TrustResearcher - UI Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # UI mode selection
    parser.add_argument('--process-ui', action='store_true', required=True,
                       help='Launch research process visualization interface')
    
    # Server configuration
    parser.add_argument('--process-port', type=int, default=7860,
                       help='Research process UI port (default: 7860)')
    parser.add_argument('--process-host', type=str, default='localhost',
                    help='Research process UI host (default: localhost; use 0.0.0.0 for LAN)')
    parser.add_argument('--share', action='store_true',
                       help='Create public Gradio links')
    parser.add_argument('--config', type=str, default='configs/',
                       help='Configuration directory path')
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("🤖 TrustResearcher - UI Launcher")
        print("=" * 60)
        
        if args.process_ui:
            return launch_process_ui(
                config_dir=args.config,
                port=args.process_port,
                share=args.share,
                host=args.process_host
            )
    
    except KeyboardInterrupt:
        print("\n👋 UI launcher stopped")
        return 0
    
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
