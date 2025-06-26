#!/usr/bin/env python3
"""
Configuration management CLI entry point.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.config_cli import main

if __name__ == "__main__":
    main()
