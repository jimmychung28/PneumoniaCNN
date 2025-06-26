#!/usr/bin/env python3
"""
Script to update import statements after reorganization.
"""

import os
import re
from pathlib import Path

def update_imports_in_file(file_path, import_mappings):
    """Update import statements in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Update import statements
        for old_import, new_import in import_mappings.items():
            # Handle different import styles
            patterns = [
                (f"from {old_import} import", f"from {new_import} import"),
                (f"import {old_import}", f"import {new_import}"),
            ]
            
            for old_pattern, new_pattern in patterns:
                content = content.replace(old_pattern, new_pattern)
        
        # Update data path references
        content = content.replace("'chest_xray/", "'data/chest_xray/")
        content = content.replace('"chest_xray/', '"data/chest_xray/')
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Updated: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    """Update all import statements."""
    print("🔄 Updating import statements after reorganization...")
    
    # Define import mappings
    import_mappings = {
        # Core modules
        'validation_utils': 'src.utils.validation_utils',
        'config_schema': 'src.config.config_schema',
        'config_loader': 'src.config.config_loader',
        'config_cli': 'src.config.config_cli',
        'data_pipeline': 'src.training.data_pipeline',
        'mixed_precision_trainer': 'src.training.mixed_precision_trainer',
        'preprocessing_pipeline': 'src.training.preprocessing_pipeline',
        'cnn': 'src.models.cnn',
        'unet_segmentation': 'src.models.unet_segmentation',
        'segmentation_classification_pipeline': 'src.models.segmentation_classification_pipeline',
        'train_two_stage_model': 'src.training.train_two_stage_model',
        'fix_tensorflow_warnings': 'src.utils.fix_tensorflow_warnings',
    }
    
    # Find all Python files
    python_files = []
    
    # Add src files
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Add script files
    for root, dirs, files in os.walk('scripts'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Add test files
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Add config files in src/config
    for file in ['src/config/config_cli.py']:
        if os.path.exists(file):
            python_files.append(file)
    
    updated_count = 0
    
    # Update each file
    for file_path in python_files:
        if update_imports_in_file(file_path, import_mappings):
            updated_count += 1
    
    print(f"\n📊 Summary: Updated {updated_count} files out of {len(python_files)} total files")
    
    # Create entry point scripts
    create_entry_points()

def create_entry_points():
    """Create convenient entry point scripts."""
    print("\n🚀 Creating entry point scripts...")
    
    # Main training entry point
    train_script = """#!/usr/bin/env python3
\"\"\"
Main entry point for CNN training.
\"\"\"
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.cnn import main

if __name__ == "__main__":
    main()
"""
    
    with open('train.py', 'w') as f:
        f.write(train_script)
    
    # Configuration CLI entry point
    config_script = """#!/usr/bin/env python3
\"\"\"
Configuration management CLI entry point.
\"\"\"
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.config_cli import main

if __name__ == "__main__":
    main()
"""
    
    with open('config.py', 'w') as f:
        f.write(config_script)
    
    print("✅ Created train.py and config.py entry points")

if __name__ == "__main__":
    main()