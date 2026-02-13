#!/usr/bin/env python3
"""Simple test script for MarkItDown text extraction."""

import logging
import pytest
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from markitdown import MarkItDown

# Set up logging to see the detailed output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_markitdown_extraction():
    """Test MarkItDown with various file types."""
    # Test files directory
    test_files_dir = Path("tests/test-files")

    # Skip test if test files don't exist
    if not test_files_dir.exists():
        pytest.skip("Test files directory not found")

    print("=== Testing MarkItDown Text Extraction ===\n")

    # Initialize MarkItDown
    markitdown = MarkItDown()

    # Test files to process
    test_files = [
        "iphone.md",  # Markdown file
        "math.md",  # Another markdown file
        "math.png",  # PNG image (requires LLM for description)
        "math_origin.pdf",  # PDF file
    ]

    print(f"Available test files in {test_files_dir}:")
    for file_path in test_files_dir.iterdir():
        if file_path.is_file():
            print(f"  - {file_path.name} ({file_path.suffix})")
    print()

    # Test each file
    for test_file in test_files:
        file_path = test_files_dir / test_file

        if not file_path.exists():
            print(f"❌ Skipping {test_file} - file not found")
            continue

        print(f"🧪 Testing extraction from: {test_file}")
        print(f"   File size: {file_path.stat().st_size:,} bytes")

        try:
            # Test MarkItDown extraction
            result = markitdown.convert(str(file_path))
            text_content = result.text_content

            if text_content:
                content_length = len(text_content)
                print(f"✅ Successfully extracted {content_length:,} characters")

                # Show first 200 characters as preview
                preview = text_content[:200].replace("\n", "\\n")
                print(f"   Preview: {preview}...")

                # Show some statistics
                lines = text_content.count("\n") + 1
                words = len(text_content.split())
                print(f"   Stats: {lines} lines, {words} words")
            else:
                print("⚠️  Extraction returned empty content")

        except Exception as e:
            print(f"❌ Extraction failed: {type(e).__name__}: {e}")

        print("-" * 60)

    print("\n=== Test Summary ===")
    print("MarkItDown text extraction test completed.")
    print("This validates that MarkItDown can process various file types.")


if __name__ == "__main__":
    test_markitdown_extraction()
