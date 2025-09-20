"""
Test configuration for the xsi-llms-attack project.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "This is a sample news article for testing purposes."

@pytest.fixture
def sample_fake_text():
    """Sample fake news text for testing."""
    return "This is clearly fake news designed to mislead readers."

@pytest.fixture 
def sample_real_text():
    """Sample real news text for testing."""
    return "According to official sources, the event happened as reported."