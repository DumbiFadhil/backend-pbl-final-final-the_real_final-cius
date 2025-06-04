import re

def safe_filename(name):
    """Sanitize a string to be safe for use as a filename."""
    return re.sub(r'[^\w\-]', '_', name)