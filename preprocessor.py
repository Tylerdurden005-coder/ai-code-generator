"""
Request Validators
Validate incoming API requests
"""

import re


def validate_request(data):
    """
    Validate code generation request
    
    Args:
        data: Request data dictionary
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    
    # Check required fields
    required_fields = ['project_name', 'description', 'category']
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        if not data[field] or not str(data[field]).strip():
            return False, f"Field '{field}' cannot be empty"
    
    # Validate project name
    project_name = data['project_name']
    if len(project_name) < 3:
        return False, "Project name must be at least 3 characters"
    
    if len(project_name) > 100:
        return False, "Project name must be less than 100 characters"
    
    # Validate description
    description = data['description']
    if len(description) < 10:
        return False, "Description must be at least 10 characters"
    
    if len(description) > 5000:
        return False, "Description must be less than 5000 characters"
    
    # Validate category
    valid_categories = ['ml', 'dl', 'nlp', 'rl', 'dsa', 'web']
    if data['category'] not in valid_categories:
        return False, f"Invalid category. Must be one of: {', '.join(valid_categories)}"
    
    # Validate boolean fields
    boolean_fields = ['include_comments', 'include_tests', 
                     'include_documentation', 'include_visualization']
    for field in boolean_fields:
        if field in data and not isinstance(data[field], bool):
            return False, f"Field '{field}' must be a boolean"
    
    return True, None


def validate_project_name(name):
    """
    Validate project name format
    
    Args:
        name: Project name string
    
    Returns:
        Boolean indicating if valid
    """
    # Allow letters, numbers, spaces, hyphens, underscores
    pattern = r'^[a-zA-Z0-9\s\-_]+$'
    return bool(re.match(pattern, name))


def validate_code_input(code):
    """
    Validate code input for analysis
    
    Args:
        code: Code string
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not code or not code.strip():
        return False, "Code cannot be empty"
    
    if len(code) > 1000000:  # 1MB limit
        return False, "Code is too large (max 1MB)"
    
    return True, None


def sanitize_input(text):
    """
    Sanitize user input to prevent injection
    
    Args:
        text: Input text
    
    Returns:
        Sanitized text
    """
    # Remove potential harmful characters
    sanitized = text.strip()
    # Remove null bytes
    sanitized = sanitized.replace('\x00', '')
    return sanitized


def validate_file_upload(file_data):
    """
    Validate uploaded file
    
    Args:
        file_data: File data dictionary
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if 'filename' not in file_data:
        return False, "No filename provided"
    
    if 'content' not in file_data:
        return False, "No file content provided"
    
    # Validate file extension
    allowed_extensions = ['.csv', '.txt', '.json', '.py']
    filename = file_data['filename'].lower()
    
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
    
    # Check file size (max 10MB)
    if len(file_data['content']) > 10 * 1024 * 1024:
        return False, "File too large (max 10MB)"
    
    return True, None