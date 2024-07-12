def load_combined_classes(file_path):
    """
    Load combined class names from a text file.
    
    Args:
    - file_path (str): Path to the text file containing class names.
    
    Returns:
    - combined_class_names (list): List of class names.
    - num_classes (int): Number of classes.
    """
    with open(file_path, 'r') as file:
        combined_class_names = [line.strip() for line in file.readlines()]
    num_classes = len(combined_class_names)
    return combined_class_names, num_classes