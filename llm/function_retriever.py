from api_handler import APIHandler
from function_verifier import FunctionVerifier
import shutil
import os
import re

class FunctionRetriever:
    def __init__(self, api_key):
        self.api_handler = APIHandler(api_key)

    def retrieve_and_verify_function(self, prompt):
        function_code = self.api_handler.get_function_from_gpt(prompt)
        verifier = FunctionVerifier(function_code)
        
        if verifier.is_valid_function():
            function_name = verifier.get_function_name()
            print(f"Retrieved valid function: {function_name}")
            return function_code
        else:
            print("The retrieved code is not a valid Python function.")
            return function_code
        
    def copy_env(self, src_dir, new_name):
        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"The source directory {src_dir} does not exist.")
        
        # Get the parent directory of the source directory
        parent_dir = os.path.dirname(src_dir)
        
        # Construct the full path for the new directory
        new_dir_path = os.path.join(parent_dir, new_name)
        
        # Ensure the new directory does not already exist
        if os.path.exists(new_dir_path):
            raise FileExistsError(f"The destination directory {new_dir_path} already exists.")
        
        # Copy the source directory to the new location
        shutil.copytree(src_dir, new_dir_path)
        
        print(f"Copied {src_dir} to {new_dir_path}")
        return new_dir_path

        
    def replace_method_in_class(self, file_path, class_name, old_method_name, new_method_code):
        # Read the original file content
        with open(file_path, 'r') as file:
            file_content = file.read()

        pattern = (
            rf'(?s)(class\s+{class_name}\s*\(.*?\):.*?)(def\s+{old_method_name}\s*\(.*?\):.*?)(?=^\s*def\s+\w+\s*\(|^class\s+\w+\s*\(|\Z)'
        )
        
        modified_content = re.sub(pattern, rf'\1{new_method_code}', file_content, flags=re.MULTILINE)


        with open(file_path, 'w') as file:
            file.write(modified_content)
        
        print(f"Replaced method {old_method_name} in {file_path}")
