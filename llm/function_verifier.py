import ast

class FunctionVerifier:
    def __init__(self, function_code):
        self.function_code = function_code

    def is_valid_function(self):
        try:
            parsed_code = ast.parse(self.function_code)
            if len(parsed_code.body) != 1 or not isinstance(parsed_code.body[0], ast.FunctionDef):
                return False
            return True
        except SyntaxError:
            return False

    def get_function_name(self):
        parsed_code = ast.parse(self.function_code)
        if isinstance(parsed_code.body[0], ast.FunctionDef):
            return parsed_code.body[0].name
        return None