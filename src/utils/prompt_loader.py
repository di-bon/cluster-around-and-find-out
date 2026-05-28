import os

def get_prompt(prompt_file: str) -> str:
    utils_dir = os.path.dirname(os.path.abspath(__file__))    
    src_dir = os.path.dirname(utils_dir)
    prompt_path = os.path.join(src_dir, "prompts", prompt_file)
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    return template