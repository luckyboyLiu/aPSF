from optimization import Architect, PromptStructure
from llm_apis import get_llm

def quick_test():
    """
    A simple function used for quickly testing the core functions of the framework.
    """
    print("--- Running a quick test of the aPSF framework ---")

    # 1. Test Architect
    print("\nTesting Architect...")
    architect = Architect(architect_llm_id="architect")
    task_desc = "Answer grade-school math word problems."
    example = "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAnswer: 6"
    
    try:
        structure = architect.discover_structure(task_desc, example)
        print("Architect discovered structure:")
        print(structure.render())
    except Exception as e:
        print(f"Could not run Architect test. Have you set up your API keys in config.py? Error: {e}")
        return

    # 2. Test LLM API
    print("\nTesting Worker LLM...")
    try:
        worker_llm = get_llm("worker")
        response = worker_llm.generate("What is 2+2?")
        print(f"Worker LLM response: {response}")
    except Exception as e:
        print(f"Could not run Worker LLM test. Have you set up your API keys in config.py? Error: {e}")


if __name__ == "__main__":
    quick_test() 