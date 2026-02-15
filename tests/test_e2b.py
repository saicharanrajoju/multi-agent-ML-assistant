import os
import sys
# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.code_executor import CodeExecutor
from src.tools.file_utils import get_dataset_path

def test_code_executor():
    print("=" * 60)
    print("🚀 Testing E2B Code Executor with REAL Dataset")
    print("=" * 60)

    try:
        # 1. Get real dataset path
        dataset_path = get_dataset_path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        print(f"✅ Found real dataset at: {dataset_path}")

        with CodeExecutor() as executor:
            # 2. Upload file
            print("\n⬆️ Uploading dataset...")
            sandbox_path = executor.upload_file(dataset_path)
            
            # 3. Run analysis code
            print("\n▶️ Running analysis code...")
            code = f"""
import pandas as pd
df = pd.read_csv('{sandbox_path}')
print(f"Dataset shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
print(f"Target distribution:\\n{{df['Churn'].value_counts()}}")
"""
            result = executor.execute_code(code)
            
            # 4. Print result
            print("\n📊 Execution Result:")
            print(f"Success: {result['success']}")
            if result['stdout']:
                print(f"STDOUT:\n{result['stdout']}")
            if result['stderr']:
                print(f"STDERR:\n{result['stderr']}")
            if result['error']:
                print(f"ERROR:\n{result['error']}")
                
    except Exception as e:
        print(f"\n❌ Test Failed with Exception: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ Test Completed")
    print("=" * 60)

if __name__ == "__main__":
    test_code_executor()
