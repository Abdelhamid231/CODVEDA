import os
import sys
import subprocess

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():

    print("      CODVEDA DATA SCIENCE INTERNSHIP - TASK RUNNER      ")

    print(" Select a task to execute its pipeline and generate results.\n")

def print_menu():
    print(" LEVEL 1 (Basic)")
    print("  [1] Task 1: Data Collection & Web Scraping")
    print("  [2] Task 2: Data Cleaning & Preprocessing")
    print("  [3] Task 3: Exploratory Data Analysis (EDA)\n")
    
    print(" LEVEL 2 (Intermediate)")
    print("  [4] Task 1: Predictive Modeling (Regression)")
    print("  [5] Task 2: Classification with Logistic Regression")
    print("  [6] Task 3: Clustering (Unsupervised Learning)\n")
    
    print(" LEVEL 3 (Advanced)")
    print("  [7] Task 1: Time Series Analysis")
    print("  [8] Task 2: NLP - Text Classification")
    print("  [9] Task 3: Neural Networks with TensorFlow/Keras\n")
    
    print("  [0] Run ALL Tasks sequentially")
    print("  [q] Quit\n")

def run_task(script_path):
    clear_screen()

    print(f" RUNNING: {script_path}")
    print("=" * 60 + "\n")
    
    # Use the same python executable that is running this menu
    python_exe = sys.executable
    
    try:
        subprocess.run([python_exe, script_path], check=True)
    except Exception as e:
        print(f"\n[ERROR] Failed to run {script_path}: {e}")

    input("Press Enter to return to the main menu...")

def main():
    tasks = {
        '1': 'Task1_Web_Scraping\\task1_web_scraping.py',
        '2': 'Task2_Data_Cleaning\\task2_data_cleaning.py',
        '3': 'Task3_EDA\\task3_eda.py',
        '4': 'Task4_Regression\\task4_regression.py',
        '5': 'Task5_Classification\\task5_classification.py',
        '6': 'Task6_Clustering\\clustering.py',
        '7': 'Task7_TimeSeries\\timeseries_analysis.py',
        '8': 'Task8_NLP\\nlp_classification.py',
        '9': 'Task9_NeuralNetworks\\neural_networks.py',
    }

    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("Enter your choice (0-9, or q to quit): ").strip().lower()
        
        if choice == 'q':
            clear_screen()
            print("Exiting Task Runner. Goodbye!\n")
            break
            
        elif choice == '0':
            for key in range(1, 10):
                run_task(tasks[str(key)])
                
        elif choice in tasks:
            run_task(tasks[choice])
            
        else:
            input("Invalid choice! Press Enter to try again...")

if __name__ == "__main__":
    main()




