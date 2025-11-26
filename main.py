import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def print_banner():
    print("\n" + "=" * 80)
    print("  HADAR - Hallucination Detection via Agentic Reasoning")
    print("  Cross-Model LLM Debate System")
    print("=" * 80 + "\n")


def print_menu():
    print("\n📋 MAIN MENU")
    print("-" * 60)
    print("1. Run LLM Debate (Standard HADAR)")
    print("2. Run LLM Debate (Adaptive Model Selection)")
    print("3. Exit")
    print("-" * 60)


def check_api_keys():
    from hadar.config import validate_api_keys
    return validate_api_keys()


def run_debate():
    print("\n🎤 Starting LLM Debate System...")
    print("-" * 60)
    
    from hadar.debate import run_debate as run_debate_fn, DEBATE_TOPIC
    
    try:
        run_debate_fn(DEBATE_TOPIC)
        print("\n✅ Debate completed successfully!")
    except Exception as e:
        print(f"\n❌ Error running debate: {e}")
        import traceback
        traceback.print_exc()


def run_debate_adaptive():
    print("\n🎤 Starting Adaptive LLM Debate System...")
    print("-" * 60)
    print("\n📝 Enter debate topic (or press Enter for default):")
    topic = input("> ").strip()
    
    if not topic:
        from hadar.debate import DEBATE_TOPIC
        topic = DEBATE_TOPIC
    
    print(f"\n🎯 Topic: {topic}")
    print("\n🔍 Selecting optimal models for topic...")
    
    from hadar.adaptive import select_models_for_topic, run_adaptive_debate
    
    try:
        selected_models = select_models_for_topic(topic)
        print(f"\n✅ Selected {len(selected_models)} consistency models")
        for i, model in enumerate(selected_models, 1):
            print(f"  {i}. {model}")
        
        run_adaptive_debate(topic, selected_models)
        print("\n✅ Adaptive debate completed successfully!")
    except Exception as e:
        print(f"\n❌ Error running adaptive debate: {e}")
        import traceback
        traceback.print_exc()


def main():
    print_banner()
    
    if not check_api_keys():
        print("Exiting...")
        return 1
    
    while True:
        print_menu()
        
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                run_debate()
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                run_debate_adaptive()
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                print("\n👋 Goodbye!")
                break
                
            else:
                print("\n❌ Invalid choice. Please select 1-3.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    exit_code = main() or 0
    sys.exit(exit_code)
