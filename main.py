"""
main.py

Entry point to run the training or other project steps.
"""

from training.ddp_training import main_training_loop

if __name__ == "__main__":
    main_training_loop()