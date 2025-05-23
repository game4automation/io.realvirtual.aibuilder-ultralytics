import os
from tensorboard import program
import sys


def start_tensorboard(log_dir):

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create a TensorBoard instance
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])

    # Start TensorBoard
    url = tb.launch()
    return url


if __name__ == '__main__':
    
    args = sys.argv

    if len(args) < 2:
        print("please provide the path to the log dir")
        pass

    else:
        log_dir = args[1]
        print(start_tensorboard(log_dir))
        while True:
            pass