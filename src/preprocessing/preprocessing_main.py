import logging
from preprocessing import task

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    task.run()
