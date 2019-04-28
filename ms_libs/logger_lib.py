class logger(object):

    def __init__(self, filename, message):
        self.filename = filename

        # create new file, OR overwrite existing file    
        with open(self.filename, 'w+') as f:
            f.write(message)

    def update_logger(self, message):
        
        with open(self.filename, 'a') as f:
            f.write('\n')
            f.write(message)
