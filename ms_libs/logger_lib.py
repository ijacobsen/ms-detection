class logger(object):

    def __init__(self, filename, message):
        self.filename = filename
        self.filename_meta = filename+'meta'

        # create new file, OR overwrite existing file    
        with open(self.filename, 'w+') as f:
            f.write(message)

        with open(self.filename_meta, 'w+') as f:
            f.write(message)

    def update_logger(self, message):
        
        with open(self.filename, 'a') as f:
            f.write('\n')
            f.write(str(message))

    def update_meta(self, message):

        with open(self.filename_meta, 'a') as f:
            f.write('\n')
            f.write(message)

