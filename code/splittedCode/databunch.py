class DataBunch():

    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c
        
    @property
    def train_ds(self): 
        return self.train_dl.dataset
    
    @property
    def valid_ds(self):
        return self.valid.dl.dataset

