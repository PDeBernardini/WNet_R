class Config:
  
    def __init__(self, mode = "train"):

      #network configure
      self.in_chans = 3
      self.out_chans = 3
      self.ch_mul = 64 

      #data configure
      self.mode = mode
      if (mode == "train"):
        self.datapath = "/content/gdrive/My Drive/Kaggle/VOC2012/JPEGImages" 
      elif (mode == "test"):
        self.datapath = "../BSDS500/data"
      self.BatchSize = 10
        #number of images extracted from the image folder simultaneously
      self.SupBatchSize = self.BatchSize * 5
      self.Shuffle = True
      self.LoadThread = 4
      self.inputsize = (224,224)
      self.dropout = 0.65

      #partition configure
        #based on: https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut
      self.K = 21 

      #training configure
      self.init_lr = 0.003
      self.lr_decay = 0.1
      self.lr_decay_iter = 1000
      self.max_iter = 50000 
      self.psi = 0.5

      #NCut loss configure
        #not implemented yet