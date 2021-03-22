class Config:
  
    def __init__(self, mode = "train"):

      #network configure
      self.in_chans = 3
      self.out_chans = 3
      self.ch_mul = 64 

      #data configure
      self.mode = mode
      self.BatchSize = 10 #paper set it to 10
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
      self.epochs = 10 #set it to 50 for actual training
      self.psi = 0.5

      #paths configure
      if (mode == "train"):
        self.datapath = "../BSDS500/data/images/train" #set this for actual training: "/content/gdrive/My Drive/Kaggle/VOC2012/JPEGImages" 
      elif (mode == "test"):
        self.datapath = "../BSDS500/data/images/test"
      self.saving_path = "./trained_models" 
      #NCut loss configure
        #not implemented yet