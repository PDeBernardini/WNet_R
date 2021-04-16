class Config:
  
    def __init__(self, mode = "train"):

      # network configure
      self.in_chans = 3
      self.out_chans = 3
      self.ch_mul = 64 

      # data configure
      self.mode = mode
      self.BatchSize = 10 #paper set it to 10
      self.Shuffle = True
      self.LoadThread = 2
      self.inputsize = (224,224)
      self.dropout = 0.65 #set to 0.65

      # partition configure
      # based on: https://github.com/lwchen6309/unsupervised-image-segmentation-by-WNet-with-NormalizedCut
      self.K = 21 

      # training configure
      self.init_lr = 0.003 #set to 0.003
      self.lr_decay = 0.1
      self.lr_decay_iter = 1 #set it to 1 for training
      self.epochs = 1 #set it to 50 for training
      self.psi = 0.5

      # paths configure
      if (mode == "train" or mode == "tune"):
        self.datapath = "../Kaggle/VOC2012/JPEGImages" #set this for actual training: "../Kaggle/VOC2012/JPEGImages" 
      elif (mode == "test"):
        self.datapath = "../BSDS500/data/images/test"
      self.saving_path = "./trained_models" 

      # postprocessing
      self.CRF_num = 15 #number of postprocessing steps

      # fine tuning
      self.alpha = 1e-2
      self.lr_tune = 3e-4
      self.epochs_tune = 1

      # temporary option (remove later)
      self.early_stop = None
       
      
   
