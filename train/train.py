from mmengine import Runner  
from mmpose.utils import register_all_modules  
from mmengine.hooks import Hook  
import threading  
from PyQt6.QtCore import QThread, pyqtSignal  
  
class GUIProgressHook(Hook):  
    def __init__(self, gui_callback):  
        self.gui_callback = gui_callback  
      
    def after_train_iter(self, runner):  
        # 更新GUI進度條  
        self.gui_callback(runner.iter, runner.max_iters)
        
class TrainingThread(QThread):  
    progress_updated = pyqtSignal(int, int)  
      
    def run(self):  
        # 在這裡運行MMPose訓練  
        runner = Runner.from_cfg(self.config)  
        runner.train()
        
# 註冊所有MMPose模塊  
register_all_modules()  
  
# 在PyQt6應用程序中創建Runner  
runner = Runner.from_cfg(config)  
runner.train()