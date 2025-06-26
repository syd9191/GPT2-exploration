import os
import json
import torch 
import random
import numpy as np
from pathlib import Path 
from typing import Union
import matplotlib.pyplot as plt
import torch.distributed as dist





# ------- Custom -------
from dataloaders.dir_streaming_dataset import DirStreamingDataset



class ModelTracker:
    def __init__(self, 
                model:torch.nn.Module,
                batch_size:int, 
                seq_length:int, 
                total_tokens:int, 
                plot_path:str, 
                model_path:str,
                stats_path:str,
                loss_path:str):
        
        self.losses:list=[]
        self.batch_size:int=batch_size
        self.seq_length:int=seq_length
        self.total_tokens:int=total_tokens
        self.last_step_updated:int=0
        self.best_loss:float=float('inf')
        self.plot_path:Path=Path(plot_path)
        self.model_path:Path=Path(model_path)
        self.stats_path:Path=Path(stats_path)
        self.loss_path:Path=Path(loss_path)
        self.best_weights=None
        self.model=model
        

        self.stats={"Tokens Exposed": 0,
                    "Steps Trained": 0,
                    "Epochs": 0,
                    "Best Loss": float('inf'),
                    "Time Spent Training":0,
                    "Model Path":str(self.model_path),
                    "Batch Size": self.batch_size
                    }
        
        for path in [self.plot_path, self.model_path, self.stats_path, self.loss_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        self._load()

    def _load(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, "r") as f:
                self.stats.update(json.load(f))
        
        if self.loss_path.exists():
            self.losses = np.load(self.loss_path).tolist()

    def update_all(self,
                   time_interval:float,
                   curr_step:int):
        if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank()==0):
            self.update(time_interval=time_interval,
                    curr_step=curr_step,
                    last_step_updated=self.last_step_updated)
            self.save_best_weights()
            self.save_stats()
            self.plot_loss()

            self.last_step_updated=curr_step

    def save_stats(self):
    # Only save from the main process (rank 0)
        if dist.is_initialized():
            if dist.get_rank() != 0:
                return  # Skip saving in all but rank 0

        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.stats_path, "w") as f:
                json.dump(self.stats, f, indent=4)
        except Exception as e:
            raise ValueError(f"Json object cannot be saved: {e}")
    
    def update(self,
           time_interval:float,
           curr_step:int,
           last_step_updated:int):
        """
        Updates Stats based on how many steps have been taken
        """
        steps=curr_step-last_step_updated

        self.stats["Steps Trained"] += steps
        self.stats["Epochs"] += (self.batch_size*self.seq_length*steps)/self.total_tokens
        self.stats["Tokens Exposed"] += self.batch_size*self.seq_length*steps
        self.stats["Time Spent Training"] += time_interval
        self.stats["Best Loss"] = self.best_loss

        np.save(self.loss_path, np.array(self.losses)) #store loss in a npy file, storing in json is retarded
        
    def add_loss(self, 
                 loss:float):
        self.losses.append(loss)
        if loss<self.best_loss and self.model is not None:
            self.best_loss=loss
            self.best_weights=self.model.state_dict()

    def save_best_weights(self):
        if self.best_weights is not None:
            torch.save(self.best_weights, self.model_path)
    
    def plot_loss(self, 
                  show:bool=False):
        print("plotting loss here")
        plt.figure(figsize=(8, 5))
        plt.plot(self.losses, label="Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=300)
        if show:
            plt.show()
        plt.close() 
        