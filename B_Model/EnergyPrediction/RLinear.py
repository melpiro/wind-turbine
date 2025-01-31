from B_Model.AbstractModel import Model as AbstactModel


from numpy_typing import np, ax
import os
import torch


from B_Model.EasyTSF.model.RLinear import RLinear




class Model(AbstactModel):

    name = "RLinear"
    
    def __init__(self, CTX:dict):
        
        # load context
        self.CTX = CTX
        self.dropout = CTX["DROPOUT"]

        # save the number of training steps
        self.nb_train = 0
        
        # model definintion
        self.model = RLinear(self.CTX["INPUT_LEN"], self.CTX["OUTPUT_LEN"], self.CTX["FEATURES_IN"], drop=self.dropout)
        
        self.loss = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.CTX["LEARNING_RATE"])
        
        
    def predict(self, x:np.float32_3d[ax.sample, ax.time, ax.feature]) -> np.float32_3d[ax.sample, ax.time, ax.feature]:
        x = torch.tensor(x)
        return self.model(x, None).detach().numpy()
    
    
    def compute_loss(self, x:np.float32_3d[ax.sample, ax.time, ax.feature], y:np.float32_3d[ax.sample, ax.time, ax.feature]) -> """tuple[
                np.float32, 
                np.float32_3d[ax.sample, ax.time, ax.feature]]""":
                    
        x = torch.tensor(x)
        y = torch.tensor(y)
        y_ = self.model(x, None)
        return self.loss(y_, y).detach().numpy(), y_.detach().numpy()
    
    
    def training_step(self, x:np.float32_3d[ax.sample, ax.time, ax.feature], y:np.float32_3d[ax.sample, ax.time, ax.feature]):
        
        x = torch.tensor(x)
        y = torch.tensor(y)
        self.opt.zero_grad()
        
        y_ = self.model(x, None)
        loss = self.loss(y_, y)
        loss.backward()
        self.opt.step()
        
        
        self.nb_train += 1
        loss = loss.detach().numpy()
        output = y_.detach().numpy()
        
        return loss, output
    
    def get_variables(self):
        params = []
        for p in self.model.parameters():
            params.append(p.detach().numpy())
        return params
    
    def set_variables(self, variables):
        for p, v in zip(self.model.parameters(), variables):
            p.data = torch.tensor(v)
            
            
        
        
        