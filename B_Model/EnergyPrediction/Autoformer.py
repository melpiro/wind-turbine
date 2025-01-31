

from numpy_typing import np, ax
import torch


from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Autoformer.models.Config import Config
from B_Model.Autoformer.models.Autoformer import Model as Autoformer



class Model(AbstactModel):

    name = "Autoformer"
    
    
    def __init__(self, CTX:dict):
        
        # load context
        self.CTX = CTX
        self.dropout = CTX["DROPOUT"]

        # save the number of training steps
        self.nb_train = 0
        
        # model definintion
        self.model = Autoformer(Config(
            seq_len=self.CTX["HISTORY"]+self.CTX["LOOK_AHEAD"],
            label_len=self.CTX["LOOK_AHEAD"],
            pred_len=self.CTX["LOOK_AHEAD"],
            enc_in=self.CTX["FEATURES_IN"] - self.CTX["EMBEDDING_SIZE"],
            dec_in=self.CTX["FEATURES_IN"] - self.CTX["EMBEDDING_SIZE"],
            c_out=1,
            d_ff=self.CTX["D_FF"],
            d_model=self.CTX["D_MODEL"],
            n_heads=self.CTX["N_HEADS"],
            e_layers=self.CTX["E_LAYERS"],
            d_layers=self.CTX["D_LAYERS"],
            
        ))
        
        self.loss = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.CTX["LEARNING_RATE"])
        
    def __get_input__(self, x:np.float32_3d[ax.sample, ax.time, ax.feature]):
        x_enc = x[:, :self.CTX["HISTORY"], :-self.CTX["EMBEDDING_SIZE"]]
        x_dec = x[:, self.CTX["HISTORY"]:, :-self.CTX["EMBEDDING_SIZE"]]
        
        if (self.CTX["EMBEDDING_SIZE"] > 0):
            x_mark_enc = torch.tensor(x[:, :self.CTX["HISTORY"], -self.CTX["EMBEDDING_SIZE"]:])
            x_mark_dec = torch.tensor(x[:, :, -self.CTX["EMBEDDING_SIZE"]:])
        else:
            x_mark_enc = torch.zeros(len(x), self.CTX["HISTORY"], 4)
            x_mark_dec = torch.zeros(len(x), self.CTX["HISTORY"]+self.CTX["LOOK_AHEAD"], 4)
        # concat x_mark_enc + x_mark_dec
        return torch.tensor(x_enc), x_mark_enc, \
               torch.tensor(x_dec), x_mark_dec
        
        
    def predict(self, x:np.float32_3d[ax.sample, ax.time, ax.feature]) -> np.float32_3d[ax.sample, ax.time, ax.feature]:
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.__get_input__(x)
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, None).detach().numpy()[:,:,0]
    
    
    def compute_loss(self, x:np.float32_3d[ax.sample, ax.time, ax.feature], y:np.float32_3d[ax.sample, ax.time, ax.feature]) -> """tuple[
                np.float32, 
                np.float32_3d[ax.sample, ax.time, ax.feature]]""":
                    
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.__get_input__(x)
        y = torch.tensor(y)
        y_ = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)[:,:,0]
        return self.loss(y_, y).detach().numpy(), y_.detach().numpy()
    
    
    def training_step(self, x:np.float32_3d[ax.sample, ax.time, ax.feature], y:np.float32_3d[ax.sample, ax.time, ax.feature]):
    
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.__get_input__(x)
        y = torch.tensor(y)
        self.opt.zero_grad()
        
        y_ = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)[:,:,0]
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
            
            
        
        
        