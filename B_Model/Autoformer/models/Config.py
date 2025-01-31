

# def main():
#     fix_seed = 2021
#     random.seed(fix_seed)
#     torch.manual_seed(fix_seed)
#     np.random.seed(fix_seed)

#     parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

#     # basic config
#     parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
#     parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
#     parser.add_argument('--model', type=str, required=True, default='Autoformer',
#                         help='model name, options: [Autoformer, Informer, Transformer]')

#     # data loader
#     parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
#     parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
#     parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
#     parser.add_argument('--features', type=str, default='M',
#                         help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
#     parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
#     parser.add_argument('--freq', type=str, default='h',
#                         help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
#     parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

#     # forecasting task
#     parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
#     parser.add_argument('--label_len', type=int, default=48, help='start token length')
#     parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

#     # model define
#     parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
#     parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
#     parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
#     parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
#     parser.add_argument('--c_out', type=int, default=7, help='output size')
#     parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
#     parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
#     parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
#     parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
#     parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
#     parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
#     parser.add_argument('--factor', type=int, default=1, help='attn factor')
#     parser.add_argument('--distil', action='store_false',
#                         help='whether to use distilling in encoder, using this argument means not using distilling',
#                         default=True)
#     parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
#     parser.add_argument('--embed', type=str, default='timeF',
#                         help='time features encoding, options:[timeF, fixed, learned]')
#     parser.add_argument('--activation', type=str, default='gelu', help='activation')
#     parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
#     parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

#     # optimization
#     parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
#     parser.add_argument('--itr', type=int, default=2, help='experiments times')
#     parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
#     parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
#     parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
#     parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
#     parser.add_argument('--des', type=str, default='test', help='exp description')
#     parser.add_argument('--loss', type=str, default='mse', help='loss function')
#     parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
#     parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

#     # GPU
#     parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
#     parser.add_argument('--gpu', type=int, default=0, help='gpu')
#     parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
#     parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


    
    
class Config:
    
    seq_len = 96 # input sequence length
    label_len = 48 # start token length
    pred_len = 24 # prediction sequence length
    
    enc_in = 7 # encoder input size
    dec_in = 7 # decoder input size
    c_out = 7 # output size
    
    d_model = 512 # dimension of model
    n_heads = 8 # num of heads
    e_layers = 2 # num of encoder layers
    d_layers = 1 # num of decoder layers
    d_ff = 2048 # dimension of fcn
    moving_avg = 25 # window size of moving average
    factor = 1 # attn factor
    dropout = 0.05 # dropout
    embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
    activation = 'gelu' # activation
    output_attention = False # whether to output attention in encoder
    freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
    
    
    def __init__(self,
                 seq_len:int=96, label_len:int=48, pred_len:int=24,
                enc_in:int=7, dec_in:int=7, c_out:int=7,    
                d_model:int=512, n_heads:int=8, e_layers:int=2, d_layers:int=1, d_ff:int=2048,
                moving_avg:int=25, factor:int=1, dropout:float=0.05, embed:str='timeF', activation:str='gelu',
                output_attention:bool=False, freq:str='h'):
        
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.factor = factor
        self.dropout = dropout
        self.embed = embed
        self.activation = activation
        self.output_attention = output_attention
        