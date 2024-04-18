import argparse


def read_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data', type=str, default='imdb-binary', help='dataset name:imdb-binary|imdb-multi|collab')        
    # gen_model
    parser.add_argument('--model_path', type=str,
                        default='model_save/', help='path to save model')
    parser.add_argument('--gen_layers', type=int, default=3, help='gnn layers')
    parser.add_argument('--gen_hidden_dim', type=int,
                        default=64, help='hidden dimension')
    parser.add_argument('--heads', type=int, default=8,
                        help='attention heads num')
    parser.add_argument('--num_out_heads', type=int,
                        default=8, help='attention heads num')
    parser.add_argument('--activation', type=str,
                        default='relu', help='activation func')
    parser.add_argument("--gen_feat_drop", type=float,
                        default=.4, help="input feature dropout")
    parser.add_argument("--gen_attn_drop", type=float,
                        default=.4, help="attention dropout")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--residual", type=int, default=1,
                        help="use residual connection")
    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--mean_encoder_type", type=str, default="mlp")
    parser.add_argument("--var_encoder_type", type=str, default="mlp")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--concat_hidden", action="store_true", default=False)
    parser.add_argument("--gen_optimizer", type=str, default="adam", help="adam||adadelta|adamw")

    #discriminator
    parser.add_argument("--dis_optimizer", type=str, default="rmsprop", help="adam|rmsprop|adadelta|adamw")
    parser.add_argument('--dis_layers', type=int, default=1, help='gnn layers')
    parser.add_argument('--dis_hidden_dim', type=int,
                        default=64, help='hidden dimension')
    parser.add_argument("--dis_feat_drop", type=float,
                        default=.0, help="input feature dropout")
    parser.add_argument("--dis_attn_drop", type=float,
                        default=.0, help="attention dropout")
    parser.add_argument("--dis_type", type=str, default="gat")
    parser.add_argument("--dis_mask_rate", type=float, default=0.0)
    parser.add_argument('--dis_activation', type=str,
                        default='leakyrelu', help='activation func')

    #GraphClassifier
    parser.add_argument('--hidden_dim', type=int,
                        default=64, help='hidden dimension')
    

    # train    
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

    parser.add_argument('--subgraphs', type=int, default=32*20, help='subgraphs num for pretrain')
    parser.add_argument('--sample_strategy', type=str, default='rw', help='random walk(rw) | 1hop neighbors(1hop)')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='max number of stage training iteration')
    parser.add_argument('--task', type=str, default='gc',
                        help='optional tesk: nc(node classification) | lp(link prediction) | gc(graph)')
    parser.add_argument('--debug', type=int, default=1, help='')
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument('--alpha', type=float, default=1.0, help='weight of dis loss')

    parser.add_argument('--pooling', type=str, default='mean', help='mean/max/sum')

    parser.add_argument('--use_cfg', type=bool, default=True, help='-')
    

    # others
    parser.add_argument('--repeats', type=int, default=5, help='repeat num')
    parser.add_argument('--device', default="0", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = read_args()
    for k, v in vars(args).items():
        print(k, ':', v)
