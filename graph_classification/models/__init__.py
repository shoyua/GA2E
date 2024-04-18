from .edcoder import PreModel
from .discrimator import Discrimator
from .classify import GraphClassifier


def build_gen_model(args, num_features):    
    model = PreModel(
        in_dim=num_features,
        num_hidden=args.gen_hidden_dim,
        num_layers=args.gen_layers,
        nhead=args.heads,
        nhead_out=args.num_out_heads,
        activation=args.activation,
        feat_drop=args.gen_feat_drop,
        attn_drop=args.gen_attn_drop,
        negative_slope=args.negative_slope,
        residual=args.residual,
        encoder_type=args.encoder,
        decoder_type=args.decoder,
        mean_encoder_type=args.mean_encoder_type,
        var_encoder_type=args.var_encoder_type,
        mask_rate=args.mask_rate,
        norm=args.norm,
        drop_edge_rate=args.drop_edge_rate,
        replace_rate=args.replace_rate,
        concat_hidden=args.concat_hidden,
    )
    return model


def build_dis_model(args, num_features):
     
    model = Discrimator(
        in_dim=num_features,
        num_hidden=args.dis_hidden_dim,
        num_layers=args.dis_layers,
        nhead=args.heads,
        nhead_out=args.num_out_heads,
        activation=args.dis_activation,
        feat_drop=args.dis_feat_drop,
        attn_drop=args.dis_attn_drop,
        negative_slope=args.negative_slope,
        residual=args.residual,
        encoder_type=args.dis_type,
        # decoder_type=decoder_type,
        mask_rate=args.dis_mask_rate,
        norm=args.norm,
        drop_edge_rate=args.drop_edge_rate,
        replace_rate=args.replace_rate,
        concat_hidden=args.concat_hidden,
    )
    return model

def build_cls_model(args,num_class):
    
    model=GraphClassifier(
        in_feats=args.gen_hidden_dim,
        hidden_size=args.hidden_dim,
        num_classes=num_class
    )
    return model