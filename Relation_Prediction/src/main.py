
import argparse
from data_loader import load_data
from train import train
import os
from datetime import datetime
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def print_setting(args):

    if not hasattr(args, 'use_context'):
         print("Warning: use_context or use_path argument missing.")
         return
    if not args.use_context:
        print("Warning: Neither context nor path usage is enabled.")

    print()
    print('=============================================')
    print(f'dataset: {args.dataset}')
    print(f'epoch: {args.epoch}')
    print(f'batch_size: {args.batch_size}')
    print(f'dim: {args.dim}')
    print(f'l2: {args.l2}')
    print(f'lr: {args.lr}')
    print(f'feature_type: {args.feature_type}')
    if hasattr(args, 'num_bin') and args.num_bin is not None:
         print(f'num_bin: {args.num_bin}')


    print(f'use relational context: {args.use_context}')
    if args.use_context:
        print(f'context_hops: {args.context_hops}')
        print(f'neighbor_samples: {args.neighbor_samples}')
        print(f'neighbor_agg: {args.neighbor_agg}')
        if hasattr(args, 'tok_k') and args.tok_k is not None:
             print(f'tok_k: {args.tok_k}')
        if hasattr(args, 'neighbor_agg') and args.neighbor_agg == 'attention':
            print(f'num_attention_heads: {args.num_heads}')

    print()
    print('=============================================')
    print('           CFM Module Settings               ')
    print('=============================================')
    print(f'CFM Variant: {args.cfm_variant}')
    print(f'CFM MLP Dimension: {args.cfm_model_dim}')
    print(f'CFM Sigma: {args.cfm_sigma}')
    if args.cfm_variant == 'OTCFM':
        print(f'CFM OT Method: {args.cfm_ot_method}')
    print()


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Embedding Training")


    parser.add_argument('--cuda', help='use gpu', default = True, action='store_true')


    parser.add_argument('--dataset', type=str, required=True, help='dataset name (FB15k, FB15k-237, wn18, wn18rr, NELL995, DDB14)')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features')

    parser.add_argument('--use_context', action='store_true', help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=16, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='concat', choices=['mean', 'concat', 'cross', 'attention'], help='neighbor aggregator')
    parser.add_argument('--tok_k', type=int, default=8, help='tok-k value')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention head')

    parser.add_argument('--lamda', type=float, default=1.2, help='train cfm')

    parser.add_argument('--cfm_model_dim', type=int, default=128, help='Dimension of the MLP in the CFM model')
    parser.add_argument('--cfm_variant', type=str, default='OTCFM',
                        choices=['CFM', 'OTCFM', 'TargetCFM', 'SBCFM', 'VPCFM'],
                        help='Variant of Conditional Flow Matching to use')
    parser.add_argument('--cfm_sigma', type=float, default=0.1, help='Sigma value for CFM, controlling noise level')
    parser.add_argument('--cfm_ot_method', type=str, default='exact',
                        help='OT method for Schrodinger Bridge CFM (SBCFM)')

    args = parser.parse_args()

    if not args.use_context and not args.use_path:
        parser.error("At least one of --use_context or --use_path must be specified.")
    if args.cuda:
         print("Using CUDA")

    else:
         print("Using CPU")


    print_setting(args)
    data = load_data(args)
    print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    train(args, data)


if __name__ == '__main__':
    main()