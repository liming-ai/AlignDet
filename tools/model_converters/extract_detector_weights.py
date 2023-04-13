import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts detector weights from a checkpoint')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    parser.add_argument('output', type=str, help='destination file name')
    parser.add_argument('--keep-heads', action='store_true', help='keep the weights of regression head')
    args = parser.parse_args()
    return args

# NOTE: unexpected key in source state_dict for `neck.lateral_convs.conv.bias` or `neck.fpn_convs.conv.bias` is caused by:
# https://github.com/open-mmlab/mmcv/blob/9709ff3f8c5a11946b85958a7f2fe1e6103d5153/mmcv/cnn/bricks/conv_module.py#L47
# We do not see the performance drop.
head_parameters = [
    # FCOS
    'retina_cls', 'retina_reg', 'conv_centerness',
    # RetinaNet
    'conv_cls', 'conv_reg',
    # DeformableDETR
    'cls_branches', 'obj_branches',
    'reg_branches.0.4', 'reg_branches.1.4',
    'reg_branches.2.4', 'reg_branches.3.4',
    'reg_branches.4.4', 'reg_branches.5.4',
    # 'reference_points', 'query_embedding',
    # both in MaskRCNN and DETR
    'fc_cls', 'fc_reg',
]


def main():
    args = parse_args()
    assert args.output.endswith('.pth')
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))

    output_dict = dict(state_dict=dict())

    for key, value in ck['state_dict'].items():
        if key.startswith('backbone'):
            key = key[9:]  # remove the first 'backbone'

            skip = False
            for param in head_parameters:
                if param in key:
                    if not args.keep_heads:
                        skip = True

            if skip:
                continue

            output_dict['state_dict'][key] = value
            print(f"{key}")

    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()