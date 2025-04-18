import argparse
from distutils.util import strtobool

def str2bool(v):
    return bool(strtobool(v))

def parse_args():
    parser = argparse.ArgumentParser()
    '''model settings'''
    parser.add_argument('-net', type=str, default='sam', help='net type')
    parser.add_argument('-seg_net', type=str, default='transunet', help='net type')
    parser.add_argument('-mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
    parser.add_argument('-pretrain', type=str2bool, default=True, help='using checkpoint or not')
    parser.add_argument('-sam_ckpt', default='logs/isic_2024_09_27_09_59_29/Model/checkpoint_best.pth', help='sam checkpoint address')
    parser.add_argument('-primitive', type=str2bool, default=True, help='is a primitive vit model or trained model')
    parser.add_argument('-use_chead', type=str2bool, default=False, help='Whether to use the classification head')
    parser.add_argument('-use_ud', type=str2bool, default=False, help='Whether to use u_decoder')
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-type', type=str, default='map', help='condition type:ave,rand,rand_map')
    parser.add_argument('-dim', type=int, default=512, help='dim_size')
    parser.add_argument('-depth', type=int, default=7, help='depth')
    parser.add_argument('-heads', type=int, default=16, help='heads number')
    parser.add_argument('-patch_size', type=int, default=16, help='patch_size')
    parser.add_argument('-mlp_dim', type=int, default=1024, help='mlp_dim')
    parser.add_argument('-uinch', type=int, default=1, help='input channel of unet')
    parser.add_argument('-base_weights', type=str, default=0, help='the weights baseline')
    parser.add_argument('-sim_weights', type=str, default=0, help='the weights sim')
    parser.add_argument('-weights', type=str, default=None, help='the weights file you want to test')
    parser.add_argument('-clip_model_path', type=str, default='/tmp/pas_segment/cached_biomedclip_model.pth', help='local clip model ckpt')
    '''training settings'''
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
    parser.add_argument('-num_folds', type=int, default=5, help='number of validation folds')
    parser.add_argument('-fold_epoch', type=int, default=8, help='folds num per epoch')
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-val_freq', type=int, default=2, help='interval between each validation')
    parser.add_argument('-vis_train', type=int, default=150, help='visualization interval for train')
    parser.add_argument('-vis_val', type=int, default=15, help='visualization interval for test')
    parser.add_argument('-reverse', type=bool, default=False, help='adversary reverse')
    '''device settings'''
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-sim_gpu', type=int, default=0, help='split sim to this gpu')
    parser.add_argument('-distributed', default='none', type=str, help='multi GPU ids to use')
    '''data settings'''
    parser.add_argument('-dataset', default='mri', type=str, help='dataset name')
    parser.add_argument('-exp_name', type=str, default='isic', help='data type')
    parser.add_argument('-data_path', type=str, default='../sda2/placenta_mri_100', help='The path of segmentation data')
    parser.add_argument('-all_classes', nargs='+', default=['background', 'placenta'], help='All classes names included in the dataset')
    parser.add_argument('-classes', nargs='+', default=['background', 'placenta'], help='classes you wanna segment')
    parser.add_argument('-image_size', type=int, default=512, help='image_size')
    parser.add_argument('-mask_size', type=int, default=512, help='mask_size')
    parser.add_argument('-w', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-thd', type=bool, default=False, help='3d or not')
    parser.add_argument('-chunk', type=int, default=96, help='crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4, help='sample pos and neg')
    parser.add_argument('-roi_size', type=int, default=96, help='resolution of roi')
    parser.add_argument('-evl_chunk', type=int, default=None, help='evaluation chunk')
    parser.add_argument('-mid_dim', type=int, default=None , help='middle dim of adapter or the rank of lora matrix')
    parser.add_argument('-multimask_output', type=int, default=1 , help='the number of masks output for multi-class segmentation, set 2 for REFUGE dataset.')

    '''java'''


    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the SSAM model checkpoint",
    )

    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Path to input image",
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help="Directory to save output masks",
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        required=False,
        choices=['point', 'box', 'mask'],
        help="Type of prompt to use",
    )

    parser.add_argument(
        "--targets",
        type=str,
        nargs='+',
        required=True,
        help="List of target organs to segment",
    )

    parser.add_argument(
        "--prompts",
        type=str,
        required=False,
        help="JSON string containing prompts for each target. Format depends on prompt_type.",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Input image size",
    )

    parser.add_argument(
        "--gpu_device",
        type=int,
        default=0,
        help="GPU device ID",
    )

    opt = parser.parse_args()

    return opt