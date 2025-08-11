from pytorch_fid import fid_score

clean_path = 'assets/datasets/CelebA-HQ_align2'
adv_path = 'assets/datasets/val_opti_xT_L1_10_tv_200/irse50_save_eps0.03_Tenc100_iter50_Tatk5_Tstart5_Tinf100_atk1_atkinf4_repeat1_skip0_lam0.0'
batch_size = 100
device = 'cuda'
fid = fid_score.calculate_fid_given_paths(
            [clean_path, adv_path],
            batch_size,
            device=device,
            dims=2048,
            num_workers=0)

print(fid)

# import torch
# import torchvision
# import torchvision.transforms as transforms
# from pytorch_fid import fid_score
#
# clean_path = 'assets/datasets/CelebA-HQ_align2'
# adv_path = 'assets/datasets/val_opti_xT/irse50_save_eps0.03_Tenc100_iter50_Tatk5_Tstart5_Tinf100_atk1_atkinf4_repeat1_skip0_lam0.0'
#
# # 加载预训练的Inception-v3模型
# inception_model = torchvision.models.inception_v3(pretrained=True)
#
# # 定义图像变换
# # transform = transforms.Compose([
# #     transforms.Resize(299),
# #     transforms.CenterCrop(299),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# # ])
#
# # 计算FID距离值
# fid_value = fid_score.calculate_fid_given_paths([clean_path, adv_path],
#                                                  inception_model)
# print('FID value:', fid_value)

