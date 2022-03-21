import torch, os, sys, torchvision, argparse
import time, math
import numpy as np
import torchvision.transforms as tfs


from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
from option import opt, log_dir
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from metrics import psnr, ssim
from models.AECRNet import *
from models.CR import *
import random
from PIL import Image
import json
from torchvision.transforms import functional as FF


warnings.filterwarnings('ignore')

models_={
	'cdnet': Dehaze(3, 3),
}


start_time = time.time()
start_time = time.time()
model_name = opt.model_name
steps = opt.eval_step * opt.epochs
T = steps

def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
	return lr


def train(net, trainset, valset, optim, criterion):
	losses = []
	start_step = 0
	max_ssim = 0
	max_psnr = 0
	ssims = []
	psnrs = []

	train_list=os.listdir('./trainset/clear_img')
	if './ipynb_checkpoints' in train_list:
		train_list.remove('./ipynb_checkpoints')
	val_list=os.listdir('./valset/clear_img')

	print(os.path.exists(opt.model_dir))
	if True:
		ckp = torch.load('./trained_models/'+'its_train_cdnet_test.pk')

		print(f'resume from {opt.model_dir}')
		losses = ckp['losses']
		net.load_state_dict(ckp['model'])
		optim.load_state_dict(ckp['optimizer'])
		start_step = ckp['step']
		max_ssim = ckp['max_ssim']
		max_psnr = ckp['max_psnr']
		psnrs = ckp['psnrs']
		ssims = ckp['ssims']
		print(f'max_psnr: {max_psnr} max_ssim: {max_ssim}')
		print(f'start_step:{start_step} start training ---')
	else:
		print('train from scratch *** ')
		print(start_step)
	for step in range(start_step+1, steps+1):
		net.train()
		lr = opt.lr
		if not opt.no_lr_sche:
			lr = lr_schedule_cosdecay(step,T)
			for param_group in optim.param_groups:
				param_group["lr"] = lr
		
		if '.ipynb_checkpoints' in train_list:
			train_list.remove('.ipynb_checkpoints')
		random.shuffle(train_list)
		train_use=train_list[0:32]
		clean=torch.zeros([0,3,752,1848])
		dirty=torch.zeros([0,3,752,1848])
		for i in train_use:
			image_input_clean=Image.open('./trainset/clear_img'+'/'+i)
			image_input_dirty=Image.open('./trainset/haze_image'+'/'+i)
			image_input_clean=image_input_clean.resize((1848,752))
			image_input_dirty=image_input_dirty.resize((1848,752))
			rand_hor = random.randint(0, 1)
			rand_rot = random.randint(0, 3)
			image_input_clean = tfs.RandomHorizontalFlip(rand_hor)(image_input_clean)
			image_input_dirty = tfs.RandomHorizontalFlip(rand_hor)(image_input_dirty)
			if rand_rot:
				image_input_clean = FF.rotate(image_input_clean, 90*rand_rot)
				image_input_dirty = FF.rotate(image_input_dirty, 90*rand_rot)
			inputs_clean=tfs.ToTensor()(image_input_clean)
			inputs_dirty=tfs.ToTensor()(image_input_dirty)
			inputs_clean=inputs_clean.unsqueeze(0)
			inputs_dirty=inputs_dirty.unsqueeze(0)
			#print(inputs_clean.shape)
		clean=torch.cat((clean,inputs_clean))
		dirty=torch.cat((dirty,inputs_dirty))
		x = clean.to(opt.device)
		y = dirty.to(opt.device)


		out= net(x)

		loss_vgg7, all_ap, all_an, loss_rec = 0, 0, 0, 0
		if opt.w_loss_l1 > 0:
			loss_rec = criterion[0](out, y)
		if opt.w_loss_vgg7 > 0:
			loss_vgg7, all_ap, all_an = criterion[1](out, y, x)

		loss = opt.w_loss_l1*loss_rec + opt.w_loss_vgg7*loss_vgg7
		loss.backward()
		
		optim.step()
		optim.zero_grad()
		losses.append(loss.item())

		print(f'\rloss:{loss.item():.5f} l1:{opt.w_loss_l1*loss_rec:.5f} contrast: {opt.w_loss_vgg7*loss_vgg7:.5f} all_ap:{all_ap:.5f} all_an:{all_an:.5f}| step :{step}/{steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',end='', flush=True)






		# with SummaryWriter(logdir=log_dir, comment=log_dir) as writer:
		# 		# 	writer.add_scalar('data/loss', loss, step)
		# 		# 	writer.add_scalar('data/loss_l1', loss_rec, step)
		# 		# 	writer.add_scalar('data/loss_vgg4', loss_vgg4, step)
		# 		# 	writer.add_scalar('data/all_ap', all_ap, step)
		# 		# 	writer.add_scalar('data/all_an', all_an, step)
		# 		# 	writer.add_scalar('data/m1', m1, step)

		if step % opt.eval_step == 0:
			epoch = int(step / opt.eval_step)

			save_model_dir = opt.model_dir
			with torch.no_grad():
				ssim_eval, psnr_eval = val(net, 'valset')

			log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}'

			

			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)

			if psnr_eval > max_psnr:
				max_ssim = max(max_ssim, ssim_eval)
				max_psnr = max(max_psnr, psnr_eval)
				save_model_dir = opt.model_dir + 'its_train_cdnet_test.pk'
				print(
					f'\n model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')

			torch.save({
				'epoch': epoch,
				'step': step,
				'max_psnr': max_psnr,
				'max_ssim': max_ssim,
				'ssims': ssims,
				'psnrs': psnrs,
				'losses': losses,
				'model': net.state_dict(),
				'optimizer': optim.state_dict()
			}, save_model_dir)

	np.save(f'./numpy_files/{model_name}_{steps}_losses.npy', losses)
	np.save(f'./numpy_files/{model_name}_{steps}_ssims.npy', ssims)
	np.save(f'./numpy_files/{model_name}_{steps}_psnrs.npy', psnrs)

def test(net,loader_test):
	net.eval()
	torch.cuda.empty_cache()
	ssims = []
	psnrs = []

	for i, (inputs, targets) in enumerate(loader_test):
		inputs = inputs.to(opt.device);targets = targets.to(opt.device)
		with torch.no_grad():
			pred, _, _, _ = net(inputs)

		ssim1 = ssim(pred, targets).item()
		psnr1 = psnr(pred, targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)

	return np.mean(ssims), np.mean(psnrs)

def val(net,loader_test):
	net.eval()
	torch.cuda.empty_cache()
	ssims = []
	psnrs = []
	
	val_list=os.listdir('./valset/clear_img')	
	if '.ipynb_checkpoints' in val_list:
            val_list.remove('.ipynb_checkpoints')
	random.shuffle(val_list)
	i=val_list[0]
	image_input_clean=Image.open('./valset/clear_img'+'/'+i)
	image_input_dirty=Image.open('./valset/haze_image'+'/'+i)
	image_input_clean=image_input_clean.resize((1848,752))
	image_input_dirty=image_input_dirty.resize((1848,752))
	rand_hor = random.randint(0, 1)
	rand_rot = random.randint(0, 3)
	inputs_clean=tfs.ToTensor()(image_input_clean)
	inputs_dirty=tfs.ToTensor()(image_input_dirty)
	inputs_clean=inputs_clean.unsqueeze(0)
	inputs_dirty=inputs_dirty.unsqueeze(0)
	x = inputs_clean.to(opt.device)
	y = inputs_dirty.to(opt.device)


	with torch.no_grad():
		pred = net(x)

	ssim1 = ssim(pred, y).item()
	psnr1 = psnr(pred, y)
	ssims.append(ssim1)
	psnrs.append(psnr1)
	print(ssims)
	print(psnrs)
	return np.mean(ssims), np.mean(psnrs)

def set_seed_torch(seed=2018):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

	set_seed_torch(666)

	net = models_[opt.net]
	net = net.to(opt.device)
	epoch_size = 16
	print("epoch_size: ", epoch_size)
	if opt.device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True

	pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print("Total_params: ==> {}".format(pytorch_total_params))

	criterion = []
	criterion.append(nn.L1Loss().to(opt.device))
	criterion.append(ContrastLoss(ablation=opt.is_ab))
	


	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	train(net, './trainset', './valset', optimizer, criterion)
	
