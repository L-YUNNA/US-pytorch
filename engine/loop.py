# https://velog.io/@heomollang/Pytorch-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B6%84%EB%A5%98-with-ViT
import tqdm
import torch
import numpy as np

def train_loop(dataloader,model,loss_fn,optimizer,device):
	epoch_loss = 0 
	model.train() 
	for (image, label) in tqdm.tqdm(dataloader): 
		img_input = image.to(device)
		target = label.to(device)

		pred = model(img_input)

		loss = loss_fn(pred, target)   
		optimizer.zero_grad() 
		loss.backward()  
		optimizer.step() 

		epoch_loss += loss.item() 

	epoch_loss /= len(dataloader) 

	return epoch_loss


@torch.no_grad() 
def test_loop(dataloader,model,loss_fn,device): 
	epoch_loss = 0
	model.eval() 

	pred_list = []
	true_list = []
	softmax = torch.nn.Softmax(dim=1) 

	for (image, label) in tqdm.tqdm(dataloader):   
		img_input = image.to(device)
		target = label.to(device)

		pred = model(img_input)

		if target is not None: 
			loss = loss_fn(pred, target)
			epoch_loss += loss.item()

		pred = softmax(pred)
		pred = pred.to("cpu").numpy() 
		true = target.to('cpu').numpy()

		pred_list.append(pred)
		true_list.append(true)

	epoch_loss /= len(dataloader)

	pred = np.concatenate(pred_list) 
	true = np.concatenate(true_list)
	return epoch_loss , pred , true


def extract_features(dataloader,model,device):
	model.eval()
	features = []
	labels = []
	with torch.no_grad():
		for (image, label) in tqdm.tqdm(dataloader):   
			img_input = image.to(device)
			labels.append(label)

			feats = model.forward_features(img_input) if hasattr(model, 'forward_features') else model.forward(img_input)
			if isinstance(feats, tuple):  # ViT 등
				feats = feats[0]
			features.append(feats.cpu())
	return torch.cat(features).numpy(), torch.cat(labels).numpy()
