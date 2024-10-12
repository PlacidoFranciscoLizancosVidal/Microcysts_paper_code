# -*- coding: utf-8 -*-
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, f1_score, precision_recall_fscore_support
from numpy import sqrt 


#############################################################################
class EarlyStopping():
	def __init__(self, mode='min', min_delta=0, patience=10):
		self.mode = mode
		self.min_delta = min_delta
		self.patience = patience
		self.best = None
		self.num_bad_epochs = 0
		self.is_better = None
		self._init_is_better(mode, min_delta)

		if patience == 0:
			self.is_better = lambda a, b: True
			self.step = lambda a: False

	def step(self, metrics):
		if self.best is None:
			self.best = metrics
			return False

		if np.isnan(metrics):
			return True

		if self.is_better(metrics, self.best):
			self.num_bad_epochs = 0
			self.best = metrics
		else:
			self.num_bad_epochs += 1

		if self.num_bad_epochs >= self.patience:
			return True

		return False

	def _init_is_better(self, mode, min_delta):
		if mode not in {'min', 'max'}:
			raise ValueError('mode ' + mode + ' is unknown!')
		if mode == 'min':
			self.is_better = lambda a, best: a < best - min_delta
		if mode == 'max':
			self.is_better = lambda a, best: a > best + min_delta
            
#############################################################################         

def epochProcess(model, loader, criterion, optimizer,train_phase=True):
    # Establecer en el modo correcto el modelo
    model.train() if train_phase else model.eval()
    
    # Variables para guardar las métricas
    running_loss = 0
    running_targets = []
    running_outputs = []
       
    with torch.set_grad_enabled(train_phase):
        # Recorremos el dataset
        for inp, cpu_target in loader:
            inp = inp.cuda(non_blocking=True)
            target = cpu_target.cuda(non_blocking=True)

            # Gradients to zero
            if train_phase:
                optimizer.zero_grad()

            # Output and backpropagation
            output = torch.sigmoid(model(inp))
            
            if criterion is not None:
                loss = criterion(output, target)

            if train_phase:
                # Loss backpropagation
                loss.backward()
                # Optimizer step
                optimizer.step()

            # Predicciones
            pred = output.cpu().detach().numpy()
            
            # Loss
            running_loss += loss.item()
            running_outputs += list(pred)
            running_targets += list(cpu_target.numpy())
                
    # Mean loss, target, preds
    running_outputs = np.vstack(running_outputs)

    return (running_loss / len(loader), running_outputs, running_targets)
#############################################################################

#conf = {
#    'experiment_name': 'FOLD_DATA_FN_L_6V2T1',
#    'model_name': 'densenet161',
#    'batch_size': 250,
#    'num_workers': 16,
#    'stopping_patience': 25,
#    'scheduler_patience': 10,
#    'scheduler_factor': 0.66,
#    'optimizer_lr': 0.01,
#    'pretrained': False,
#}

def extractMetrics(output, target):
    binary_output = np.argmax(output, axis=1)
    binary_target = [x[1] for x in target]
    
    metrics = {}
    metrics['auc'] = float(roc_auc_score(target, output))
    metrics['output'] = [[float(y) for y in x] for x in output]
    metrics['target'] = [[float(y) for y in x] for x in target]
    
    metrics['accuracy'] = float(accuracy_score(binary_target, binary_output))
    metrics['mcc'] = float(matthews_corrcoef(binary_target, binary_output))

    precision, recall, f1_score, _ = precision_recall_fscore_support(binary_target, binary_output,average='binary')

    metrics['precision'] = float(precision)
    metrics['recall'] = float(recall)
    metrics['f1_score'] = float(f1_score)

    return metrics
    
