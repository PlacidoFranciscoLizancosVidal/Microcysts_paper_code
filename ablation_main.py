from util_training import * 
from util_storage import * 
from dataset_library_alt import * 

from pathlib import Path
_p = Path

import torch, sys
torch.manual_seed(512)

import numpy as np
from util_models import *
from copy import deepcopy

from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from util_models import obtainModel

from util_storage import *
##################################
fold_info = 'FOLD_DATA_FN_L_6V2T1'
p_images = './dataset/ORIGINALS/'
p_labels = './dataset/MICROCYSTS/'
p_roi = './dataset/SUBMASKS_64/'
##################################


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
c = {
    'experiment_name' : 'Minicist_V5_FASE_2_DENSENET_64x64_ALLREPSFULL',
    'model_name' : 'densenet161',
    'batch_size' : 64,
    'num_workers' : 8,
    'stopping_patience' : 31,
    'scheduler_patience' : 10,
    'scheduler_factor' : 0.75,
    'optimizer_lr' : 0.005,
    'pretrained' : True 
}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#window_ranges = list(range(14,74,10))
#window_ranges = list(range(10,64,10))
window_ranges = [30]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def _(name):
    o = ['png','bmp','jpg','jpeg','tif','tiff']
    t = ['.png','.bmp','.jpg','.jpeg','.tif','.tiff']
    
    for i in range(len(o)):
        name = name.replace(o[i], t[i])
        
    return name
    

fold_data = loadObj(fold_info)
saveObj(c, c['experiment_name'] + '_conf')

for fold_index in range(9,len(fold_data)):
    print('FOLD: {}'.format(fold_index))
    fold_image_paths = fold_data[fold_index]
    for window_size in window_ranges:
        target_path = _p(c['experiment_name'], str(window_size), str(fold_index))
        os.makedirs(target_path, exist_ok = True)

        print('Window size: ', window_size)
        ref = [x.replace('.npy','').replace('dataset/','') for x in fold_image_paths['train']]
        print('Loading train dataset...')        
        train_dataset = MicrocystDataset(
            [p_images + _(x) for x in ref],
            [p_roi + x + '.png' for x in ref],
            [p_labels + x + '.png' for x in ref],
            window_size = window_size
        )

        ref = [x.replace('.npy','').replace('dataset/','')  for x in fold_image_paths['val']]
        print('Loading val dataset...')
        val_dataset = MicrocystDataset(
            [p_images + _(x) for x in ref],
            [p_roi + x + '.png' for x in ref],
            [p_labels + x + '.png' for x in ref],
            window_size = window_size
        )

        ref = [x.replace('.npy','').replace('dataset/','')  for x in fold_image_paths['test']]
        print('Loading test dataset...')
        test_dataset = MicrocystDataset(
            [p_images + _(x) for x in ref],
            [p_roi + x + '.png' for x in ref],
            [p_labels + x + '.png' for x in ref],
            window_size = window_size
        )
        
        
        ## EXPERIMENTATION
        torch.backends.cudnn.benchmark = True
        train_loader, validation_loader, test_loader =  map(lambda x: DataLoader(x, batch_size=c['batch_size'], shuffle=True, num_workers=c['num_workers'], pin_memory = True), [train_dataset, val_dataset, test_dataset])

        fold_model = obtainModel(c['model_name'], 2, pretrained=c['pretrained'])
        fold_model = fold_model.cuda()

        # For weighting purposes
        train_positives = train_dataset.getPositiveLen()
        train_negatives = train_dataset.getNegativeLen()
        val_positives = val_dataset.getPositiveLen()
        val_negatives = val_dataset.getNegativeLen()
        test_positives = test_dataset.getPositiveLen()
        test_negatives = test_dataset.getNegativeLen()
        tw = torch.Tensor([train_positives/train_negatives, 1])
        vw = torch.Tensor([val_positives/val_negatives, 1])
        tsw = torch.Tensor([test_positives/test_negatives, 1])

        print('Train weight: ',tw[0])
        print('Validation weight: ', vw[0])
        print('Test weight: ', tsw[0])
        
        print('TP: {} TN: {}'.format(train_positives,train_negatives))
        print('VP: {} VN: {}'.format(val_positives,val_negatives))
        print('TP: {} TN: {}'.format(test_positives,test_negatives))

        #criterion_train = criterion_val = criterion_test = BCEWithLogitsLoss()#pos_weight = tw.cuda())
        criterion_train = BCEWithLogitsLoss(pos_weight = tw.cuda())
        criterion_val = BCEWithLogitsLoss(pos_weight = vw.cuda())
        criterion_test = BCEWithLogitsLoss(pos_weight = tsw.cuda())

        # Criterion, scheduler y optimizer
        optimizer = torch.optim.AdamW(fold_model.parameters(), lr=c['optimizer_lr'], amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=c['scheduler_patience'], factor=c['scheduler_factor'], verbose=True)
        stopper = EarlyStopping(patience=c['stopping_patience'])

        # Comienzo del entrenamiento
        best_model = None
        best_val = None
        best_epoch = None

        res = []
        # Train
        current_epoch_index = 0
        entrenar = True
        while entrenar:
            current_epoch_index += 1
            print('\nEpoch {}'.format(current_epoch_index))
            print('-' * 10)
            sys.stdout.flush()

            #Train epoch
            train_loss, train_output, train_target = epochProcess(fold_model, train_loader, criterion_train, optimizer, train_phase = True)
            
            tm = extractMetrics(train_output, train_target)
            tm['loss'] = train_loss
            print('[Train] Loss: {}, AUC: {} Prec: {} Rec: {} Acc: {} F1 Sc: {}, MCC: {}'.format(train_loss,
                                                                                                  tm['auc'],
                                                                                                  tm['precision'],
                                                                                                  tm['recall'],
                                                                                                  tm['accuracy'],
                                                                                                  tm['f1_score'],
                                                                                                  tm['mcc']))
            # Validate epoch
            val_loss, val_output, val_target = epochProcess(fold_model, validation_loader, criterion_val, optimizer, train_phase = False)
            vm = extractMetrics(val_output, val_target)
            vm['loss'] = val_loss
            print('[Val.] Loss: {}, AUC: {} Prec: {} Rec: {} Acc: {} F1 Sc: {}, MCC: {}'.format(val_loss,
                                                                                                  vm['auc'],
                                                                                                  vm['precision'],
                                                                                                  vm['recall'],
                                                                                                  vm['accuracy'],
                                                                                                  vm['f1_score'],
                                                                                                  vm['mcc']))
            # Test epoch
            test_loss , test_output, test_target = epochProcess(fold_model, test_loader, criterion_test, optimizer, train_phase = False)
            tsm = extractMetrics(test_output, test_target)
            tsm['loss'] = test_loss
            print('[Test] Loss: {}, AUC: {} Prec: {} Rec: {} Acc: {} F1 Sc: {}, MCC: {}\n'.format(test_loss,
                                                                                                  tm['auc'],
                                                                                                  tm['precision'],
                                                                                                  tm['recall'],
                                                                                                  tm['accuracy'],
                                                                                                  tm['f1_score'],
                                                                                                  tm['mcc']))

            res.append({'train':tm, 'val': vm, 'test': tsm})
            
            # Guardamos el modelo si es el mejor
            if best_val is None:
                print('New best weights with val loss of {} stored. Previous: {} ({})'.format(val_loss, best_val, best_epoch))
                best_model = deepcopy(fold_model.state_dict())
                best_epoch = current_epoch_index
                best_val = val_loss
            elif best_val > val_loss:
                print('New best weights with val loss of {} stored. Previous: {} ({})'.format(val_loss, best_val, best_epoch))
                best_model = deepcopy(fold_model.state_dict())
                best_epoch = current_epoch_index
                best_val = val_loss
            
            # Scheduler
            scheduler.step(val_loss)
            for g in optimizer.param_groups:
                print('LR: {}'.format(g['lr']))

            # Stopper
            if (stopper.step(val_loss)):
                print('Early Stopping patience reached at {} epochs. Training ending.'.format(stopper.num_bad_epochs))
                sys.stdout.flush()
                entrenar = False
            else:
                print('Early Stopping bad epochs: ' + str(stopper.num_bad_epochs))
                sys.stdout.flush()

        # End of epoch training
        torch.save(best_model, _p(target_path, 'model.pt'))
        saveObj(res, _p(target_path, 'fold_results'))
        saveObj({'tp': int(train_positives), 'tn': int(train_negatives), 'vp': int(val_positives),'vn': int(val_negatives),'tsp': int(test_positives),'tsn': int(test_negatives)}, _p(target_path, 'num_images'))
