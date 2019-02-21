import torch
import numpy as np
from tensorboardX import SummaryWriter
import os 
import shutil 
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data.numpy()
    #return (pred > 0).sum()*1.0/dista.size()[0]
    return (pred>0).sum()/dista.size()[0]


def extract_embeddings(dataloader, model, nfeat):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), nfeat))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            #if cuda:
            #    images = images.cuda()
            images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def embedding_acc(dataloader, model, nfeat):
    accs = AverageMeter()
    with torch.no_grad():
        model.eval()
        #embeddings = np.zeros((len(dataloader.dataset), nfeat))
        #labels = np.zeros(len(dataloader.dataset))
        #k = 0
        for images, target in dataloader:
            
            # anchor, positive, negative
            images[0] = images[0].cuda()
            images[1] = images[1].cuda()
            images[2] = images[2].cuda()
            embedded_x = model.get_embedding(images[0])
            embedded_y = model.get_embedding(images[1])
            embedded_z = model.get_embedding(images[2])
            dist_p = F.pairwise_distance(embedded_x, embedded_y, 2)
            dist_n = F.pairwise_distance(embedded_x, embedded_z, 2)
            
            acc = accuracy(dist_n, dist_p)
            accs.update(acc, images[0].size(0))
            #embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            #labels[k:k+len(images)] = target.numpy()
            #k += len(images)
    print('Test set: Average Embedding Accuracy: {:.2f}%\n'.format(100. * accs.avg))            
    return accs.avg

def save_checkpoint(state, is_best, model_name, filename='checkpoint.pth.tar', iscls=True):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        if iscls:
            shutil.copyfile(filename, 'runs/%s/'%(model_name) + 'model_cls_best.pth.tar')
        else:
            shutil.copyfile(filename, 'runs/%s/'%(model_name) + 'model_emb_best.pth.tar')

   
def fit(clsonly, writer, args, train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, val_loader_cls=None, val_loader_emb=None, use_mixup=False, adaptive_margin=False, normsoft=False):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    best_acc = 0.0
    best_acc_emb = 0.0
    
    if args.triplet > 0:
        if use_mixup:
            model_name = "mixup/{}/nfeat_{}_triplet_{}_margin_{}_alpha_{}_ntry{}".format(args.dataset, args.nfeat, args.triplet, args.margin, args.alpha, args.ntry)
            if adaptive_margin:
                model_name = "mixup/{}/nfeat_{}_triplet_{}_adaptive_margin_{}_alpha_{}_ntry{}".format(args.dataset, args.nfeat, args.triplet, args.margin, args.alpha, args.ntry)
        else:
            model_name = "{}/{}/nfeat_{}_triplet_{}_margin_{}_ntry{}".format(args.dataset, args.type, args.nfeat, args.triplet, args.margin, args.ntry)
    else:
        if normsoft == True:
            model_name = "{}/{}/normsoft_nfeat_{}_triplet_{}_lr_{}_ntry{}".format(args.dataset, args.type, args.nfeat, args.triplet, args.lr, args.ntry)
        else:
            model_name = "{}/{}/nfeat_{}_triplet_{}_lr_{}_ntry{}".format(args.dataset, args.type, args.nfeat, args.triplet, args.lr, args.ntry)
    
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            writer.add_scalar('data/test_acc', metric.value(), epoch)
            prec1 = metric.value()

        print(message)
        writer.add_scalar('data/train_loss', train_loss, epoch)
        writer.add_scalar('data/test_loss', val_loss, epoch)
        
        if args.save_embedding == True:
            if epoch == n_epochs-1:
                if val_loader_cls is not None:
                    val_embeddings_baseline, val_labels_baseline = extract_embeddings(val_loader_cls, model, args.nfeat)
                else:
                    val_embeddings_baseline, val_labels_baseline = extract_embeddings(val_loader, model, args.nfeat)
                writer.add_embedding(val_embeddings_baseline, metadata=val_labels_baseline, global_step=epoch)

            if val_loader_emb is not None:
                # compute embedding performance 
                em_acc = embedding_acc(val_loader_emb, model, args.nfeat)
                writer.add_scalar('data/test_emacc', em_acc, epoch)
                is_best_emb = em_acc > best_acc_emb
                best_acc_emb = max(em_acc, best_acc_emb)
                writer.add_scalar('data/best_emb_acc', best_acc_emb, epoch)


        if clsonly:
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_acc
            best_acc = max(prec1, best_acc)

            writer.add_scalar('data/best_test_acc', best_acc, epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model_name,
                'state_dict': model.state_dict(),
                'best_prec1': best_acc,
                'optimizer' : optimizer.state_dict(),
                'manual_seed' : args.seed
            }, is_best, model_name, filename='model_cls.pth.tar', iscls=True)
        else:

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model_name,
                'state_dict': model.state_dict(),
                'best_emb_acc': best_acc_emb,
                'optimizer' : optimizer.state_dict(),
                'manual_seed' : args.seed
            }, is_best_emb, model_name, filename='model_triplet.pth.tar', iscls=False)            
    
    
        


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
