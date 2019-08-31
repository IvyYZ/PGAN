from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
import torch.nn as nn 
from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
from .utils import Bar
from torch.nn import functional as F
import pdb

    
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        import pdb
        #pdb.set_trace()
        n = inputs.size(0)

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        # loss = self.ranking_loss(dist_ap, dist_an, y)
        loss = self.ranking_loss(dist_an, dist_ap, y)  # error of 11-3

        return loss


class BaseTrainer(object):
    def __init__(self, model, criterion, X, Y, SMLoss_mode=0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader))
        accumulation_steps=8
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            #pdb.set_trace()

            inputs, targets = self._parse_data(inputs)
            #loss0, loss1, loss2, loss3, loss4, loss5, prec1,f_loss ,L_loss= self._forward(inputs, targets)
#            loss0, loss1, loss2, loss3, loss4, loss5, prec, t_loss0,t_loss1,t_loss2,t_loss3,t_loss4,t_loss5= self._forward(inputs, targets)
            loss0, loss1, loss2, loss3, loss4, loss5, prec,f_loss1,f_loss2,f_loss3,f_loss4,f_loss5,f_loss6,f_loss7,f_loss8,f_loss9,f_loss10,f_loss11,f_loss12,f_loss13,f_loss14,f_loss15,t_loss0,t_loss1,t_loss2,t_loss3,t_loss4,t_loss5= self._forward(inputs, targets)
#===================================================================================
            loss_1 = (loss0+loss1+loss2+loss3+loss4+loss5)/6
            loss_2 =(t_loss0+t_loss1+t_loss2+t_loss3+t_loss4+t_loss5)/6
            #loss_2=(f_loss1+f_loss2+f_loss3+f_loss4+f_loss5+f_loss6+f_loss7+f_loss8+f_loss9+f_loss10+f_loss11+f_loss12+f_loss13+f_loss14+f_loss15)/15
            
            #xiaodui5.15----------------------------------------------------
            loss=loss_1+loss_2*0.001#+loss_2
            loss+=loss
#            torch.autograd.backward([loss0, loss1, loss2, loss3, loss4, loss5,t_loss0,t_loss1,t_loss2,\
#                                     t_loss3,t_loss4,t_loss5],[torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda(),\
#                                    torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda(),\
#                                    torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda(),\
#                                    torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda()]) 

            torch.autograd.backward([ loss0, loss1, loss2, loss3, loss4, loss5,f_loss1,f_loss2,f_loss3,f_loss4,f_loss5,\
                                     f_loss6,f_loss7,f_loss8,f_loss9,f_loss10,f_loss11,f_loss12,f_loss13,f_loss14,f_loss15,\
                                     t_loss0,t_loss1,t_loss2,t_loss3,t_loss4,t_loss5],[torch.ones(1).cuda(), torch.ones(1).cuda(),\
                                      torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),\
                                      torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),\
                                      torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),\
                                      torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),\
                                      torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),\
                                      torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda()]) 
            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec, targets.size(0))
            
#            optimizer.step()
#            optimizer.zero_grad() 
            if((i+1)%accumulation_steps)==0: #batchsize,gradient add
#                pdb.set_trace()
#                            loss=loss_1+loss_2
#                loss=loss/accumulation_steps

                
                optimizer.step()
                optimizer.zero_grad() 
            
#            loss=loss_1+loss_2
#            losses.update(loss.data[0], targets.size(0))
#            precisions.update(prec1, targets.size(0))
#
#            optimizer.zero_grad()
#            torch.autograd.backward([ loss0, loss1, loss2, loss3, loss4, loss5,loss_2],[torch.ones(1).cuda(), torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda()]) 
#            #loss_2.backward()
#            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bt:.3f} {N_bta:.3f} | Data {N_dt:.3f} {N_dta:.3f} | Loss {N_loss:.3f} {N_lossa:.3f} | Prec {N_prec:.2f} {N_preca:.2f}'.format(
                      N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),
                              N_bt=batch_time.val, N_bta=batch_time.avg,
                              N_dt=data_time.val, N_dta=data_time.avg,
                              N_loss=losses.val, N_lossa=losses.avg,
                              N_prec=precisions.val, N_preca=precisions.avg,
							  )
            bar.next()
        bar.finish()



    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        #pdb.set_trace()
#        (imgs, _, pids, _),(imgs2, _, pids2, _) = inputs
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        #pdb.set_trace()
        outputs = self.model(*inputs)
        index = (targets-751).data.nonzero().squeeze_()
		
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss0 = self.criterion(outputs[1][0],targets)
            loss1 = self.criterion(outputs[1][1],targets)
            loss2 = self.criterion(outputs[1][2],targets)
            loss3 = self.criterion(outputs[1][3],targets)
            loss4 = self.criterion(outputs[1][4],targets)
            loss5 = self.criterion(outputs[1][5],targets)
            prec, = accuracy(outputs[1][2].data, targets.data)
            prec = prec[0]
            

#            pdb.set_trace()
            ba=len(outputs[1][0])
            #target2=Variable(torch.mul(torch.ones(ba,1),-1).cuda())
            target2=Variable(torch.zeros(ba).cuda())
            # F.pairwise_distance(x1,x2,p=2)
            
            #f_loss1=self.criterion(F.cosine_similarity(outputs[2][0],outputs[2][1])-1,target2)
            #f_loss1=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][0],outputs[2][1]),target2),0.01)
#            f_loss1=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][0],outputs[2][1]),target2),0.01)
#            f_loss2=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][0],outputs[2][2]),target2),0.01)
#            f_loss3=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][0],outputs[2][3]),target2),0.01)
#            f_loss4=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][0],outputs[2][4]),target2),0.01)
#            f_loss5=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][0],outputs[2][5]),target2),0.01)
#            
#
#
#            f_loss6=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][1],outputs[2][2]),target2),0.01)
#            f_loss7=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][1],outputs[2][3]),target2),0.01)
#            f_loss8=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][1],outputs[2][4]),target2),0.01)
#            f_loss9=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][1],outputs[2][5]),target2),0.01)
#            
#
#
#            f_loss10=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][2],outputs[2][3]),target2),0.01)
#            f_loss11=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][2],outputs[2][4]),target2),0.01)
#            f_loss12=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][2],outputs[2][5]),target2),0.01)
#            
#
#            f_loss13=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][3],outputs[2][4]),target2),0.01)
#            f_loss14=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][3],outputs[2][5]),target2),0.01)
#
#            f_loss15=torch.mul(F.l1_loss(F.pairwise_distance(outputs[2][4],outputs[2][5]),target2),0.01)

            parm1=0.1
            f_loss1=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][0],outputs[2][1]),target2),parm1)
            f_loss2=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][0],outputs[2][2]),target2),parm1)           
            f_loss3=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][0],outputs[2][3]),target2),parm1)
            f_loss4=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][0],outputs[2][4]),target2),parm1)
            f_loss5=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][0],outputs[2][5]),target2),parm1)
            


            f_loss6=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][1],outputs[2][2]),target2),parm1)
            f_loss7=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][1],outputs[2][3]),target2),parm1)
            f_loss8=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][1],outputs[2][4]),target2),parm1)
            f_loss9=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][1],outputs[2][5]),target2),parm1)
            


            f_loss10=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][2],outputs[2][3]),target2),parm1)
            f_loss11=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][2],outputs[2][4]),target2),parm1)
            f_loss12=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][2],outputs[2][5]),target2),parm1)
            

            f_loss13=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][3],outputs[2][4]),target2),parm1)
            f_loss14=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][3],outputs[2][5]),target2),parm1)

            f_loss15=torch.mul(F.l1_loss(F.cosine_similarity(outputs[2][4],outputs[2][5]),target2),parm1)           

            
            tripletLoss=TripletLoss(0.3)
            t_loss0=tripletLoss(outputs[2][0],targets)
            t_loss1=tripletLoss(outputs[2][1],targets)
            t_loss2=tripletLoss(outputs[2][2],targets)
            t_loss3=tripletLoss(outputs[2][3],targets)
            t_loss4=tripletLoss(outputs[2][4],targets)
            t_loss5=tripletLoss(outputs[2][5],targets)
            
            
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss0, loss1, loss2, loss3, loss4, loss5, prec,f_loss1,f_loss2,f_loss3,f_loss4,\
            f_loss5,f_loss6,f_loss7,f_loss8,f_loss9,f_loss10,f_loss11,f_loss12,f_loss13,f_loss14,\
            f_loss15,t_loss0,t_loss1,t_loss2,t_loss3,t_loss4,t_loss5

#        return loss0, loss1, loss2, loss3, loss4, loss5, prec,t_loss0,t_loss1,t_loss2,t_loss3,t_loss4,t_loss5