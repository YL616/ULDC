import numpy as np
import torch
from modules import transform, resnet, network, contrastive_loss,my_evidence_loss,my_sdm_loss,computeLabel
from torch.nn.functional import normalize
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1";

def train_net(net, data_loader, optimizer, batch_size, zeta,epoch):
    net.train()
    for param in net.parameters():
        param.requires_grad = True

    loss = 0;
    for step, ((x_i, x_j), label) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        h_i = net.resnet(x_i)
        h_j = net.resnet(x_j)
        z_i = normalize(net.instance_projector(h_i), dim=1)
        z_j = normalize(net.instance_projector(h_j), dim=1)
        loss = contrastive_loss.C3_loss(z_i, z_j, batch_size, zeta)

        cosine_similarity_matrix = my_evidence_loss.compute_cosine_similarity(z_i,z_j,batch_size);
        # if epoch==8 and step==10:
        #     path = "./AFincos_similarity.txt";
        #     np.savetxt(path,cosine_similarity_matrix.numpy(),fmt='%.4f')

# un->select
        alpha_i2t, alpha_t2i, norm_e, sims_tanh = my_evidence_loss.my_get_alpha(cosine_similarity_matrix, 0.1);
        uncertainty1 = my_evidence_loss.MyUncertianty(alpha_i2t, batch_size)
        uncertainty2 = my_evidence_loss.MyUncertianty(alpha_t2i, batch_size)
        uncertainty = 0.5*uncertainty1+0.5*uncertainty2
        label = computeLabel.deleteHigh_Uncertainty(cosine_similarity_matrix,uncertainty,batch_size,8,20,6)
        alpha_i2t, alpha_t2i, norm_e, sims_tanh = my_evidence_loss.my_get_alpha(cosine_similarity_matrix, 0.1);
        my_mse_loss_tanh = torch.mean(my_evidence_loss.my_mseloss_Uncen_Notonlyone(label,alpha_i2t,batch_size,0.1)) + \
                           1.0*torch.mean(my_evidence_loss.my_mseloss_Uncen_Notonlyone(label,alpha_t2i,batch_size,0.1));
        alpha_i2t, alpha_t2i, norm_e = my_evidence_loss.my_get_alpha_relu(cosine_similarity_matrix);
        my_mse_loss_relu = torch.mean(my_evidence_loss.my_mseloss_Uncen_Notonlyone(label,alpha_i2t,batch_size,0.1)) + \
                           1.0*torch.mean(my_evidence_loss.my_mseloss_Uncen_Notonlyone(label,alpha_t2i,batch_size,0.1));
        loss = loss + 0.5*my_mse_loss_tanh+0.5*my_mse_loss_relu;

        sdm_loss = my_sdm_loss.compute_sdm_Uncertain(z_i,z_j,batch_size,label,1.0);


        loss = loss+sdm_loss;

        loss.backward()
        optimizer.step()
    print("loss:", loss);


    return net , optimizer
