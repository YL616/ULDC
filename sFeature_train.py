import numpy as np
import torch
from modules import transform, resnet, network, contrastive_loss,my_evidence_loss,my_sdm_loss,computeLabel
from torch.nn.functional import normalize
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1";

def train_net(net, data_loader, optimizer, batch_size, zeta):
#训练模式下，神经网络会保留所有的中间变量，以便在反向传播时计算梯度。相反，在测试模式下，神经网络会舍弃这些中间变量，以便减少内存占用。因此，在训练神经网络时，我们需要将其设置为训练模式，以便在反向传播时计算梯度
    net.train()  #设置神经网络为训练模式，以便在反向传播时计算梯度
    for param in net.parameters():
        param.requires_grad = True

    # feature = torch.zeros(19500,512)

    loss = 0;
# 为什么x_i和x_j能够表示图像增强后的两个版本？step是步骤没问题，但((x_i, x_j), _)是什么？
#     for step, ((x_i, x_j), _) in enumerate(data_loader):   #怎么确定data_loader中数据的表现形式？——这里应该是同一数据X增强后的两个分量X_i和X_j——看main中dataset下面一行的定义会明白！
    # 19500/128 = 152.34
    for step, ((x_i, x_j), label) in enumerate(data_loader):   #怎么确定data_loader中数据的表现形式？——这里应该是同一数据X增强后的两个分量X_i和X_j——看main中dataset下面一行的定义会明白！
        # 下面的x_i和x_j的大小应该是batch_size*属性：比如是128*(28*28)??? 因为enumerate一次取出batch_size个元素
        optimizer.zero_grad()
        # print("label:",list(label));
        # print("label.shape:",label.shape);

        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        h_i = net.resnet(x_i)   #h_i torch.Size([128, 512])
        h_j = net.resnet(x_j)

        # print("h_i",h_i.shape)
        # print("step:", step)
    # 得到特征的语句
        # if step!=152:
        #     feature[step*128:(step+1)*128,] = h_i
        # else:
        #     feature[step*128:,] = h_i

        # z_ 是投影到低维空间表示的结果，且进行了归一化
        z_i = normalize(net.instance_projector(h_i), dim=1)  #normalize:对括号内的参数值进行归一化处理,为了确保特征向量的长度为1，以便在计算对比损失时进行比较。
        z_j = normalize(net.instance_projector(h_j), dim=1)  #instance_projector是一个神经网络模块，它将输入的特征向量投影到一个低维空间中。
       # 我认为这里应该是矩阵的归一化，行数仍为batch_size，但列数经过投影应该减少。
       #  if step==0:
       #      print("x_i:", x_i.shape);  #torch.Size([128, 3, 224, 224])
       #      print("h_i:", h_i.shape);  #torch.Size([128, 512])
       #      print("z_i:",z_i.shape);   #torch.Size([128, 128])

#--------------------------------原代码C3的loss----------------------------------------#
        loss = contrastive_loss.C3_loss(z_i, z_j, batch_size, zeta)
        # print("z_i:",z_i);
        # print("z_j:",z_j);
        # loss = 0;


##########-------------根据不确定性进行筛选的整体模块--------------###############
        alpha_i2t, alpha_t2i, norm_e, sims_tanh = my_evidence_loss.my_get_alpha(cosine_similarity_matrix, 0.1);
        uncertainty1 = my_evidence_loss.MyUncertianty(alpha_i2t, batch_size)
        uncertainty2 = my_evidence_loss.MyUncertianty(alpha_t2i, batch_size)
        uncertainty = 0.5 * uncertainty1 + 0.5 * uncertainty2
        # print("cos:",cosine_similarity_matrix.shape)
        # print("un:",uncertainty.shape)
        # print("alpha:",alpha_t2i.shape)
        # print(list(uncertainty))
        label = computeLabel.deleteHigh_Uncertainty(cosine_similarity_matrix, uncertainty, batch_size, 12, 10,
                                                    10)  # 给的倒数第二个k值，是不确定性在前几大不能要。最后一个是筛选后相似度的临界值
        # print(label.shape)

        # 用筛选的方式，直接能得到的标签就是NotOnlyOne的方阵，不是一个数值在0-10的一维向量，所以直接把标签传进去就可以了。
        alpha_i2t, alpha_t2i, norm_e, sims_tanh = my_evidence_loss.my_get_alpha(cosine_similarity_matrix, 0.1);
        my_mse_loss_tanh = torch.mean(my_evidence_loss.my_mseloss_Uncen_Notonlyone(label, alpha_i2t, batch_size, 0.1)) + \
                           1.0 * torch.mean(
            my_evidence_loss.my_mseloss_Uncen_Notonlyone(label, alpha_t2i, batch_size, 0.1));
        alpha_i2t, alpha_t2i, norm_e = my_evidence_loss.my_get_alpha_relu(cosine_similarity_matrix);
        my_mse_loss_relu = torch.mean(my_evidence_loss.my_mseloss_Uncen_Notonlyone(label, alpha_i2t, batch_size, 0.1)) + \
                           1.0 * torch.mean(
            my_evidence_loss.my_mseloss_Uncen_Notonlyone(label, alpha_t2i, batch_size, 0.1));
        # my_mse_loss = 0.5*my_mse_loss;
        loss = loss + 0.5 * my_mse_loss_tanh + 0.5 * my_mse_loss_relu;
        sdm_loss = my_sdm_loss.compute_sdm_Uncertain(z_i, z_j, batch_size, label, 1.0);




        # --------------------------------自己新增的狄利克雷的loss----------------------------------------#
        cosine_similarity_matrix = my_evidence_loss.compute_cosine_similarity(z_i,z_j,batch_size);
# un->select
        # alpha_i2t, alpha_t2i, norm_e, sims_tanh = my_evidence_loss.my_get_alpha(cosine_similarity_matrix, 0.1);
        # # alpha_i2t, alpha_t2i, norm_e = my_evidence_loss.my_get_alpha_relu(cosine_similarity_matrix);
        # my_mse_loss = torch.mean(my_evidence_loss.my_mse_loss_Notonlyone_distribute(label,alpha_i2t,batch_size,0.1));
        # my_mse_loss += torch.mean(my_evidence_loss.my_mse_loss_Notonlyone_distribute(label,alpha_t2i,batch_size,0.1));
        # # my_mse_loss = 0.5*my_mse_loss;
        # loss = loss+my_mse_loss;

  # 截止0624CIFAR-10数据集最优结果——用真实标签作为监督数据，得到最优结果的代码
  #       alpha_i2t, alpha_t2i, norm_e, sims_tanh = my_evidence_loss.my_get_alpha(cosine_similarity_matrix, 0.1);
  #       my_mse_loss_tanh = torch.mean(my_evidence_loss.my_mse_loss_Notonlyone(label,alpha_i2t,batch_size,0.1)) + \
  #                          1.0*torch.mean(my_evidence_loss.my_mse_loss_Notonlyone(label,alpha_t2i,batch_size,0.1));
  #       alpha_i2t, alpha_t2i, norm_e = my_evidence_loss.my_get_alpha_relu(cosine_similarity_matrix);
  #       my_mse_loss_relu = torch.mean(my_evidence_loss.my_mse_loss_Notonlyone(label,alpha_i2t,batch_size,0.1)) + \
  #                          1.0*torch.mean(my_evidence_loss.my_mse_loss_Notonlyone(label,alpha_t2i,batch_size,0.1));
  #       # my_mse_loss = 0.5*my_mse_loss;
  #       loss = loss + 0.5*my_mse_loss_tanh+0.5*my_mse_loss_relu;

 # 0621下午服务器卡2
 #        alpha_i2t, alpha_t2i, norm_e = my_evidence_loss.my_get_alpha_relu(cosine_similarity_matrix);
 #        my_mse_loss = torch.mean(my_evidence_loss.my_mse_loss_Notonlyone(label,alpha_i2t,batch_size,0.1));
 #        my_mse_loss += torch.mean(my_evidence_loss.my_mse_loss_Notonlyone(label,alpha_t2i,batch_size,0.1));
 #        # my_mse_loss = 0.5*my_mse_loss;
 #        loss = loss+my_mse_loss;

 # 0626发现只有自己损失函数用了真实标签后的弥补措施:使用相似度矩阵大于0.3计算出label  849
 #        alpha_i2t, alpha_t2i, norm_e, sims_tanh = my_evidence_loss.my_get_alpha(cosine_similarity_matrix, 0.1);
 #        my_mse_loss_tanh = torch.mean(my_evidence_loss.my_mse_loss_pseudo_Notonlyone(cosine_similarity_matrix, alpha_i2t, batch_size, 0.1)) + \
 #                           1.0 * torch.mean(my_evidence_loss.my_mse_loss_pseudo_Notonlyone(cosine_similarity_matrix, alpha_t2i, batch_size, 0.1));
 #        alpha_i2t, alpha_t2i, norm_e = my_evidence_loss.my_get_alpha_relu(cosine_similarity_matrix);
 #        my_mse_loss_relu = torch.mean(my_evidence_loss.my_mse_loss_pseudo_Notonlyone(cosine_similarity_matrix, alpha_i2t, batch_size, 0.1)) + \
 #                           1.0 * torch.mean(my_evidence_loss.my_mse_loss_pseudo_Notonlyone(cosine_similarity_matrix, alpha_t2i, batch_size, 0.1));
 #        # my_mse_loss = 0.5*my_mse_loss;
 #        loss = loss + 0.5 * my_mse_loss_tanh + 0.5 * my_mse_loss_relu;


 #  # 0627使用相似度矩阵前k(12)大的值计算出label：#CIAFR100数据集是6  ImageNet_Dogs数据集是8   ImageNet-10数据集是12
        alpha_i2t, alpha_t2i, norm_e, sims_tanh = my_evidence_loss.my_get_alpha(cosine_similarity_matrix, 0.1);
        my_mse_loss_tanh = torch.mean(my_evidence_loss.my_mse_loss_pseudo_fristKNotonlyone(cosine_similarity_matrix, alpha_i2t, batch_size, 0.1)) + \
                           1.0 * torch.mean(my_evidence_loss.my_mse_loss_pseudo_fristKNotonlyone(cosine_similarity_matrix, alpha_t2i, batch_size, 0.1));
        alpha_i2t, alpha_t2i, norm_e = my_evidence_loss.my_get_alpha_relu(cosine_similarity_matrix);
        my_mse_loss_relu = torch.mean(my_evidence_loss.my_mse_loss_pseudo_fristKNotonlyone(cosine_similarity_matrix, alpha_i2t, batch_size, 0.1)) + \
                           1.0 * torch.mean(my_evidence_loss.my_mse_loss_pseudo_fristKNotonlyone(cosine_similarity_matrix, alpha_t2i, batch_size, 0.1));
        # loss = loss + 0.5 * my_mse_loss_tanh + 0.5 * my_mse_loss_relu;
        loss = 0.5 * my_mse_loss_tanh + 0.5 * my_mse_loss_relu;   # 不要C3的loss



    #lss提供的原调用代码
        # loss = torch.mean(mse_loss_tanh(labels_distribute, alpha_t2i_t, batch_size, 0.1)) + 1.0 * torch.mean(
        #     mse_loss_tanh(labels_distribute, alpha_i2t_t, batch_size, 0.1))
        # loss1 = torch.mean(mse_loss(labels, alpha_t2i, batch_size, 0.1)) + 1.0 * torch.mean(
        #     mse_loss(labels, alpha_i2t, batch_size, 0.1))
        # loss = 0.5 * loss + 0.5 * loss1

# --------------------------------自己新增的sdm的loss----------------------------------------#
        # 原函数参数：image_fetures, text_fetures,batch_size, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8
        # 参数：z_i,z_j,batch_size,label,1.0    # self.logit_scale = torch.ones([]) * (1 / args.temperature)  1.0

        # sdm_loss = my_sdm_loss.compute_sdm(z_i,z_j,batch_size,label,1.0);

        # sdm_loss = my_sdm_loss.compute_pseudo_sdm(z_i,z_j,batch_size,cosine_similarity_matrix,1.0);

        sdm_loss = my_sdm_loss.compute_pseudo_fristk_sdm(z_i,z_j,batch_size,cosine_similarity_matrix,1.0);
        loss = loss+sdm_loss;

        # if step==0:
        #     path = "./cos_similarity_num0.txt";
        #     with open(path, 'a') as f:
        #         f.write(str(cosine_similarity_matrix)+'\n');

        loss.backward()
        optimizer.step()
    print("loss:", loss);


    return net , optimizer   #, feature
