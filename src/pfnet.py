for epoch in range(resume_epoch, opt.niter):
    if epoch < 30:
        alpha1 = 0.01
        alpha2 = 0.02
    elif epoch < 80:
        alpha1 = 0.05
        alpha2 = 0.1
    else:
        alpha1 = 0.1
        alpha2 = 0.2

    for i, data in enumerate(dataloader, 0):
        real_point, target = data

        batch_size = real_point.size()[0]

        input_cropped1, real_center = crop_points(real_point.to(device), opt.crop_point_num)

        real_point = real_point.to(device)
        real_center = real_center.to(device)
        input_cropped1 = input_cropped1.to(device)
        ############################
        # (1) data prepare
        ###########################
        real_center = Variable(real_center, requires_grad=True)
        real_center_key1_idx = utils.farthest_point_sample(real_center, 64, RAN=False)
        real_center_key1 = utils.index_points(real_center, real_center_key1_idx)
        real_center_key1 = Variable(real_center_key1, requires_grad=True)

        real_center_key2_idx = utils.farthest_point_sample(real_center, 128, RAN=True)
        real_center_key2 = utils.index_points(real_center, real_center_key2_idx)
        real_center_key2 = Variable(real_center_key2, requires_grad=True)

        input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1], RAN=True)
        input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)
        input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2], RAN=False)
        input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)
        input_cropped1 = Variable(input_cropped1, requires_grad=True)
        input_cropped2 = Variable(input_cropped2, requires_grad=True)
        input_cropped3 = Variable(input_cropped3, requires_grad=True)
        input_cropped2 = input_cropped2.to(device)
        input_cropped3 = input_cropped3.to(device)
        input_cropped = [input_cropped1, input_cropped2, input_cropped3]
        point_netG = point_netG.train()
        point_netG.zero_grad()

        fake_center1, fake_center2, fake = point_netG(input_cropped) # different resolution
        fake = torch.unsqueeze(fake, 1)
        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################

        CD_LOSS = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1))

        errG_l2 = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1)) \ # formula as in pdf
                  + alpha1 * criterion_PointLoss(fake_center1, real_center_key1) \
                  + alpha2 * criterion_PointLoss(fake_center2, real_center_key2)

        errG_l2.backward()
        optimizerG.step()
        print('[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
              % (epoch, opt.niter, i, len(dataloader),
                 errG_l2, CD_LOSS))
        f = open('loss_PFNet.txt', 'a')
        f.write('\n' + '[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                % (epoch, opt.niter, i, len(dataloader),
                   errG_l2, CD_LOSS))
        f.close()
    schedulerD.step()
    schedulerG.step()

    if epoch % 10 == 0:
        torch.save({'epoch': epoch + 1,
                    'state_dict': point_netG.state_dict()},
                   'Checkpoint/point_netG' + str(epoch) + '.pth')



