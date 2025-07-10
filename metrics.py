import numpy as np
import torch
from skimage import color
import os
import matplotlib.pyplot as plt
import logging

class Evaluator():
    def __init__(self, criterion, data_loader, split_name, log_dirpath):
        """ Initialize the evaluator
        :param criterion: pytorch loss function
        :param data_loader: pytorch data loader
        :param split_name: string
        :param log_dirpath:
        """
        super().__init__()
        self.criterion = criterion
        self.data_loader = data_loader
        self.split_name = split_name
        self.log_dirpath = log_dirpath

        self.MSE = torch.nn.MSELoss(reduction='sum')
        self.E00 = deltaE00()
        self.E76 = deltaE76()

    def evaluate(self, net, split, epoch=0):
        """ Evaluate the network on the given split
        :param net: pytorch network
        :param split: string
        :param epoch: int
        :return: dict
        """
        mse_list, e76_list, e00_list, mae_list = [], [], [], []

        out_dirpath = os.path.join(self.log_dirpath, self.split_name)
        if not os.path.isdir(out_dirpath):
            os.mkdir(out_dirpath)

        net.eval()

        with torch.no_grad():
            for batch_num, data_dict in enumerate(self.data_loader):
                target_images = data_dict['target']
                output_images = net(data_dict)

                for target_img, output_img, name, cc_area in zip(target_images, output_images, data_dict['name'], data_dict['color_chart_area']):
                    mse_list.append(self.MSE(target_img.to(output_img.device), output_img).item() / (target_img.shape[0] * target_img.shape[1] * target_img.shape[2] - (cc_area.item()*3)))
                    e76_list.append(self.E76.compute(((target_img.cpu().detach().permute(1,2,0).numpy()) * 255).astype(np.uint8),
                                            ((output_img.cpu().detach().permute(1,2,0).numpy()) * 255).astype(np.uint8)) / (target_img.shape[1] * target_img.shape[2] - cc_area.item()))
                    e00_list.append(self.E00.compute(((target_img.cpu().detach().permute(1,2,0).numpy()) * 255).astype(np.uint8),
                                            ((output_img.cpu().detach().permute(1,2,0).numpy()) * 255).astype(np.uint8), color_chart_area=cc_area.item()))
                    mae_list.append(calc_mae(((target_img.cpu().detach().permute(1,2,0).numpy()) * 255).astype(np.uint8),
                                            ((output_img.cpu().detach().permute(1,2,0).numpy()) * 255).astype(np.uint8)) / (target_img.shape[1] * target_img.shape[2] - cc_area.item()))
                    if batch_num < 2000:
                        e00_plt = e00_list[-1]
                        target_img = (torch.clamp(target_img.permute(1, 2, 0), 0, 1) * 255).cpu().detach().numpy().astype('uint8')
                        output_img = (torch.clamp(output_img.permute(1, 2, 0), 0, 1) * 255).cpu().detach().numpy().astype('uint8')
                        plt.imsave(os.path.join(out_dirpath, f"{name}_{epoch}_{e00_plt:.2f}.png"), output_img)

            metrics = {f'{split}/mse_mean': np.mean(mse_list), f'{split}/mse_median': np.median(mse_list),
                       f'{split}/mse_trimean': trimean(mse_list), f'{split}/e76_mean': np.mean(e76_list),
                       f'{split}/e76_median': np.median(e76_list), f'{split}/e76_trimean': trimean(e76_list),
                       f'{split}/e00_mean': np.mean(e00_list), f'{split}/e00_median': np.median(e00_list),
                       f'{split}/e00_trimean': trimean(e00_list), f'{split}/mae_mean': np.mean(mae_list),
                       f'{split}/mae_median': np.median(mae_list), f'{split}/mae_trimean': trimean(mae_list)}


            logging.info(f"###### {split} ######")
            logging.info("###### mean ### median ## trimean ##")
            logging.info(f"E00: {np.mean(e00_list):.4f} {np.median(e00_list):.4f} {trimean(e00_list):.4f}")
            mse_list = [m*255*255 for m in mse_list]
            logging.info(f"MSE: {np.mean(mse_list):.4f} {np.median(mse_list):.4f} {trimean(mse_list):.4f}")
            logging.info(f"MAE: {np.mean(mae_list):.4f} {np.median(mae_list):.4f} {trimean(mae_list):.4f}")
            logging.info(f"E76: {np.mean(e76_list):.4f} {np.median(e76_list):.4f} {trimean(e76_list):.4f}")

            return metrics

class deltaE76():
    def __init__(self):
        super().__init__()

    def compute(self, img1, img2, color_chart_area=0):
        """ Compute the deltaE76 between two numpy RGB images
        From M. Afifi: https://github.com/mahmoudnafifi/WB_sRGB/blob/master/WB_sRGB_Python/evaluation/calc_deltaE.py
        :param img1: numpy RGB image
        :param img2: numpy RGB image
        :return: deltaE76
        """

        # Convert to Lab
        img1 = color.rgb2lab(img1)
        img2 = color.rgb2lab(img2)

        # reshape to 1D array
        img1 = img1.reshape(-1, 3).astype(np.float32)
        img2 = img2.reshape(-1, 3).astype(np.float32)

        # compute deltaE76
        de76 = np.sqrt(np.sum(np.power(img1 - img2, 2), 1))

        return sum(de76)


class deltaE00():
    def __init__(self):
        super().__init__()
        self.kl = 1
        self.kc = 1
        self.kh = 1

    def compute(self, img1, img2, color_chart_area=0):
        """ Compute the deltaE00 between two numpy RGB images
        From M. Afifi: https://github.com/mahmoudnafifi/WB_sRGB/blob/master/WB_sRGB_Python/evaluation/calc_deltaE2000.py
        :param img1: numpy RGB image
        :param img2: numpy RGB image
        :return: deltaE00
        """

        # Convert to Lab
        img1 = color.rgb2lab(img1)
        img2 = color.rgb2lab(img2)

        # reshape to 1D array
        img1 = img1.reshape(-1, 3).astype(np.float32)
        img2 = img2.reshape(-1, 3).astype(np.float32)

        # compute deltaE00
        Lstd = np.transpose(img1[:, 0])
        astd = np.transpose(img1[:, 1])
        bstd = np.transpose(img1[:, 2])
        Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
        Lsample = np.transpose(img2[:, 0])
        asample = np.transpose(img2[:, 1])
        bsample = np.transpose(img2[:, 2])
        Cabsample = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
        Cabarithmean = (Cabstd + Cabsample) / 2
        G = 0.5 * (1 - np.sqrt((np.power(Cabarithmean, 7)) / (np.power(
            Cabarithmean, 7) + np.power(25, 7))))
        apstd = (1 + G) * astd
        apsample = (1 + G) * asample
        Cpsample = np.sqrt(np.power(apsample, 2) + np.power(bsample, 2))
        Cpstd = np.sqrt(np.power(apstd, 2) + np.power(bstd, 2))
        Cpprod = (Cpsample * Cpstd)
        zcidx = np.argwhere(Cpprod == 0)
        hpstd = np.arctan2(bstd, apstd)
        hpstd[np.argwhere((np.abs(apstd) + np.abs(bstd)) == 0)] = 0
        hpsample = np.arctan2(bsample, apsample)
        hpsample = hpsample + 2 * np.pi * (hpsample < 0)
        hpsample[np.argwhere((np.abs(apsample) + np.abs(bsample)) == 0)] = 0
        dL = (Lsample - Lstd)
        dC = (Cpsample - Cpstd)
        dhp = (hpsample - hpstd)
        dhp = dhp - 2 * np.pi * (dhp > np.pi)
        dhp = dhp + 2 * np.pi * (dhp < (-np.pi))
        dhp[zcidx] = 0
        dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
        Lp = (Lsample + Lstd) / 2
        Cp = (Cpstd + Cpsample) / 2
        hp = (hpstd + hpsample) / 2
        hp = hp - (np.abs(hpstd - hpsample) > np.pi) * np.pi
        hp = hp + (hp < 0) * 2 * np.pi
        hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]
        Lpm502 = np.power((Lp - 50), 2)
        Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
        Sc = 1 + 0.045 * Cp
        T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + \
            0.32 * np.cos(3 * hp + np.pi / 30) \
            - 0.20 * np.cos(4 * hp - 63 * np.pi / 180)
        Sh = 1 + 0.015 * Cp * T
        delthetarad = (30 * np.pi / 180) * np.exp(
            - np.power((180 / np.pi * hp - 275) / 25, 2))
        Rc = 2 * np.sqrt((np.power(Cp, 7)) / (np.power(Cp, 7) + np.power(25, 7)))
        RT = - np.sin(2 * delthetarad) * Rc
        klSl = self.kl * Sl
        kcSc = self.kc * Sc
        khSh = self.kh * Sh
        de00 = np.sqrt(np.power((dL / klSl), 2) + np.power((dC / kcSc), 2) +
                       np.power((dH / khSh), 2) + RT * (dC / kcSc) * (dH / khSh))

        return np.sum(de00) / (np.shape(de00)[0] - color_chart_area)

def calc_mae(source, target, color_chart_area=0):
  source = np.reshape(source, [-1, 3]).astype(np.float32)
  target = np.reshape(target, [-1, 3]).astype(np.float32)
  source_norm = np.sqrt(np.sum(np.power(source, 2), 1))
  target_norm = np.sqrt(np.sum(np.power(target, 2), 1))
  norm = source_norm * target_norm
  L = np.shape(norm)[0]
  inds = norm != 0
  angles = np.sum(source[inds, :] * target[inds, :], 1) / norm[inds]
  angles[angles > 1] = 1
  f = np.arccos(angles)
  f[np.isnan(f)] = 0
  f = f * 180 / np.pi
  return np.sum(f)

def trimean(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    median = sorted_values[n // 2]

    if n % 2 == 0:
        lower_half = sorted_values[:n // 2]
        upper_half = sorted_values[n // 2:]
        q1 = (lower_half[len(lower_half) // 2 - 1] + lower_half[len(lower_half) // 2]) / 2
        q3 = (upper_half[len(upper_half) // 2 - 1] + upper_half[len(upper_half) // 2]) / 2
    else:
        lower_half = sorted_values[:n // 2]
        upper_half = sorted_values[n // 2 + 1:]
        q1 = lower_half[len(lower_half) // 2]
        q3 = upper_half[len(upper_half) // 2]

    trimean = (q1 + 2 * median + q3) / 4
    return trimean