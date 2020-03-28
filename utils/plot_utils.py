import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_psnr(apath, epoch,psnr_vec, mode=None):
    axis = np.linspace(1, len(psnr_vec), len(psnr_vec))
    fig = plt.figure()
    plt.plot(axis, psnr_vec)
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig('{}psnr_{}.pdf'.format(apath,mode))
    plt.close(fig)


def plot_loss(apath, epoch,loss_vec):
    if len(loss_vec) == epoch:
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.plot(axis, loss_vec)
        #plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('{}loss.pdf'.format(apath))
        plt.close(fig)
    else:
        axis = np.linspace(1, len(loss_vec), len(loss_vec))
        fig = plt.figure()
        plt.plot(axis, loss_vec)
        #plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('{}loss_resume.pdf'.format(apath))
        plt.close(fig)