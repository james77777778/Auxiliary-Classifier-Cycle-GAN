import os
import os.path as osp
from datetime import datetime
import itertools
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from absl import flags, app
import numpy as np

from src.utils.dataset import UnalignedDataset
from src.models.cyclegan import CycleGAN, ACCycleGAN, init_weights
from src.models.loss import GANLoss, PatchLoss
from src.utils.image_pool import ImageClassPool
from src.utils.utils import dict2table, set_seed


FLAGS = flags.FLAGS
# train steps
flags.DEFINE_integer('epochs', 200, "total number of training epochs")
flags.DEFINE_integer('batch_size', 1, "number of batch size")
flags.DEFINE_integer('eval_step', 10, "number of epoch to evaluate")
flags.DEFINE_integer('save_step', 50, "number of epoch to save model")
# model type
flags.DEFINE_enum('model', 'cyclegan', ["cyclegan", "accyclegan"],
                  "model type")
# training parameters
flags.DEFINE_float('lr', 2e-4, "learning rate")
flags.DEFINE_integer('image_size', 256, "image size for training")
flags.DEFINE_float('lambda_A', 10.0, "weight of G_A")
flags.DEFINE_float('lambda_B', 10.0, "weight of G_B")
flags.DEFINE_float('lambda_idt', 0.1, "weight of identity loss")
# loss type
flags.DEFINE_string('loss', 'L1', "loss type for CycleGAN")
# dataset
flags.DEFINE_string(
    'dataset', 'summer2winter_yosemite',
    "dataset name at ./dataset/[dataset_name]")
# logging
flags.DEFINE_string('logdir', 'logs', "log directory")
flags.DEFINE_string('run_name', 'cyc_gan', 'specify run name')


class CycleGANTrainer(object):
    def __init__(self):
        # parameters
        set_seed(2020)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            # extremely slow if use cpu to train
            self.device = torch.device("cpu")

        # dataset
        self.train_dataset = UnalignedDataset(
            osp.join("dataset", FLAGS.dataset, "train"),
            image_size=FLAGS.image_size)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=FLAGS.batch_size,
                                       shuffle=True,
                                       num_workers=2)
        self.test_dataset = UnalignedDataset(
            osp.join("dataset", FLAGS.dataset, "test"), is_train=True,
            image_size=FLAGS.image_size)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=1,  # use 1 for evaluatation
                                      shuffle=False)

        # replay buffer
        self.fake_A_pool = ImageClassPool(50)
        self.fake_B_pool = ImageClassPool(50)

        # model
        if FLAGS.model == "cyclegan":
            self.model = CycleGAN().to(self.device)
            init_weights(self.model, init_type="kaiming")
        elif FLAGS.model == "accyclegan":
            self.model = ACCycleGAN().to(self.device)
            init_weights(self.model, init_type="kaiming")

        # loss
        self.criterionGAN = GANLoss("lsgan").to(self.device)
        if FLAGS.loss == "L1":
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
        elif FLAGS.loss == "patch":
            self.criterionCycle = PatchLoss().to(self.device)
            self.criterionIdt = PatchLoss().to(self.device)

        # opt
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(
                self.model.netG_A.parameters(),
                self.model.netG_B.parameters()
            ),
            lr=FLAGS.lr,
            betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.model.netD_A.parameters(),
                                              lr=FLAGS.lr,
                                              betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.model.netD_B.parameters(),
                                              lr=FLAGS.lr,
                                              betas=(0.5, 0.999))
        self.schedulers = [
            self.get_scheduler(opt) for opt in
            [self.optimizer_G, self.optimizer_D_A, self.optimizer_D_B]
        ]

        # logs
        self.run_name = (datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "-" +
                         FLAGS.run_name)
        self.log_dir = osp.join(FLAGS.logdir, self.run_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.save_dir = osp.join(self.log_dir, "model_states")
        os.makedirs(self.save_dir, exist_ok=True)

        # write params to summary
        self.writer.add_text('Text', dict2table(FLAGS.flag_values_dict()), 0)

    def train(self, batch):
        self.model.train()
        device = self.device
        real_A, real_B = batch["A"].to(device), batch["B"].to(device)
        c_A, c_B = batch["class_A"].to(device), batch["class_B"].to(device)
        lambda_A, lambda_B = FLAGS.lambda_A, FLAGS.lambda_B
        lambda_idt = FLAGS.lambda_idt
        # forward
        res = self.model(real_A, real_B)
        # train G
        self.model.set_requires_grad([self.model.netD_A, self.model.netD_B],
                                     False)
        # idt loss
        idt_A = self.model.netG_A(real_B)
        loss_idt_A = self.criterionIdt(idt_A, real_B) * lambda_A * lambda_idt
        idt_B = self.model.netG_B(real_A)
        loss_idt_B = self.criterionIdt(idt_B, real_A) * lambda_B * lambda_idt
        # gan loss
        D_A_res = self.model.netD_A(res["fake_B"])
        D_B_res = self.model.netD_B(res["fake_A"])
        loss_G_A = self.criterionGAN(D_A_res["realness"], True)
        loss_G_B = self.criterionGAN(D_B_res["realness"], True)
        if FLAGS.model == "cyclegan":
            loss_G_Ac = loss_G_Bc = 0
        elif FLAGS.model == "accyclegan":
            loss_G_Ac = self.criterionGAN(D_A_res["class"], c_A)
            loss_G_Bc = self.criterionGAN(D_B_res["class"], c_B)
        # cycle loss
        loss_cycle_A = self.criterionCycle(res["rec_A"], real_A) * lambda_A
        loss_cycle_B = self.criterionCycle(res["rec_B"], real_B) * lambda_B
        loss_G = (loss_G_A + loss_G_B + loss_G_Ac + loss_G_Bc +
                  loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B)
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()
        # train D
        self.model.set_requires_grad([self.model.netD_A, self.model.netD_B],
                                     True)
        # train D_A
        fake_B, c_B = self.fake_B_pool.query(res["fake_B"], c_B)
        real_res_A = self.model.netD_A(real_B)
        fake_res_A = self.model.netD_A(fake_B)
        pred_A_real = real_res_A["realness"]
        pred_A_class1 = real_res_A["class"]
        pred_A_fake = fake_res_A["realness"]
        pred_A_class2 = fake_res_A["class"]
        loss_D_A_real = self.criterionGAN(pred_A_real, True)
        loss_D_A_fake = self.criterionGAN(pred_A_fake, False)
        if FLAGS.model == "cyclegan":
            loss_D_A_class1 = loss_D_A_class2 = 0
        elif FLAGS.model == "accyclegan":
            loss_D_A_class1 = self.criterionGAN(pred_A_class1, c_B)
            loss_D_A_class2 = self.criterionGAN(pred_A_class2, c_B)
        loss_D_A = ((loss_D_A_real + loss_D_A_fake) * 0.5 +
                    (loss_D_A_class1 + loss_D_A_class2) * 0.5)
        self.optimizer_D_A.zero_grad()
        loss_D_A.backward()
        self.optimizer_D_A.step()
        # train D_B
        fake_A, c_A = self.fake_A_pool.query(res["fake_A"], c_A)
        real_res_B = self.model.netD_B(real_A)
        fake_res_B = self.model.netD_B(fake_A)
        pred_B_real = real_res_B["realness"]
        pred_B_class1 = real_res_B["class"]
        pred_B_fake = fake_res_B["realness"]
        pred_B_class2 = fake_res_B["class"]
        loss_D_B_real = self.criterionGAN(pred_B_real, True)
        loss_D_B_fake = self.criterionGAN(pred_B_fake, False)
        if FLAGS.model == "cyclegan":
            loss_D_B_class1 = loss_D_B_class2 = 0
        elif FLAGS.model == "accyclegan":
            loss_D_B_class1 = self.criterionGAN(pred_B_class1, c_A)
            loss_D_B_class2 = self.criterionGAN(pred_B_class2, c_A)
        loss_D_B = ((loss_D_B_real + loss_D_B_fake) * 0.5 +
                    (loss_D_B_class1 + loss_D_B_class2) * 0.5)
        self.optimizer_D_B.zero_grad()
        loss_D_B.backward()
        self.optimizer_D_B.step()
        return {"loss_G": loss_G.item(), "loss_D_A": loss_D_A.item(),
                "loss_D_B": loss_D_B.item()}

    def evaluate(self, batch):
        self.model.train()
        device = self.device
        real_A, real_B = batch["A"].to(device), batch["B"].to(device)
        c_A, c_B = batch["class_A"].to(device), batch["class_B"].to(device)
        lambda_A, lambda_B = FLAGS.lambda_A, FLAGS.lambda_B
        lambda_idt = FLAGS.lambda_idt
        with torch.no_grad():
            # forward
            res = self.model(real_A, real_B)
            # idt loss
            idt_A = self.model.netG_A(real_B)
            loss_idt_A = (self.criterionIdt(idt_A, real_B) * lambda_A *
                          lambda_idt)
            idt_B = self.model.netG_B(real_A)
            loss_idt_B = (self.criterionIdt(idt_B, real_A) * lambda_B *
                          lambda_idt)
            # gan loss
            D_A_res = self.model.netD_A(res["fake_B"])
            D_B_res = self.model.netD_B(res["fake_A"])
            loss_G_A = self.criterionGAN(D_A_res["realness"], True)
            loss_G_B = self.criterionGAN(D_B_res["realness"], True)
            if FLAGS.model == "cyclegan":
                loss_G_Ac = loss_G_Bc = 0
            elif FLAGS.model == "accyclegan":
                loss_G_Ac = self.criterionGAN(D_A_res["class"], c_A)
                loss_G_Bc = self.criterionGAN(D_B_res["class"], c_B)
            # cycle loss
            loss_cycle_A = self.criterionCycle(res["rec_A"], real_A) * lambda_A
            loss_cycle_B = self.criterionCycle(res["rec_B"], real_B) * lambda_B
            loss_G = (loss_G_A + loss_G_B + loss_G_Ac + loss_G_Bc +
                      loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B)
            # train D
            # train D_A
            fake_B, c_B = self.fake_B_pool.query(res["fake_B"], c_B,
                                                 is_train=False)
            real_res_A = self.model.netD_A(real_B)
            fake_res_A = self.model.netD_A(fake_B)
            pred_A_real = real_res_A["realness"]
            pred_A_class1 = real_res_A["class"]
            pred_A_fake = fake_res_A["realness"]
            pred_A_class2 = fake_res_A["class"]
            loss_D_A_real = self.criterionGAN(pred_A_real, True)
            loss_D_A_fake = self.criterionGAN(pred_A_fake, False)
            if FLAGS.model == "cyclegan":
                loss_D_A_class1 = loss_D_A_class2 = 0
            elif FLAGS.model == "accyclegan":
                loss_D_A_class1 = self.criterionGAN(pred_A_class1, c_B)
                loss_D_A_class2 = self.criterionGAN(pred_A_class2, c_B)
            loss_D_A = ((loss_D_A_real + loss_D_A_fake) * 0.5 +
                        (loss_D_A_class1 + loss_D_A_class2) * 0.5)
            # train D_B
            fake_A, c_A = self.fake_A_pool.query(res["fake_A"], c_A,
                                                 is_train=False)
            real_res_B = self.model.netD_B(real_A)
            fake_res_B = self.model.netD_B(fake_A)
            pred_B_real = real_res_B["realness"]
            pred_B_class1 = real_res_B["class"]
            pred_B_fake = fake_res_B["realness"]
            pred_B_class2 = fake_res_B["class"]
            loss_D_B_real = self.criterionGAN(pred_B_real, True)
            loss_D_B_fake = self.criterionGAN(pred_B_fake, False)
            if FLAGS.model == "cyclegan":
                loss_D_B_class1 = loss_D_B_class2 = 0
            elif FLAGS.model == "accyclegan":
                loss_D_B_class1 = self.criterionGAN(pred_B_class1, c_A)
                loss_D_B_class2 = self.criterionGAN(pred_B_class2, c_A)
            loss_D_B = ((loss_D_B_real + loss_D_B_fake) * 0.5 +
                        (loss_D_B_class1 + loss_D_B_class2) * 0.5)
        return {"res": res, "loss_G": loss_G.item(),
                "loss_D_A": loss_D_A.item(), "loss_D_B": loss_D_B.item()}

    def get_scheduler(self, optimizer):
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - FLAGS.epochs/2) / float(FLAGS.epochs/2)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return scheduler

    def update_lr(self):
        for sch in self.schedulers:
            sch.step()

    def save_checkpoint(self, checkpoint, save_path, save_name):
        save_path = osp.join(save_path, 'model_{}.pth'.format(save_name))
        torch.save(checkpoint, save_path)

    def main(self):
        iters = 1
        device = self.device
        total = (
            np.ceil(len(self.train_loader.dataset) /
                    self.train_loader.batch_size) *
            FLAGS.epochs)
        pbar = tqdm(total=total, dynamic_ncols=True, position=0)
        for epo in range(FLAGS.epochs):
            # train
            for i, batch in enumerate(self.train_loader):
                train_loss = self.train(batch)
                iters += 1
                # pbar
                pbar.set_postfix({'epo': epo})
                self.writer.add_scalar(
                    "train/G_L1", train_loss["loss_G"], iters)
                self.writer.add_scalar(
                    "train/D_A", train_loss["loss_D_A"], iters)
                self.writer.add_scalar(
                    "train/D_B", train_loss["loss_D_B"], iters)
                pbar.update(1)
            self.writer.add_scalar("train/epoch", epo+1, iters)
            # test
            if (epo % FLAGS.eval_step == 0) or (epo == FLAGS.epochs-1):
                total_loss_G = 0
                total_loss_D_A = 0
                total_loss_D_B = 0
                for i, batch in enumerate(self.test_loader):
                    test_loss = self.evaluate(batch)
                    total_loss_G += test_loss["loss_G"]
                    total_loss_D_A += test_loss["loss_D_A"]
                    total_loss_D_B += test_loss["loss_D_B"]
                    if i < 5:
                        res = test_loss["res"]
                        images = torch.cat((
                            batch["A"].to(device), res["fake_B"],
                            batch["B"].to(device), res["fake_A"]))
                        # denormalize
                        images = self.test_dataset.denormalize(images)
                        grid = torchvision.utils.make_grid(images)
                        self.writer.add_image("idx_{}".format(i), grid, iters)
                avg_loss_G = total_loss_G/len(self.test_loader.dataset)
                avg_loss_D_A = total_loss_D_A/len(self.test_loader.dataset)
                avg_loss_D_B = total_loss_D_B/len(self.test_loader.dataset)
                self.writer.add_scalar("test/G_L1", avg_loss_G, iters)
                self.writer.add_scalar("test/D_A", avg_loss_D_A, iters)
                self.writer.add_scalar("test/D_B", avg_loss_D_B, iters)
            # save checkpoint
            if (epo % FLAGS.save_step == 0) or (epo == FLAGS.epochs-1):
                checkpoint = {
                    "epoch": epo,
                    "model": self.model.state_dict()}
                self.save_checkpoint(
                    checkpoint, self.save_dir, "{}".format(epo))
            # update lr
            self.update_lr()
        print("Finish training: model states at {}".format(self.save_dir))


def main(agrv):
    # start training
    trainer = CycleGANTrainer()
    trainer.main()


if __name__ == "__main__":
    app.run(main)
