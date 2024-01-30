import os
import numpy as np
import hdbscan
import pickle
import tensorboardX
import torch
from torch.nn import functional as F
from torch import optim
import tqdm
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage

from torch_ngp.nerf.utils import Trainer
from autolabel.dataset import SAM_BIT_LEN

DEPTH_EPSILON = 0.01

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class SimpleTrainer(Trainer):

    def train(self, dataloader, epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name))

        if self.model.cuda_ray:
            self.model.mark_untrained_grid(dataloader._data.poses,
                                           dataloader._data.intrinsics)

        if not hasattr(self, 'con_ema'):
            self.con_ema = ExponentialMovingAverage(self.model.contrastive_features.parameters(),
                            decay=self.ema_decay if self.ema_decay is not None else 0.95)

        for i in range(0, epochs):
            self.train_iterations(dataloader, 1000, epoch=i+1)
            if self.opt.slow_center:
                self.update_sam_centers(dataloader)
            self.epoch += 1

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def update_sam_centers(self, dataloader):
        dataset = dataloader._data
        self.model.eval()

        bar = tqdm(dataset.indices, desc="Updating SAM centers")
        with torch.inference_mode():
            for image_index in bar:
                data = dataset._next_update(image_index)
                rays_o = torch.tensor(data['rays_o']).to(self.device)  # [B, 3]
                rays_d = torch.tensor(data['rays_d']).to(self.device)  # [B, 3]
                direction_norms = torch.tensor(data['direction_norms']).to(self.device)  # [B, 1]
                num_masks = data['num_masks']
                sample_mask_size = data['sample_mask_size']

                outputs = self.model.render(rays_o,
                                        rays_d,
                                        direction_norms,
                                        staged=False,
                                        bg_color=None,
                                        perturb=True,
                                        contrastive_ema=self.con_ema,
                                        **vars(self.opt))
                contrastive_features = outputs['contrastive_features']
                contrastive_features = contrastive_features.reshape(num_masks, sample_mask_size, -1)
                sam_center = torch.mean(contrastive_features, dim=1)
                dataset.update_sam_centers(image_index, sam_center.cpu().numpy())
    
    def train_iterations(self, dataloader, iterations, epoch):
        self.model.train()
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(dataloader._data.poses,
                                           dataloader._data.intrinsics)
        iterator = iter(dataloader)
        bar = tqdm(range(iterations), desc=f"[Epoch {epoch}] Loss: N/A")
        for _ in bar:
            data = next(iterator)
            self.global_step += 1
            for opt in self.optimizers:
                opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                _, _, loss = self.train_step(data)
            if self.use_tensorboardX:
                self.writer.add_scalar("train/loss", loss.item(),
                                       self.global_step)
            self.scaler.scale(loss).backward()
            for opt in self.optimizers:
                self.scaler.step(opt)
            self.scaler.update()
            self.con_ema.update()
            bar.set_description(f"[Epoch {epoch}] Loss: {loss:.04f}")
        if self.ema is not None:
            self.ema.update()
        self._step_scheduler(loss)

    def compute_contrastive_loss(self, 
                                 features, 
                                 sam_sampling=True,
                                 anchor_indices=None,
                                 positive_indices=None,
                                 negative_indices=None, 
                                 sam_centers=None,
                                 sam_labels=None, 
                                 batch_size=None, 
                                 chunk_size=None):
        if sam_sampling:
            assert anchor_indices is not None
            assert positive_indices is not None
            assert negative_indices is not None

            if not hasattr(self, 'con_loss_fn'):
                self.con_loss_fn = torch.nn.CrossEntropyLoss()

            # loss_all = 0
            anchor_features = features[anchor_indices]
            positive_features = features[positive_indices].detach()
            negative_features = features[negative_indices].detach()

            logits_pos = F.cosine_similarity(anchor_features, positive_features, dim=-1)
            logits_neg = F.cosine_similarity(anchor_features[:, None, :], negative_features, dim=-1)
            logits = torch.cat((logits_pos[:, None], logits_neg), dim=1)

            labels = torch.zeros(anchor_features.shape[0], dtype=torch.int64).to(self.device)
            loss_all = self.con_loss_fn(logits/self.opt.contrastive_temperature, labels)

            if sam_centers is not None:
                loss_center = F.l1_loss(anchor_features, sam_centers)
                loss_center += (1 - F.cosine_similarity(anchor_features, sam_centers, dim=-1)).mean()
                loss_all += 0.5 * loss_center                   

        else:
            assert sam_labels is not None
            assert batch_size is not None
            assert chunk_size is not None
            assert features.shape[0] == sam_labels.shape[0]

            loss_all = 0
            chunks = batch_size // chunk_size
            for chunk in range(chunks):
                start = chunk * chunk_size
                end = (chunk + 1) * chunk_size

                contrastive_features = features[start: end]
                labels = sam_labels[start: end]

                num_features = chunk_size
                con_feature_sim_mat = sim_matrix(contrastive_features, contrastive_features)

                loss_contrastive = 0
                ONE = torch.tensor(1, device=con_feature_sim_mat.device)
                for i in range(SAM_BIT_LEN):
                    label = torch.bitwise_and(ONE << i, labels).bool()
                    label_mask = (label.expand(num_features, num_features).transpose(1, 0)
                                        == label.expand(num_features, num_features))

                    self_mask = torch.eye(
                        con_feature_sim_mat.shape[0], dtype=torch.bool, device=con_feature_sim_mat.device)

                    zero_mask = torch.logical_or(
                        torch.sum(label_mask, dim=1) <= 1, 
                        torch.sum(label_mask, dim=1) >= num_features - 1
                    )
                    zero_mask = zero_mask.expand(num_features, num_features).transpose(1, 0)

                    label_mask = label_mask / self.opt.contrastive_temperature
                    loss = - torch.logsumexp(
                        con_feature_sim_mat.masked_fill(torch.logical_not(
                            label_mask), -6e4).masked_fill(self_mask, -6e4).masked_fill(zero_mask, 0),
                        dim=-1
                    ) + torch.logsumexp(
                        con_feature_sim_mat.masked_fill(zero_mask, 0),
                        dim=-1
                    )
                    loss_contrastive += loss.mean()
                loss_contrastive = loss_contrastive / SAM_BIT_LEN
                loss_all += loss_contrastive
            loss_all = loss_all / chunks
        return loss_all
    
    def train_step(self, data):
        rays_o = data['rays_o'].to(self.device)  # [B, 3]
        rays_d = data['rays_d'].to(self.device)  # [B, 3]
        direction_norms = data['direction_norms'].to(self.device)  # [B, 1]
        gt_rgb = data['pixels'].to(self.device)  # [B, 3]
        gt_depth = data['depth'].to(self.device)  # [B, 3]

        outputs = self.model.render(rays_o,
                                    rays_d,
                                    direction_norms,
                                    staged=False,
                                    bg_color=None,
                                    perturb=True,
                                    **vars(self.opt))

        pred_rgb = outputs['image']

        loss = self.opt.rgb_weight * self.criterion(pred_rgb, gt_rgb).mean()

        pred_depth = outputs['depth']
        has_depth = (gt_depth > DEPTH_EPSILON)
        depth_loss = torch.abs(pred_depth[has_depth] - gt_depth[has_depth])

        loss = loss + self.opt.depth_weight * depth_loss.mean()

        if self.opt.feature_loss:
            gt_features = data['features'].to(self.device)
            p_features = outputs['semantic_features']
            loss_feature = F.l1_loss(
                p_features[:, :gt_features.shape[1]], gt_features)
            loss += self.opt.feature_weight * loss_feature
            if self.use_tensorboardX:
                self.writer.add_scalar("train/loss_feature", loss_feature.item(),
                                        self.global_step)

        if self.opt.feature_constrastive_learning:
            if self.opt.sam_sampling:
                anchor_indices = data['anchor_indices'].to(self.device)
                positive_indices = data['positive_indices'].to(self.device)
                negative_indices = data['negative_indices'].to(self.device)
                sam_centers = data['sam_centers']
                if sam_centers is not None:
                    sam_centers = sam_centers.to(self.device)
                loss_contrastive = self.compute_contrastive_loss(
                    features=outputs['contrastive_features'],
                    sam_sampling=self.opt.sam_sampling,
                    anchor_indices=anchor_indices,
                    positive_indices=positive_indices,
                    negative_indices=negative_indices,
                    sam_centers=sam_centers
                )
            else:
                sam = data['sam'].to(self.device)
                chunk_size = data['chunk_size']
                batch_size = len(sam)
                loss_contrastive = self.compute_contrastive_loss(
                    features=outputs['contrastive_features'],
                    sam_sampling=self.opt.sam_sampling,
                    sam_labels=sam,
                    batch_size=batch_size,
                    chunk_size=chunk_size
                )
            
            loss += self.opt.contrastive_weight * loss_contrastive

            if self.use_tensorboardX:
                self.writer.add_scalar("train/loss_contrastive", loss_contrastive.item(),
                                        self.global_step)

        return pred_rgb, gt_rgb, loss

    def compute_instance_centers(self, dataset):
        self.log("Start computing instance centers ...")
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                self.model.eval()
                instance_features = []
                for i in tqdm(dataset.indices, desc="Processing contrastive features"):
                    batch = dataset._get_test(i)
                    # get instance and semantic features
                    rays_o = torch.tensor(batch['rays_o']).to(self.device)
                    rays_d = torch.tensor(batch['rays_d']).to(self.device)
                    direction_norms = torch.tensor(batch['direction_norms']).to(self.device)
                    outputs = self.model.render(rays_o,
                                                rays_d,
                                                direction_norms,
                                                staged=True,
                                                perturb=False)
                    instance_feature = outputs['contrastive_features'].cpu().numpy()
                    instance_features.append(instance_feature)
        instance_features = np.stack(instance_features, axis=0)

        # feature clustering
        self.log("Clustering features ...")
        num_image, image_height, image_width, feature_dim = instance_features.shape
        instance_features = instance_features.reshape(-1, feature_dim)
        clust = hdbscan.HDBSCAN(min_cluster_size=100, gen_min_span_tree=True) # cluster size depends on the image size
        sample_indices = np.random.permutation(instance_features.shape[0])[:200000]
        clust.fit(instance_features[sample_indices, :])

        exemplar = [np.mean(exemplars, axis=0) for exemplars in clust.exemplars_]
        exemplar = np.vstack(exemplar)
        self.log(f"Total {len(clust.exemplars_)} instance centers.")

        self.model.set_instance_centers(exemplar)
        self.model.set_instance_clusterer(clust)

    def save_instance_centers(self, save_cluster=True):
        name = f'{self.name}_ep{self.epoch:04d}_instance_centers'
        file_path = f"{self.ckpt_path}/{name}.npy"
        np.save(file_path, self.model.instance_centers)

        if save_cluster:
            name = f'{self.name}_ep{self.epoch:04d}_cluster'
            file_path = f"{self.ckpt_path}/{name}.pkl"
            with open(file_path, 'wb') as outp:
                pickle.dump(self.model.instance_clusterer, outp, pickle.HIGHEST_PROTOCOL)
    
    def test_step(self, data):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        direction_norms = data['direction_norms']  # [B, N, 1]
        H, W = data['H'], data['W']

        outputs = self.model.render(rays_o,
                                    rays_d,
                                    direction_norms,
                                    staged=True,
                                    perturb=False,
                                    **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)
        pred_semantic = outputs['semantic']
        pred_features = outputs['semantic_features']
        _, _, C = pred_semantic.shape
        pred_semantic = pred_semantic.reshape(-1, H, W, C)

        return pred_rgb, pred_depth, pred_semantic, pred_features

    def eval_step(self, data):
        rays_o = data['rays_o'].to(self.device)  # [B, 3]
        rays_d = data['rays_d'].to(self.device)  # [B, 3]
        direction_norms = data['direction_norms'].to(self.device)  # [B, 1]
        gt_rgb = data['pixels'].to(self.device)  # [B, H, W, 3]
        gt_depth = data['depth'].to(self.device)  # [B, H, W]
        gt_semantic = data['semantic'].to(self.device)  # [B, H, W]
        H, W, _ = gt_rgb.shape

        outputs = self.model.render(rays_o,
                                    rays_d,
                                    direction_norms,
                                    staged=True,
                                    bg_color=None,
                                    perturb=False,
                                    **vars(self.opt))

        pred_rgb = outputs['image'].reshape(H, W, 3)
        pred_depth = outputs['depth'].reshape(H, W)
        pred_semantic = outputs['semantic']

        loss = self.criterion(pred_rgb, gt_rgb).mean()
        has_depth = gt_depth > DEPTH_EPSILON
        loss += self.opt.depth_weight * torch.abs(pred_depth[has_depth] -
                                                  gt_depth[has_depth]).mean()

        has_semantic = gt_semantic >= 0
        if has_semantic.sum().item() > 0:
            semantic_loss = F.cross_entropy(pred_semantic[has_semantic, :],
                                            gt_semantic[has_semantic])
            loss += self.opt.semantic_weight * semantic_loss

        pred_semantic = pred_semantic.reshape(H, W, pred_semantic.shape[-1])

        return pred_rgb[None], pred_depth[None], pred_semantic[None], gt_rgb[
            None], loss

    def _step_scheduler(self, loss):
        if isinstance(self.lr_schedulers[0],
                      optim.lr_scheduler.ReduceLROnPlateau):
            [s.step(loss) for s in self.lr_schedulers]
        else:
            [s.step() for s in self.lr_schedulers]


class InteractiveTrainer(SimpleTrainer):

    def __init__(self, *args, **kwargs):
        lr_scheduler = kwargs['lr_scheduler']
        kwargs['lr_scheduler'] = None
        super().__init__(*args, **kwargs)
        self.loader = None
        self.lr_scheduler = lr_scheduler(self.optimizer)

    def init(self, loader):
        self.model.train()
        self.iterator = iter(loader)
        self.step = 0
        self.model.mark_untrained_grid(loader._data.poses,
                                       loader._data.intrinsics)

    def train(self, loader):
        while True:
            self.model.train()
            self.train_one_epoch(loader)

    def train_one_epoch(self, loader):
        iterator = iter(loader)
        bar = tqdm(range(1000), desc="Loss: N/A")
        for _ in bar:
            data = next(iterator)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                _, _, loss = self.train_step(data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            bar.set_description(f"Loss: {loss:.04f}")
        if self.ema is not None:
            self.ema.update()
        self._step_scheduler(loss)

    def take_step(self):
        data = next(self.iterator)
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.fp16):
            _, _, loss = self.train_step(data)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.step += 1
        if self.step % 100 == 0:
            self.ema.update()
            self._step_scheduler(loss)
        return loss

    def dataset_updated(self, loader):
        self.loader = loader
