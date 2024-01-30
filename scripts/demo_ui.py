import sys
import argparse
import os
import open3d as o3d
from plyfile import PlyData
import numpy as np
import torch
import torch.nn.functional as F
import threading
import multiprocessing as mp
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from sklearn.metrics.pairwise import cosine_similarity
from autolabel.constants import COLORS
from autolabel.utils.feature_utils import get_feature_extractor
from autolabel.dataset import SceneDataset
from autolabel import utils, model_utils


class PointCloudVisualizer:

    def __init__(self, flags, queue):
        self.flags = flags
        self.queue = queue
        self.visualizer = o3d.visualization.Visualizer()
        self.label_mapping = dict()
        self._load_scene_model()
        self._load_pointcloud()
        self._load_text_model()
        self._load_point_features()
        self._load_point_instance_ids()

    def _load_pointcloud(self):
        mesh_path = os.path.join(self.flags.scene, "mesh.ply")
        if not os.path.exists(mesh_path):
            raise ValueError(f"Mesh file {mesh_path} does not exist.")
        plydata = PlyData.read(mesh_path)
        points = np.hstack([
            plydata['vertex']['x'].reshape(-1, 1),
            plydata['vertex']['y'].reshape(-1, 1),
            plydata['vertex']['z'].reshape(-1, 1)
        ])
        points_rgb = np.hstack([
            plydata['vertex']['red'].reshape(-1, 1),
            plydata['vertex']['green'].reshape(-1, 1),
            plydata['vertex']['blue'].reshape(-1, 1)
        ])
        points_rgb = points_rgb.astype(np.float32) / 255.0
        aabb = np.loadtxt(
            os.path.join(self.flags.scene, 'bbox.txt')
        )[:6].reshape(2, 3)
        scene_center = (aabb[0] + aabb[1]) / 2
        points = points - scene_center
        fixed = np.zeros_like(points)
        fixed[:, 0] = points[:, 1]
        fixed[:, 1] = points[:, 2]
        fixed[:, 2] = points[:, 0]
        self.points = torch.tensor(fixed, dtype=torch.float16)
        self.point_infos = {'ori_rgb': points_rgb}
        self.pc = o3d.geometry.PointCloud()
        self.pc.points = o3d.utility.Vector3dVector(fixed)
        self.pc.colors = o3d.utility.Vector3dVector(points_rgb)
        # self.pc.paint_uniform_color([0.5, 0.5, 0.5])
        self.visualizer.create_window()
        self.visualizer.add_geometry(self.pc)

    def _load_scene_model(self):
        models = list()
        nerf_dir = model_utils.get_nerf_dir(self.flags.scene, self.flags)
        if not os.path.exists(nerf_dir):
            raise ValueError(f"Model directory {nerf_dir} does not exist.")
        for model in os.listdir(nerf_dir):
            checkpoint_dir = os.path.join(nerf_dir, model, 'checkpoints')
            if os.path.exists(checkpoint_dir):
                models.append(model)
        model_path = os.path.join(nerf_dir, models[0])
        print("Loading models: ", model_path)
        params = model_utils.read_params(model_path)
        dataset = SceneDataset('test',
                               self.flags.scene,
                               factor=4.0,
                               batch_size=self.flags.batch_size,
                               lazy=True)
        n_classes = dataset.n_classes if dataset.n_classes is not None else 2
        model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds,
                                         n_classes, params).cuda()
        checkpoint_dir = os.path.join(model_path, 'checkpoints')
        model_utils.load_checkpoint(model, checkpoint_dir)
        self.model = model.eval()

    def _load_text_model(self):
        self.extractor = get_feature_extractor('lseg', self.flags.checkpoint)

    def _load_point_features(self):
        semantic_features = self._point_features(points=self.points)
        self.point_infos['semantic'] = semantic_features

    def _load_point_instance_ids(self):
        instance_ids = self._point_instance_ids(points=self.points)
        self.point_infos['instance_id'] = instance_ids
        instance_colors = np.zeros((len(instance_ids), 3))
        ins_ids = np.unique(instance_ids)
        for ins_id in ins_ids:
            if ins_id == 0:
                continue
            instance_colors[instance_ids == ins_id] = np.random.rand(3, )
        self.point_infos['instance_colors'] = instance_colors

    def _denoise_semantic(self, pred_semantic_labels, pred_instance_labels):
        pred_semantic_denoised = np.copy(pred_semantic_labels)
        instance_ids = np.unique(pred_instance_labels)
        for ins_id in instance_ids:
            if ins_id == 0:
                continue
            
            semantic_ids = pred_semantic_labels[pred_instance_labels == ins_id]
            ids, cnts = np.unique(semantic_ids, return_counts=True)
            pred_semantic_denoised[pred_instance_labels == ins_id] = ids[np.argmax(cnts)]
        return pred_semantic_denoised
    
    def _update_colors(self, msg):
        print(msg)
        if isinstance(msg, list):
            prompts = msg
            if len(prompts) > 0:
                # prompts.append("others")
                text_features = self.extractor.encode_text(prompts)
                semantic_features = self._point_features()
                pred_instance_labels = self._point_instance_ids()
                similarities = torch.zeros(
                    (semantic_features.shape[0], text_features.shape[0]),
                    dtype=torch.float32,
                    device=semantic_features.device)
                batch_size = 50000
                for i in range(0, semantic_features.shape[0], batch_size):
                    batch = semantic_features[i:i + batch_size]
                    for prompt_index in range(text_features.shape[0]):
                        similarities[i:i + batch_size, prompt_index] = (
                            batch * text_features[prompt_index][None]).sum(dim=-1)
                
                update_mask, _ = similarities.max(dim=-1)
                update_mask = update_mask.cpu().numpy() > 0.85
                closest_prompt = similarities.argmax(dim=-1).cpu().numpy()
                denoised_closest_prompt = self._denoise_semantic(closest_prompt, pred_instance_labels)
                
                colors = np.asarray(self.pc.colors)
                colors[update_mask] = COLORS[denoised_closest_prompt[update_mask] % COLORS.shape[0]] / 255.
            else:
                colors = self.point_infos['ori_rgb']
        elif isinstance(msg, str):
            if msg == "show_instance":
                colors = self.point_infos['instance_colors']
        else:
            raise ValueError("Not support msg type {}".format(type(msg)))
        self.pc.colors = o3d.utility.Vector3dVector(colors)
        self.visualizer.update_geometry(self.pc)

    def _point_features(self, points=None):
        if points is not None:
            out = []
            for i in range(0, len(points), self.flags.batch_size):
                batch = points[i:i + self.flags.batch_size]
                batch = batch.cuda()
                with torch.no_grad():
                    density = self.model.density(batch)
                    features = self.model.semantic(density['geo_feat'])
                    features = features / torch.norm(features, dim=-1, keepdim=True)
                features = features.to(torch.float32)
                out.append(features)
            semantic_features = torch.cat(out, dim=0)
        else:
            semantic_features = self.point_infos['semantic']
        return semantic_features
    
    def _point_instance_ids(self, points=None):
        if points is not None:
            pred_instances = []
            for i in range(0, len(points), self.flags.batch_size):
                batch = points[i:i + self.flags.batch_size]
                batch = batch.cuda()
                with torch.no_grad():
                    xyz_feature_encoding = self.model.feature_encoder(batch, bound=self.model.bound)
                    instance_feature = self.model.contrastive(xyz_feature_encoding, None)
                    # instance_feature = instance_feature.reshape(-1, feature_dim)
                    instance_feature = instance_feature.cpu().numpy()
                    sim_mat = cosine_similarity(instance_feature, self.model.instance_centers)
                    pred_instance = np.argmax(sim_mat, axis=1) + 1 # start from 1, 0 means noise
                pred_instances.append(pred_instance) 
            instance_ids = np.concatenate(pred_instances, axis=0) 
        else:
            instance_ids = self.point_infos['instance_id']
        return instance_ids

    def run(self):
        while True:
            if not self.queue.empty():
                msg = self.queue.get(False)
                self._update_colors(msg)
            self.visualizer.update_geometry(self.pc)
            if not self.visualizer.poll_events():
                return
            self.visualizer.update_renderer()


def run_visualizer(flags, queue):
    visualizer = PointCloudVisualizer(flags, queue)
    visualizer.run()


class ListView(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.items = []

    def add_item(self, item):
        index = len(self.items)
        color = COLORS[index % len(COLORS)]
        self.items.append(item)
        label = QtWidgets.QLabel(item)
        label.setMargin(20)
        label.setStyleSheet(
            f"background-color: rgb({color[0]}, {color[1]}, {color[2]});")
        self.layout.addWidget(label)
        self.update()

    def get_items(self):
        return self.items

    def reset(self):
        self.items = []
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)


class SegmentingApplication(QtWidgets.QMainWindow):

    def __init__(self, queue):
        super().__init__()
        self.classes = []
        self.setWindowTitle("Segmentation Classes")
        self.input_button = QtWidgets.QPushButton("Add")
        self.input_button.clicked.connect(self._add_class)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset_classes)
        self.show_instance_button = QtWidgets.QPushButton("Show all instances")
        self.show_instance_button.clicked.connect(self._show_all_instances)
        self.list_view = ListView()
        input_line = self._create_input_line()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list_view)
        layout.addWidget(input_line)
        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.class_queue = queue

    def _create_input_line(self):
        layout = QtWidgets.QHBoxLayout()
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setPlaceholderText("Class description prompt")
        self.line_edit.returnPressed.connect(self._add_class)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.input_button)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.show_instance_button)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()

    def _add_class(self):
        self.list_view.add_item(self.line_edit.text())
        self.line_edit.clear()
        self._publish_classes()

    def _reset_classes(self):
        self.list_view.reset()
        self._publish_classes()

    def _show_all_instances(self):
        self.class_queue.put("show_instance")

    def _publish_classes(self):
        self.class_queue.put(self.list_view.get_items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=str)
    parser.add_argument('--workspace', default=None)
    parser.add_argument('--checkpoint',
                        type=str,
                        required=True,
                        help='path to feature model checkpoint')
    # parser.add_argument('--model', type=str, default='model.pth')
    parser.add_argument('--batch-size', type=int, default=1024)
    flags = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    queue = mp.Queue()
    window = SegmentingApplication(queue)
    window.show()

    thread = threading.Thread(target=run_visualizer, args=(flags, queue))
    thread.start()
    app.exec()
    thread.join()


if __name__ == "__main__":
    main()