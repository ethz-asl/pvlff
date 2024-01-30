import argparse
import os
from pathlib import Path
import numpy as np
import json
import pickle
import pandas
from rich.table import Table
from rich.console import Console
from rich import print as rprint
from autolabel.evaluation import OpenVocabEvaluator2D, OpenVocabEvaluator3D
from autolabel.evaluation import OpenVocabInstancePQEvaluator, PanopticStat
from autolabel.dataset import SceneDataset, LenDataset
from autolabel import utils, model_utils


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenes', nargs='+')
    parser.add_argument('--batch-size', default=8182, type=int)
    parser.add_argument('--vis', default=None, type=str)
    parser.add_argument('--workspace', type=str, default=None)
    parser.add_argument('--out',
                        default=None,
                        type=str,
                        help="Where to write results as json, if anywhere.")
    parser.add_argument('--label-map', type=str, required=True)
    parser.add_argument('--feature-checkpoint', '-f', type=str, required=True)
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help="Only evaluate every Nth frame to save time or for debugging.")
    parser.add_argument(
        '--pc',
        action='store_true',
        help=
        "Evaluate point cloud segmentation accuracy instead of 2D segmentation maps."
    )
    parser.add_argument(
        '--panoptic', 
        action='store_true',
        help='Evaluate panoptic segmenation.')
    parser.add_argument('--print-verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--only-scene-classes', action='store_true')
    parser.add_argument('--random',
                        action='store_true',
                        help="Randomize the order of the scenes.")
    parser.add_argument('--time', action='store_true')
    parser.add_argument('--denoise-method', 
                        type=str, 
                        default='average_similarity',
                        choices=['majority_voting', 'average_similarity', 'average_feature'],
                        help="The denoise method for semantics.")
    return parser.parse_args()


def gather_models(flags, scene_dirs):
    models = set()
    for scene in scene_dirs:
        nerf_dir = model_utils.get_nerf_dir(scene, flags)
        if not os.path.exists(nerf_dir):
            continue
        for model in os.listdir(nerf_dir):
            checkpoint_dir = os.path.join(nerf_dir, model, 'checkpoints')
            if os.path.exists(checkpoint_dir):
                models.add(model)
    return list(models)


def read_label_map(path):
    return pandas.read_csv(path)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def write_results(out, tables, json_result, panoptic_stat=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dumped = json.dumps(json_result, cls=NumpyEncoder, indent=2)
    with open(out / 'results.json', 'w') as f:
        f.write(dumped)

    with open(out / 'table.txt', 'w') as f:
        for table in tables:
            rprint(table, file=f)
            rprint('\n\n', file=f)
    
    if panoptic_stat is not None:
        with open(out / 'panoptic_stat.pkl', 'wb') as outp:
            pickle.dump(panoptic_stat, outp, pickle.HIGHEST_PROTOCOL)


def main(flags):
    if len(flags.scenes) == 1 and not os.path.exists(
            os.path.join(flags.scenes[0], 'rgb')):
        # We are dealing with a directory full of scenes and not a list of scenes
        scene_dir = flags.scenes[0]
        scene_dirs = [
            os.path.join(scene_dir, scene)
            for scene in os.listdir(scene_dir)
            if os.path.exists(os.path.join(scene_dir, scene, 'rgb'))
        ]
    else:
        scene_dirs = flags.scenes

    original_labels = read_label_map(flags.label_map)
    n_classes = len(original_labels)

    scene_names = [os.path.basename(os.path.normpath(p)) for p in scene_dirs]
    scenes = [(s, n) for s, n in zip(scene_dirs, scene_names)]
    if flags.random:
        import random
        random.shuffle(scenes)
    else:
        scenes = sorted(scenes, key=lambda x: x[1])
    
    if flags.panoptic:
        panoptic_stats = PanopticStat()
    else:
        ious = []
        accs = []
        ious_d = []
        accs_d = []
    evaluator = None

    for scene_index, (scene, scene_name) in enumerate(scenes):
        model = gather_models(flags, [scene])
        if len(model) == 0:
            print(f"Skipping scene {scene_name} because no models were found.")
            continue
        else:
            model = model[0]
        print(f"Using model {model}")

        print(f"Evaluating scene {scene_name}")

        nerf_dir = model_utils.get_nerf_dir(scene, flags)
        model_path = os.path.join(nerf_dir, model)
        if not os.path.exists(model_path):
            print(f"Skipping scene {scene_name} because no models were found.")
            continue
        params = model_utils.read_params(model_path)
        dataset = SceneDataset('test',
                               scene,
                               factor=4.0,
                               batch_size=flags.batch_size,
                               lazy=True)
        if flags.only_scene_classes:
            classes_in_scene = dataset.scene.metadata.get('classes', None)
            if classes_in_scene is None:
                label_map = original_labels
            else:
                mask = original_labels['id'].isin(classes_in_scene)
                label_map = original_labels[mask]
        else:
            label_map = original_labels

        n_classes = dataset.n_classes if dataset.n_classes is not None else 2
        model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds,
                                         n_classes, params).cuda()

        checkpoint_dir = os.path.join(model_path, 'checkpoints')
        if not os.path.exists(checkpoint_dir) or len(
                os.listdir(checkpoint_dir)) == 0:
            continue

        model_utils.load_checkpoint(model, checkpoint_dir)
        model = model.eval()
        if flags.vis is not None:
            vis_path = os.path.join(flags.vis, scene_name)
        else:
            vis_path = None

        if evaluator is None:
            if flags.panoptic:
                evaluator = OpenVocabInstancePQEvaluator(
                    features=params.features,
                    name=scene_name,
                    checkpoint=flags.feature_checkpoint,
                    debug=flags.debug,
                    stride=flags.stride,
                    save_figures=vis_path,
                    time=flags.time,
                    denoise_method=flags.denoise_method
                )
            else:
                if flags.pc:
                    evaluator = OpenVocabEvaluator3D(
                        features=params.features,
                        name=scene_name,
                        checkpoint=flags.feature_checkpoint,
                        stride=flags.stride,
                        debug=flags.debug,
                        time=flags.time)
                else:
                    evaluator = OpenVocabEvaluator2D(
                        features=params.features,
                        name=scene_name,
                        checkpoint=flags.feature_checkpoint,
                        debug=flags.debug,
                        stride=flags.stride,
                        save_figures=vis_path,
                        time=flags.time)
        assert evaluator.features == params.features
        evaluator.reset(model, label_map, vis_path)
        if flags.panoptic:
            panoptic_stat = evaluator.eval(dataset)
            panoptic_stats += panoptic_stat
            tables, json_result = print_panoptic_results(panoptic_stat, 
                                        categories=evaluator.evaluated_labels,
                                        label_mapping=evaluator.label_mapping,
                                        label_type_mapping=evaluator.label_type_mapping,
                                        verbose=flags.print_verbose)
            if flags.out:
                write_results(
                    os.path.join(flags.out, scene_name), tables, json_result, panoptic_stat)
        else:
            iou, acc, iou_d, acc_d = evaluator.eval(dataset)
            ious.append(iou)
            accs.append(acc)
            ious_d.append(iou_d)
            accs_d.append(acc_d)
            table = print_iou_acc_results([iou], [acc])
            table_d = print_iou_acc_results([iou_d], [acc_d], table_title="Denoised")
            if flags.out:
                write_results(
                    os.path.join(flags.out, scene_name), 
                    [table, table_d], 
                    {'iou': iou, 'acc': acc, 'iou_d': iou_d, 'acc_d': acc_d})
        del model
    if flags.panoptic:
        final_tables, final_json_result = print_panoptic_results(panoptic_stats, 
                                                 categories=evaluator.evaluated_labels,
                                                 label_mapping=evaluator.label_mapping,
                                                 label_type_mapping=evaluator.label_type_mapping,
                                                 verbose=flags.print_verbose)
        if flags.out:
            write_results(
                os.path.join(flags.out, 'final'), final_tables, final_json_result, panoptic_stats)
    else:
        table = print_iou_acc_results(ious, accs)
        table_d = print_iou_acc_results(ious_d, accs_d, table_title="Denoised")
        if flags.out:
            write_results(
                os.path.join(flags.out, 'final'),
                [table, table_d], 
                {'ious': ious, 'accs': accs, 'ious_d': ious_d, 'accs_d': accs_d})


def print_panoptic_results(panoptic_stat, categories, label_mapping, label_type_mapping, verbose=False):

    json_result = {}
    print_tables = []

    def percentage_to_string(num):
        if num is None:
            return "N/A"
        else:
            v = num * 100
            return f"{v:.1f}"

    console = Console()
    # panoptic segmentation
    pq_total_result, pq_per_class_result = panoptic_stat.pq_average(categories, label_type_mapping, verbose=verbose)
    table = Table(show_lines=True, caption_justify='left')
    table.add_column('Class')
    table.add_column('PQ')
    table.add_column('SQ')
    table.add_column('RQ')
    if verbose:
        table.add_column('tp')
        table.add_column('fp')
        table.add_column('fn')

    table.title = "Panoptic Evaluation"
    json_result['panoptic'] = {}
    per_class_result = {}
    for category_id in categories:
        pq_info = pq_per_class_result[category_id]
        if pq_info['valid']:
            if verbose:
                table.add_row(label_mapping[category_id], 
                        percentage_to_string(pq_info['pq']),
                        percentage_to_string(pq_info['sq']),
                        percentage_to_string(pq_info['rq']),
                        str(pq_info['tp']),
                        str(pq_info['fp']),
                        str(pq_info['fn']))
                per_class_result[label_mapping[category_id]] = {
                    'PQ': pq_info['pq'] * 100, 'SQ': pq_info['sq'] * 100, 'RQ': pq_info['rq'] * 100,
                    'tp': pq_info['tp'], 'fp': pq_info['fp'], 'fn': pq_info['fn']
                }
            
            else:
                table.add_row(label_mapping[category_id], 
                        percentage_to_string(pq_info['pq']),
                        percentage_to_string(pq_info['sq']),
                        percentage_to_string(pq_info['rq']))
                per_class_result[label_mapping[category_id]] = {
                    'PQ': pq_info['pq'] * 100, 'SQ': pq_info['sq'] * 100, 'RQ': pq_info['rq'] * 100
                }
    json_result['panoptic']['per_class_result'] = per_class_result
    if verbose:
        table.add_row('Total:\n{} valid panoptic categories.'.format(
                        pq_total_result['n']),
                  percentage_to_string(pq_total_result['pq']), 
                  percentage_to_string(pq_total_result['sq']), 
                  percentage_to_string(pq_total_result['rq']),
                  '{:.1f}'.format(pq_total_result['tp']),
                  '{:.1f}'.format(pq_total_result['fp']),
                  '{:.1f}'.format(pq_total_result['fn']))
        json_result['panoptic']['total'] = {
            'PQ': pq_total_result['pq'] * 100, 'SQ': pq_total_result['sq'] * 100, 'RQ': pq_total_result['rq'] * 100,
            'tp': pq_total_result['tp'], 'fp': pq_total_result['fp'], 'fn': pq_total_result['fn']
        }
    else:
        table.add_row('Total:\n{} valid panoptic categories.'.format(
                        pq_total_result['n']),
                  percentage_to_string(pq_total_result['pq']), 
                  percentage_to_string(pq_total_result['sq']), 
                  percentage_to_string(pq_total_result['rq']))
        json_result['panoptic']['total'] = {
            'PQ': pq_total_result['pq'] * 100, 'SQ': pq_total_result['sq'] * 100, 'RQ': pq_total_result['rq'] * 100
        }
    console.print(table)
    print_tables.append(table)

    # semantic segmentation
    semantic_total_result, semantic_per_class_result = panoptic_stat.semantic_average(categories)
    table = Table(show_lines=True, caption_justify='left')
    table.add_column('Class')
    table.add_column('S_iou')
    table.add_column('S_acc')
    table.add_column('S_iou_d')
    table.add_column('S_acc_d')

    table.title = "Semantic Evaluation"

    json_result['semantic'] = {}
    per_class_result = {}
    for category_id in categories:
        semantic = semantic_per_class_result[category_id]
        if semantic['valid']:
            table.add_row(label_mapping[category_id],
                    percentage_to_string(semantic['iou']),
                    percentage_to_string(semantic['acc']),
                    percentage_to_string(semantic['iou_d']),
                    percentage_to_string(semantic['acc_d']))
            per_class_result[label_mapping[category_id]] = {
                'S_iou': semantic['iou'] * 100, 'S_acc': semantic['acc'] * 100, 'S_iou_d': semantic['iou_d'] * 100, 'S_acc_d': semantic['acc_d'] * 100
            }
    json_result['semantic']['per_class_result'] = per_class_result

    table.add_row('Total:\n{} valid semantic categories'.format(
                    semantic_total_result['n']),
                percentage_to_string(semantic_total_result['iou']),
                percentage_to_string(semantic_total_result['acc']),
                percentage_to_string(semantic_total_result['iou_d']),
                percentage_to_string(semantic_total_result['acc_d']))
    json_result['semantic']['total'] = {
        'S_iou': semantic_total_result['iou'] * 100, 'S_acc': semantic_total_result['acc'] * 100, 
        'S_iou_d': semantic_total_result['iou_d'] * 100, 'S_acc_d': semantic_total_result['acc_d'] * 100
    }
    console.print(table)
    print_tables.append(table)

    # instance segmentation
    instance_result = panoptic_stat.instance_average()
    table = Table(show_lines=True, caption_justify='left')
    table.add_column('mCov')
    table.add_column('mWCov')
    table.add_column('mPrec')
    table.add_column('mRec')

    table.title = "Instance Evaluation"
    table.add_row(
        percentage_to_string(instance_result['mCov']),
        percentage_to_string(instance_result['mWCov']),
        percentage_to_string(instance_result['mPrec']),
        percentage_to_string(instance_result['mRec']))
    json_result['instance'] = {
        'mCov': instance_result['mCov'] * 100, 'mWCov': instance_result['mWCov'] * 100,
        'mPrec': instance_result['mPrec'] * 100, 'mRec': instance_result['mRec'] * 100
    }
    console.print(table)
    print_tables.append(table)
    return print_tables, json_result


def print_iou_acc_results(ious, accs, table_title="Direct"):
    table = Table()
    table.add_column('Class')
    table.add_column('mIoU')
    table.add_column('mAcc')
    table.title = table_title

    def percentage_to_string(iou):
        if iou is None:
            return "N/A"
        else:
            v = iou * 100
            return f"{v:.1f}"

    reduced_iou = {}
    for iou in ious:
        for key, value in iou.items():
            if key not in reduced_iou:
                reduced_iou[key] = []
            if value is None:
                continue
            reduced_iou[key].append(value)
    reduced_acc = {}
    for acc in accs:
        for key, value in acc.items():
            if key not in reduced_acc:
                reduced_acc[key] = []
            if value is None:
                continue
            reduced_acc[key].append(value)
    for key, values in reduced_iou.items():
        if key == 'total':
            continue
        mIoU = np.mean(values)
        mAcc = np.mean(reduced_acc[key])
        table.add_row(key, percentage_to_string(mIoU),
                      percentage_to_string(mAcc))

    scene_total = percentage_to_string(
        np.mean([r['total'] for r in ious if 'total' in r]))
    scene_total_acc = percentage_to_string(
        np.mean([r['total'] for r in accs if 'total' in r]))
    table.add_row('Total', scene_total, scene_total_acc)

    console = Console()
    console.print(table)
    return table


if __name__ == "__main__":
    main(read_args())
