import os
import json
import numpy as np
import scipy.sparse
import yaml
from datasets.imdb import imdb
from model.utils.config import cfg
from PIL import Image

class tower(imdb):
    def __init__(self, mode, data_path):
        imdb.__init__(self, 'tower_' + mode)
        self._mode = mode
        self._data_path = data_path
        self._frames_path = os.path.join(data_path, 'video_clips/')
        self._annotation_path = os.path.join(data_path, 'annotation_clips/')
        
        # Load classes
        self._classes = ['__background__']
        self._class_to_ind = {self._classes[0]: 0}
        with open('//dataloader/attributes_settings.yaml', 'r') as f:
            attributes_settings = yaml.safe_load(f)
        self._classes += list(attributes_settings['classes'].keys())[2:]
        for idx, cls in enumerate(self._classes[1:], start=1):
            self._class_to_ind[cls] = idx

        # Load video and frame information
        self._video_list = []
        self._frame_list = []
        with open(os.path.join(data_path, 'trainval_split.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        for scene_data in data[mode]:
            if not scene_data['clips']:
                continue
            self._video_list.append(str(scene_data['scene']))
            for clip in scene_data['clips']:
                self._frame_list.append('MVI_' + str(scene_data['scene']) + '#' + clip + '.json')

        self._roidb_handler = self.gt_roidb

    def image_path_at(self, video_name, frame_name, frame_file_name):
        """
        Return the absolute path to a specific frame in the video sequence.
        """
        if len(str(frame_name)) == 1:
            frame_name = '000' + str(frame_name)
        if len(str(frame_name)) == 2:
            frame_name = '00' + str(frame_name)
        if len(str(frame_name)) == 3:
            frame_name = '0' + str(frame_name)
        if len(str(frame_name)) == 4:
            frame_name = str(frame_name)
        return os.path.join(self._frames_path, frame_file_name + '/img0' + frame_name + '_' + video_name + '.jpg')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        """
        gt_roidb = []
        for frame_names in self._frame_list:
            annotation_file = os.path.join(self._annotation_path, frame_names)
            if not os.path.exists(annotation_file):
                continue
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            for one_annotation in annotation:
                ann = self._load_tower_annotation(one_annotation, one_annotation['video_name'], one_annotation['frame'], frame_names[:-5])
                gt_roidb.append(ann)
        return gt_roidb

    def _load_tower_annotation(self, annotation, video_name, frame_name, frame_file_name):
        """
        Load bounding boxes and class information from the annotation file.
        """
        width, height = self._get_size(video_name, frame_name, frame_file_name)
        boxes = np.zeros((len(annotation['id']), 4), dtype=np.uint16)
        gt_classes = np.zeros((len(annotation['id']),), dtype=np.int32)
        overlaps = np.zeros((len(annotation['id']), self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((len(annotation['id']),), dtype=np.float32)

        for ix, obj in enumerate(annotation['category_id']):
            x1, y1, w, h = annotation['bbox'][ix]
            cls = annotation['category_id'][ix]
            boxes[ix, :] = [x1, y1, x1 + w, y1 + h]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (w + 1) * (h + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'image': self.image_path_at(video_name, frame_name, frame_file_name),
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'width': width,
                'height': height,
                'flipped': False,
                'seg_areas': seg_areas,
                'need_crop': True}

    def _get_size(self, video_name, frame_name, frame_file_name):
        """
        Get the size of the image.
        """
        image_file = self.image_path_at(video_name, frame_name, frame_file_name)
        with Image.open(image_file) as img:
            return img.width, img.height
        




if __name__ == '__main__':
    import yaml
    data_path = '/home/Users/lcx/datasets/wasting/'
    mode = 'train'  # or 'test'
    tower_dataset = tower(mode, data_path)
    
    # Example usage
    print("Number of videos:", len(tower_dataset._video_list))
    print("Number of frames in first video:", len(tower_dataset._frame_list[0]))
    print("First frame path:", tower_dataset.image_path_at(tower_dataset._video_list[0], tower_dataset._frame_list[0][0]))