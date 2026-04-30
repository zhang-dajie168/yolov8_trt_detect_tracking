# botsort_rdk.py
import numpy as np
import cv2
from collections import deque, OrderedDict
from scipy import linalg
from scipy.optimize import linear_sum_assignment
import os
import time

# 导入TensorRT版本的OSNet
try:
    from .OSNetTensorRTInference import OSNetTensorRTReID
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from OSNetTensorRTInference import OSNetTensorRTReID

# -------------------- 基础跟踪类 --------------------
class TrackState:
    """跟踪状态枚举"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class BaseTrack:
    """基础跟踪类"""
    _count = 0

    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        BaseTrack._count = 0

# -------------------- 卡尔曼滤波器 --------------------
class KalmanFilterXYWH:
    """XYWH格式的卡尔曼滤波器"""
    
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """初始化跟踪"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """预测步骤"""
        if mean is None:
            return mean, covariance
            
        mean_state = mean.copy()
        if len(mean_state) > 4:
            mean_state[4] = 0  # vx
            mean_state[5] = 0  # vy
            mean_state[6] = 0  # vw
            mean_state[7] = 0  # vh

        # 简化预测
        mean = np.dot(mean_state, self._motion_mat.T)
        covariance = np.dot(np.dot(self._motion_mat, covariance), self._motion_mat.T)
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """更新步骤"""
        # 简化更新
        if mean is None:
            return measurement.copy(), covariance
            
        # Kalman gain
        kalman_gain = covariance @ self._update_mat.T @ np.linalg.inv(
            self._update_mat @ covariance @ self._update_mat.T + np.eye(4) * 0.1
        )
        
        # 更新
        new_mean = mean + kalman_gain @ (measurement - self._update_mat @ mean)
        new_covariance = covariance - kalman_gain @ self._update_mat @ covariance
        
        return new_mean, new_covariance

    def project(self, mean, covariance):
        """投影到测量空间"""
        mean = np.dot(self._update_mat, mean)
        covariance = np.dot(np.dot(self._update_mat, covariance), self._update_mat.T)
        return mean, covariance

    def multi_predict(self, mean, covariance):
        """批量预测"""
        if len(mean) == 0:
            return mean, covariance
        
        new_mean = []
        new_covariance = []
        for i in range(len(mean)):
            m, c = self.predict(mean[i], covariance[i])
            new_mean.append(m)
            new_covariance.append(c)
        
        return np.array(new_mean), np.array(new_covariance)

# -------------------- TensorRT OSNet ReID模型 (替换PyTorch版本) --------------------
class OSNetReIDTensorRT:
    """OSNet TensorRT ReID模型 - 高性能推理版本"""
    
    def __init__(self, engine_path, input_size=(256, 128)):
        """
        初始化TensorRT版本的OSNet ReID模型
        
        Args:
            engine_path: TensorRT引擎文件路径 (.engine)
            input_size: 输入图像尺寸 (height, width)
        """
        self.input_size = input_size
        self.height, self.width = input_size
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT引擎文件不存在: {engine_path}")
        
        print(f"加载TensorRT OSNet引擎: {engine_path}")
        
        # 创建TensorRT推理器（全局单例）
        self.reid_model = OSNetTensorRTReID(engine_path, input_size)
        self.feature_dim = self.reid_model.feature_dim
        
        print(f"✅ TensorRT OSNet模型加载成功 (特征维度: {self.feature_dim})")

    def extract_feature(self, image):
        """提取单个图像的特征向量"""
        try:
            # 检查图像有效性
            if image is None or image.size == 0:
                return np.zeros(self.feature_dim, dtype=np.float32)
            
            # 确保图像是BGR格式（OpenCV默认）
            if len(image.shape) == 2:  # 灰度图转BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # 使用TensorRT推理
            feature = self.reid_model.extract_feature(image)
            
            return feature
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)

    def extract_features_batch(self, bboxes, orig_img):
        """批量提取特征（优化版本）"""
        if len(bboxes) == 0:
            return []
        
        crops = []
        valid_indices = []
        img_h, img_w = orig_img.shape[:2]
        
        # 提取有效的检测框区域
        for i, box in enumerate(bboxes):
            # 支持不同的边界框格式
            if len(box) >= 4:
                if isinstance(box, (list, np.ndarray)):
                    x1, y1, w, h = box[:4]
                    x2, y2 = x1 + w, y1 + h
                else:
                    continue
            else:
                continue
                
            # 边界裁剪
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img_w, int(x2)), min(img_h, int(y2))
            
            if x2 > x1 and y2 > y1:
                crop = orig_img[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
                    valid_indices.append(i)
        
        if not crops:
            return [np.zeros(self.feature_dim, dtype=np.float32) for _ in range(len(bboxes))]
        
        # 批量提取特征（使用TensorRT的批处理优化）
        try:
            features = self.reid_model.extract_features_batch(crops)
        except Exception as e:
            print(f"批量特征提取失败，降级到单张处理: {e}")
            # 降级到单张处理
            features = []
            for crop in crops:
                feat = self.extract_feature(crop)
                features.append(feat)
        
        # 按照原始顺序组织特征
        result_features = [np.zeros(self.feature_dim, dtype=np.float32) for _ in range(len(bboxes))]
        for idx, feat in zip(valid_indices, features):
            result_features[idx] = feat
        
        return result_features
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'reid_model'):
            del self.reid_model

# -------------------- 跟踪轨迹类 --------------------
class BOTrack(BaseTrack):
    """BoT-SORT轨迹类"""
    
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, xywh, score, cls, feat=None, feat_history=50):
        super().__init__()
        self._tlwh = np.asarray(xywh[:4], dtype=np.float32)
        self.kalman_filter = KalmanFilterXYWH()
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[4] if len(xywh) > 4 else -1

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        """更新特征"""
        if feat is None:
            return
            
        # 确保特征已归一化
        feat_norm = np.linalg.norm(feat)
        if feat_norm > 0:
            feat = feat / feat_norm
            
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            # 重新归一化
            smooth_norm = np.linalg.norm(self.smooth_feat)
            if smooth_norm > 0:
                self.smooth_feat = self.smooth_feat / smooth_norm
                
        self.features.append(feat)

    def predict(self):
        """预测"""
        if self.mean is None:
            return
            
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            if len(mean_state) > 4:
                mean_state[4] = 0  # vx
                mean_state[5] = 0  # vy
                mean_state[6] = 0  # vw
                mean_state[7] = 0  # vh

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def activate(self, kalman_filter, frame_id):
        """激活轨迹"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def update(self, new_track, frame_id):
        """更新轨迹"""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls

    def re_activate(self, new_track, frame_id, new_id=False):
        """重新激活轨迹"""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls

    @property
    def tlwh(self):
        """获取tlwh格式的边界框"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2  # xywh -> tlwh
        return ret

    @property
    def xyxy(self):
        """获取xyxy格式的边界框"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """tlwh转xywh"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        """批量预测"""
        if len(stracks) <= 0:
            return
            
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                if len(multi_mean[i]) > 4:
                    multi_mean[i][4] = 0  # vx
                    multi_mean[i][5] = 0  # vy
                    multi_mean[i][6] = 0  # vw
                    multi_mean[i][7] = 0  # vh

        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

# -------------------- 匹配工具 --------------------
def linear_assignment(cost_matrix, thresh):
    """线性分配"""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = [[row_ind[i], col_ind[i]] for i in range(len(row_ind)) if cost_matrix[row_ind[i], col_ind[i]] <= thresh]
    
    unmatched_a = list(set(range(cost_matrix.shape[0])) - set([m[0] for m in matches]))
    unmatched_b = list(set(range(cost_matrix.shape[1])) - set([m[1] for m in matches]))
    
    return matches, unmatched_a, unmatched_b

def iou_distance(atracks, btracks):
    """IoU距离计算"""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    atlbrs = [track.tlwh for track in atracks]
    btlbrs = [track.tlwh for track in btracks]
    
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    
    for i, a in enumerate(atlbrs):
        for j, b in enumerate(btlbrs):
            # 计算交集
            inter_x1 = max(a[0], b[0])
            inter_y1 = max(a[1], b[1])
            inter_x2 = min(a[0] + a[2], b[0] + b[2])
            inter_y2 = min(a[1] + a[3], b[1] + b[3])
            
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            
            # 计算并集
            a_area = a[2] * a[3]
            b_area = b[2] * b[3]
            union_area = a_area + b_area - inter_area
            
            if union_area > 0:
                ious[i, j] = inter_area / union_area
    
    return 1 - ious  # 返回成本矩阵

def embedding_distance(tracks, detections, metric='cosine'):
    """特征距离计算"""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    # 收集有效特征
    track_feats = []
    track_indices = []
    for i, track in enumerate(tracks):
        if track.smooth_feat is not None:
            track_feats.append(track.smooth_feat)
            track_indices.append(i)
    
    det_feats = []
    det_indices = []
    for i, det in enumerate(detections):
        if det.curr_feat is not None:
            det_feats.append(det.curr_feat)
            det_indices.append(i)
    
    if len(track_feats) == 0 or len(det_feats) == 0:
        return np.ones((len(tracks), len(detections)), dtype=np.float32)
    
    track_feats = np.array(track_feats)
    det_feats = np.array(det_feats)
    
    # 余弦距离
    if metric == 'cosine':
        similarity = np.dot(track_feats, det_feats.T)
        distances = 1 - similarity
        distances = np.clip(distances, 0, 1)  # 限制范围
    else:
        # 欧氏距离
        distances = np.zeros((len(track_feats), len(det_feats)))
        for i, t_feat in enumerate(track_feats):
            for j, d_feat in enumerate(det_feats):
                distances[i, j] = np.linalg.norm(t_feat - d_feat)
    
    # 构建完整距离矩阵
    full_distances = np.ones((len(tracks), len(detections)), dtype=np.float32)
    for i, ti in enumerate(track_indices):
        for j, di in enumerate(det_indices):
            full_distances[ti, di] = distances[i, j]
    
    return full_distances

def fuse_score(cost_matrix, detections):
    """融合分数"""
    if cost_matrix.size == 0:
        return cost_matrix
        
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim

# -------------------- 主跟踪器 --------------------
class BOTSORT:
    """BoT-SORT跟踪器（TensorRT加速版）"""
    
    def __init__(self, args=None, frame_rate=30):
        # 默认参数
        self.args = {
            'track_high_thresh': 0.25,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.25,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'fuse_score': True,
            'gmc_method': 'sparseOptFlow',
            'proximity_thresh': 0.5,
            'appearance_thresh': 0.8,
            'with_reid': False,  # 默认关闭，需要手动开启
            'reid_engine_path': '/home/wheeltec/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/models/osnet_x0_25.engine',
            'reid_input_size': (256, 128)
        }
        
        # 更新用户参数
        if args is not None:
            if isinstance(args, dict):
                self.args.update(args)
            else:
                for key in self.args.keys():
                    if hasattr(args, key):
                        self.args[key] = getattr(args, key)
        
        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * self.args['track_buffer'])
        self.kalman_filter = KalmanFilterXYWH()
        
        # 初始化TensorRT ReID模型（全局单例）
        self.encoder = None
        if self.args['with_reid']:
            try:
                self.encoder = OSNetReIDTensorRT(
                    engine_path=self.args['reid_engine_path'],
                    input_size=self.args['reid_input_size']
                )
                print("✅ TensorRT ReID模型初始化成功")
            except Exception as e:
                print(f"❌ TensorRT ReID模型初始化失败: {e}")
                print("将继续运行但不使用ReID特征")
                self.args['with_reid'] = False
        
        self.reset()

    def reset(self):
        """重置跟踪器"""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        BaseTrack.reset_id()

    def init_track(self, detections, img=None):
        """初始化跟踪"""
        if len(detections) == 0:
            return []
        
        bboxes = []
        scores = []
        classes = []
        
        # 解析检测结果
        for det in detections:
            if len(det) >= 5:  # xywh + score
                bboxes.append(det[:4])
                scores.append(det[4])
                classes.append(det[5] if len(det) > 5 else 0)
        
        if len(bboxes) == 0:
            return []
        
        # 提取ReID特征（使用TensorRT）
        features = []
        if self.args['with_reid'] and self.encoder is not None and img is not None:
            features = self.encoder.extract_features_batch(bboxes, img)
        
        # 创建跟踪对象
        tracks = []
        for i, (bbox, score, cls) in enumerate(zip(bboxes, scores, classes)):
            feat = features[i] if i < len(features) else None
            tracks.append(BOTrack(bbox, score, cls, feat))
        
        return tracks

    def get_dists(self, tracks, detections):
        """获取距离矩阵"""
        dists = iou_distance(tracks, detections)
        dists_mask = dists > (1 - self.args['proximity_thresh'])

        if self.args['fuse_score']:
            dists = fuse_score(dists, detections)

        if self.args['with_reid'] and self.encoder is not None:
            emb_dists = embedding_distance(tracks, detections)
            emb_dists[emb_dists > (1 - self.args['appearance_thresh'])] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
            
        return dists

    def update(self, detections, img=None):
        """更新跟踪"""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # 初始化检测
        detections = self.init_track(detections, img)
        
        # 分离高置信度和低置信度检测
        scores = [det.score for det in detections]
        remain_inds = [i for i, s in enumerate(scores) if s >= self.args['track_high_thresh']]
        inds_low = [i for i, s in enumerate(scores) if s > self.args['track_low_thresh']]
        inds_high = [i for i, s in enumerate(scores) if s < self.args['track_high_thresh']]
        
        inds_second = [i for i in inds_low if i in inds_high]
        detections_second = [detections[i] for i in inds_second]
        detections = [detections[i] for i in remain_inds]

        # 第一步关联：高置信度检测
        strack_pool = self.joint_stracks(self.tracked_stracks, self.lost_stracks)
        BOTrack.multi_predict(strack_pool)
        
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, self.args['match_thresh'])

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 第二步关联：低置信度检测
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, 0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 处理未匹配的轨迹
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 初始化新轨迹
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args['new_track_thresh']:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # 更新状态
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        # 返回跟踪结果
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        results = []
        for track in output_stracks:
            x1, y1, w, h = track.tlwh
            results.append([x1, y1, w, h, track.track_id, track.score, track.cls, 
                           None, None])
        
        return results

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """合并轨迹列表"""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """轨迹列表差集"""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]
    
    def __del__(self):
        """清理资源 - 确保ReID模型被正确释放"""
        if hasattr(self, 'encoder') and self.encoder is not None:
            print("清理TensorRT ReID模型资源...")
            try:
                del self.encoder
            except:
                pass
        if hasattr(self, 'kalman_filter'):
            self.kalman_filter = None