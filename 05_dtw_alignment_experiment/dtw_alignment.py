import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
from tqdm import tqdm
import sys
import time
import psutil
import tracemalloc
import functools
import traceback
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import glob
import re

# 設置更詳細的日誌格式
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('dtw_alignment_detailed.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 配置設置
CONFIG = {
    'min_confidence_threshold': 0.7,  # 最小信心閾值
    'max_time_difference': 1000,  # 最大時間差異（毫秒）
    'batch_size': 20,  # 增加批次大小
    'pause_between_batches': 1,  # 減少批次間暫停時間
    'num_workers': max(1, multiprocessing.cpu_count() - 1),  # CPU核心數
    'memory_limit': 0.8  # 最大記憶體使用率
}

def log_performance(func_name, start_time, start_memory):
    """記錄性能指標"""
    end_time = time.time()
    process = psutil.Process()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    duration = end_time - start_time
    memory_diff = end_memory - start_memory
    
    logging.debug(f"{func_name} 執行時間: {duration:.2f} 秒")
    logging.debug(f"{func_name} 內存使用: {memory_diff:.2f} MB")

class PerformanceMonitor:
    @staticmethod
    def log_time_and_memory(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            logging.debug(f"{func.__name__} 執行時間: {end_time - start_time:.2f} 秒")
            logging.debug(f"{func.__name__} 內存使用: {end_memory - start_memory:.2f} MB")
            
            return result
        return wrapper

class FastDTWAligner:
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir
        self.teacher_pattern = "*Teacher*processed_features.npz"
        self.student_pattern = "*Student*processed_features.npz"
        self.timeout = 5  # 超時時間（秒）
        self.last_progress_time = time.time()
        
    def find_feature_files(self):
        """查找所有特徵文件並進行配對"""
        logger.info(f"在目錄 {self.feature_dir} 中搜索特徵文件")
        
        # 查找所有教師和學生的特徵文件
        teacher_files = glob.glob(os.path.join(self.feature_dir, "features", self.teacher_pattern))
        student_files = glob.glob(os.path.join(self.feature_dir, "features", self.student_pattern))
        
        logger.info(f"找到 {len(teacher_files)} 個教師特徵文件和 {len(student_files)} 個學生特徵文件")
        
        # 解析文件名以獲取課程和話語信息
        teacher_dict = {}  # (lesson, utterance) -> file_path
        student_dict = {}  # (lesson, utterance, student_id) -> file_path
        
        for file_path in teacher_files:
            match = re.search(r"Lesson(\d+)_(\w+)_Teacher_utterance(\d+)", os.path.basename(file_path))
            if match:
                lesson, teacher_name, utterance = match.groups()
                key = (lesson, utterance)
                teacher_dict[key] = file_path
                logger.debug(f"教師文件: {file_path} -> Lesson {lesson}, Utterance {utterance}")
        
        for file_path in student_files:
            match = re.search(r"Lesson(\d+)_(\w+)_Student(\d+)_utterance(\d+)", os.path.basename(file_path))
            if match:
                lesson, teacher_name, student_id, utterance = match.groups()
                key = (lesson, utterance, student_id)
                student_dict[key] = file_path
                logger.debug(f"學生文件: {file_path} -> Lesson {lesson}, Utterance {utterance}, Student {student_id}")
        
        # 配對教師和學生文件
        valid_pairs = []
        for (lesson, utterance, student_id), student_file in student_dict.items():
            teacher_key = (lesson, utterance)
            if teacher_key in teacher_dict:
                valid_pairs.append((teacher_dict[teacher_key], student_file))
                logger.debug(f"配對: Lesson {lesson}, Utterance {utterance}, Student {student_id}")
        
        # 生成配對報告
        report = {
            "total_utterances": len(teacher_dict),
            "total_students": len(set(key[2] for key in student_dict.keys())),
            "valid_pairs": len(valid_pairs),
            "invalid_pairs": len(student_dict) - len(valid_pairs),
            "missing_teacher_audio": len(student_dict) - len(valid_pairs),
            "pairs": [
                {
                    "teacher_file": os.path.basename(t),
                    "student_file": os.path.basename(s)
                }
                for t, s in valid_pairs
            ]
        }
        
        # 保存配對報告
        os.makedirs("baseline", exist_ok=True)
        report_path = os.path.join("baseline", "pairing_validation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配對報告已保存到 {report_path}")
        return valid_pairs
    
    def custom_distance(self, x, y):
        """自定義距離度量，結合歐氏距離和特徵相關性"""
        # 歐氏距離
        euclidean_dist = np.sqrt(np.sum((x - y) ** 2))
        
        # 特徵相關性（使用餘弦相似度）
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x == 0 or norm_y == 0:
            correlation = 0
        else:
            correlation = np.dot(x, y) / (norm_x * norm_y)
        
        # 結合兩種度量（給予相關性更高的權重）
        return euclidean_dist * (1 - correlation)
    
    def preprocess_features(self, features):
        """預處理特徵"""
        # 確保特徵是2D數組
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # 如果特徵是(時間, 特徵)格式，轉置為(特徵, 時間)
        if features.shape[0] == 13:  # 13個MFCC特徵
            features = features.T
        
        return features
    
    def validate_features(self, student_features, teacher_features):
        """驗證特徵是否符合要求"""
        # 預處理特徵
        student_features = self.preprocess_features(student_features)
        teacher_features = self.preprocess_features(teacher_features)
        
        # 檢查特徵維度
        if student_features.shape[1] != teacher_features.shape[1]:
            raise ValueError(f"特徵維度不匹配: 學生={student_features.shape[1]}, 教師={teacher_features.shape[1]}")
        
        # 檢查特徵長度比例
        student_len = student_features.shape[0]
        teacher_len = teacher_features.shape[0]
        length_ratio = max(student_len, teacher_len) / min(student_len, teacher_len)
        
        if length_ratio > self.max_size_diff_ratio:
            print(f"警告：特徵長度差異較大: 比例={length_ratio:.2f}, 最大允許={self.max_size_diff_ratio}")
        
        # 計算Sakoe-Chiba帶寬（基於序列長度）
        self.radius = int(min(student_len, teacher_len) * self.sakoe_chiba_radius)
        print(f"使用Sakoe-Chiba帶寬: {self.radius}")
        
        # 檢查特徵大小並降採樣
        if max(student_len, teacher_len) > self.max_feature_size:
            downsample_rate = max(student_len, teacher_len) // self.max_feature_size + 1
            student_features = student_features[::downsample_rate]
            teacher_features = teacher_features[::downsample_rate]
            print(f"特徵過大，使用降採樣率 1/{downsample_rate}")
        
        return student_features, teacher_features
    
    def apply_slope_constraint(self, path):
        """應用斜率約束"""
        slopes = np.diff(path[:, 1]) / np.diff(path[:, 0])
        valid_slopes = (slopes >= 1/self.slope_constraint) & (slopes <= self.slope_constraint)
        if not np.all(valid_slopes):
            raise ValueError(f"路徑斜率超出限制範圍: {1/self.slope_constraint:.2f} - {self.slope_constraint:.2f}")
        return path
    
    def apply_step_size_constraint(self, path):
        """應用步長約束"""
        steps = np.diff(path, axis=0)
        step_sizes = np.sqrt(np.sum(steps**2, axis=1))
        if np.any(step_sizes > self.max_step_size):
            raise ValueError(f"步長超出限制：最大允許={self.max_step_size}")
        return path
    
    def align_features(self, student_features, teacher_features):
        """使用FastDTW進行特徵對齊，並應用約束"""
        print("開始FastDTW對齊...")
        start_time = time.time()
        
        try:
            # 驗證並預處理特徵
            student_features, teacher_features = self.validate_features(student_features, teacher_features)
            
            # 執行FastDTW
            distance, path = fastdtw(student_features, teacher_features, 
                                   radius=self.radius,
                                   dist=self.distance_metric)
            
            path = np.array(path)
            
            # 應用約束
            path = self.apply_slope_constraint(path)
            path = self.apply_step_size_constraint(path)
            
            processing_time = time.time() - start_time
            
            print(f"""
FastDTW對齊完成：
路徑長度：{len(path)}
處理時間：{processing_time:.2f}秒
DTW距離：{distance:.2f}
Sakoe-Chiba帶寬：{self.radius}
斜率約束：{self.slope_constraint}
最大步長：{self.max_step_size}
""")
            
            return path, distance
            
        except Exception as e:
            print(f"FastDTW對齊失敗: {str(e)}")
            raise
    
    def evaluate_alignment(self, path, student_features, teacher_features):
        """評估對齊結果"""
        # 計算RMSE（毫秒）
        rmse = np.sqrt(mean_squared_error(path[:, 0], path[:, 1])) * 10
        
        # 計算最大偏差（毫秒）
        max_deviation = np.max(np.abs(path[:, 0] - path[:, 1])) * 10
        
        # 計算對應率（偏差<200ms視為正確對應）
        correct_alignments = np.sum(np.abs(path[:, 0] - path[:, 1]) * 10 < 200)
        correspondence_rate = (correct_alignments / len(path)) * 100
        
        # 檢查結果是否符合要求
        if rmse > self.max_rmse:
            raise ValueError(f"RMSE過大: {rmse:.2f} > {self.max_rmse}")
        
        if correspondence_rate < self.min_correspondence_rate:
            raise ValueError(f"對應率過低: {correspondence_rate:.2f}% < {self.min_correspondence_rate}%")
        
        return {
            'rmse': float(rmse),
            'max_deviation': float(max_deviation),
            'correspondence_rate': float(correspondence_rate),
            'path_length': len(path)
        }
    
    def align_files(self, student_features, teacher_features):
        """對齊兩個特徵序列"""
        # 執行FastDTW對齊
        path, distance = self.align_features(student_features, teacher_features)
        
        # 評估對齊結果
        evaluation = self.evaluate_alignment(path, student_features, teacher_features)
        evaluation['dtw_distance'] = float(distance)
        
        return path.tolist(), evaluation

    def process_file_pair(self, teacher_file, student_file):
        """處理一對教師和學生的特徵文件"""
        try:
            logger.info(f"開始處理文件對:\n教師: {os.path.basename(teacher_file)}\n學生: {os.path.basename(student_file)}")
            
            # 加載特徵
            teacher_data = np.load(teacher_file)
            student_data = np.load(student_file)
            
            # 獲取特徵數據 - 只使用MFCC特徵
            teacher_features = teacher_data['mfcc']  # 先不轉置
            student_features = student_data['mfcc']  # 先不轉置
            
            # 檢查並修正特徵維度
            def normalize_features(features, name):
                """標準化特徵維度為(frames, 39)"""
                if features.shape[1] == 39:
                    return features
                elif features.shape[0] == 39:
                    return features.T
                elif features.shape[1] == 13:
                    # 如果是13維MFCC,需要先轉置確保時間幀在第一維
                    if features.shape[0] == 13:
                        features = features.T
                    # 複製三次得到39維
                    features_39 = np.concatenate([features] * 3, axis=1)
                    return features_39
                else:
                    raise ValueError(f"{name}特徵維度不正確: {features.shape}")
            
            teacher_features = normalize_features(teacher_features, "教師")
            student_features = normalize_features(student_features, "學生")
            
            logger.debug(f"教師特徵形狀: {teacher_features.shape}")
            logger.debug(f"學生特徵形狀: {student_features.shape}")
            
            # 計算DTW距離和路徑
            start_time = time.time()
            distance, path = fastdtw(teacher_features, student_features, dist=euclidean)
            processing_time = time.time() - start_time
            
            # 計算對齊質量指標
            path_np = np.array(path)
            teacher_indices = path_np[:, 0]
            student_indices = path_np[:, 1]
            
            # 計算時間戳對齊
            teacher_timestamps = np.arange(len(teacher_features)) * 0.01  # 假設幀移為10ms
            student_timestamps = np.arange(len(student_features)) * 0.01
            
            aligned_teacher_times = teacher_timestamps[teacher_indices]
            aligned_student_times = student_timestamps[student_indices]
            
            # 計算時間差異統計
            time_differences = aligned_teacher_times - aligned_student_times
            mean_difference = np.mean(time_differences)
            std_difference = np.std(time_differences)
            
            # 生成結果報告
            result = {
                "teacher_file": os.path.basename(teacher_file),
                "student_file": os.path.basename(student_file),
                "dtw_distance": float(distance),
                "processing_time": processing_time,
                "teacher_length": len(teacher_features),
                "student_length": len(student_features),
                "mean_time_difference": float(mean_difference),
                "std_time_difference": float(std_difference),
                "alignment_path": [[int(i), int(j)] for i, j in path]  # 使用alignment_path作為鍵名
            }
            
            logger.info(f"處理完成:\nDTW距離: {distance:.2f}\n處理時間: {processing_time:.2f}秒\n平均時間差異: {mean_difference:.3f}秒\n時間差異標準差: {std_difference:.3f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"處理文件對時出錯: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return None

def process_all_files(feature_dir):
    """處理所有配對的文件"""
    try:
        # 初始化DTW對齊器
        aligner = FastDTWAligner(feature_dir)
        
        # 查找並配對特徵文件
        valid_pairs = aligner.find_feature_files()
        
        if not valid_pairs:
            logger.warning("未找到有效的文件對")
            return
        
        logger.info(f"開始處理 {len(valid_pairs)} 對文件")
        
        # 逐個處理文件對
        results = []
        for i, (teacher_file, student_file) in enumerate(valid_pairs, 1):
            logger.info(f"處理第 {i}/{len(valid_pairs)} 對文件")
            
            result = aligner.process_file_pair(teacher_file, student_file)
            if result:
                results.append(result)
                
                # 保存中間結果
                if i % 10 == 0 or i == len(valid_pairs):
                    save_results(results)
        
        # 保存最終結果
        save_results(results)
        
    except Exception as e:
        logger.error(f"處理過程中出錯: {str(e)}")
        logger.error(traceback.format_exc())

def save_results(results):
    """保存處理結果"""
    try:
        os.makedirs("baseline", exist_ok=True)
        output_file = os.path.join("baseline", "alignment_results.json")
        
        # 轉換numpy數組為列表
        serializable_results = []
        for result in results:
            if result:
                result_copy = result.copy()
                result_copy["alignment_path"] = [
                    [int(i), int(j)] for i, j in result["alignment_path"]
                ]
                serializable_results.append(result_copy)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"結果已保存到 {output_file}")
        
    except Exception as e:
        logger.error(f"保存結果時出錯: {str(e)}")
        logger.error(traceback.format_exc())

def validate_directories():
    """驗證必要的目錄結構"""
    required_dirs = [
        'preprocess_teacher_audio',
        'preprocess_student_audio',
        'baseline'
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.error(f"找不到必要的目錄：{dir_name}")
            return False
        if not any(dir_path.glob('*_features.npy')):
            logger.warning(f"目錄 {dir_name} 中沒有找到特徵文件")
            
    return True

def create_pairing_map():
    """創建並驗證配對映射，支持一個老師對應多個學生"""
    teacher_dir = Path('preprocess_teacher_audio')
    student_dir = Path('preprocess_student_audio')
    pairing_map = {}
    
    # 收集教師音頻
    logger.info("正在收集教師音頻文件...")
    teacher_files = list(teacher_dir.glob('*_features.npy'))
    for teacher_file in teacher_files:
        filename = teacher_file.stem
        parts = filename.split('_')
        lesson = '_'.join(parts[:2])
        utterance = parts[-2]
        key = (lesson, utterance)
        
        if key not in pairing_map:
            pairing_map[key] = {
                'teacher': teacher_file,
                'students': {},  # 改為字典以存儲每個學生的信息
                'status': 'pending'
            }
    
    # 收集學生音頻
    logger.info("正在收集學生音頻文件...")
    student_files = list(student_dir.glob('*_features.npy'))
    for student_file in student_files:
        filename = student_file.stem
        parts = filename.split('_')
        lesson = '_'.join(parts[:2])
        utterance = parts[-2]
        student_id = next(part for part in parts if part.startswith('Student'))
        key = (lesson, utterance)
        
        if key in pairing_map:
            if student_id not in pairing_map[key]['students']:
                pairing_map[key]['students'][student_id] = []
            pairing_map[key]['students'][student_id].append(student_file)
    
    return pairing_map

def validate_pairing(pairing_map):
    """驗證配對的有效性，考慮多個學生的情況"""
    validation_results = {
        'total_utterances': len(pairing_map),
        'total_students': 0,
        'valid_pairs': 0,
        'invalid_pairs': 0,
        'missing_student_audio': 0,
        'student_statistics': {},
        'details': []
    }
    
    # 收集所有學生ID
    all_students = set()
    for pair_info in pairing_map.values():
        all_students.update(pair_info['students'].keys())
    
    # 初始化每個學生的統計信息
    for student_id in all_students:
        validation_results['student_statistics'][student_id] = {
            'total_utterances': 0,
            'completed_utterances': 0,
            'missing_utterances': 0
        }
    
    for (lesson, utterance), pair_info in pairing_map.items():
        status = {
            'lesson': lesson,
            'utterance': utterance,
            'teacher_file': str(pair_info['teacher'].name),
            'student_count': len(pair_info['students']),
            'status': 'valid' if pair_info['students'] else 'missing_student_audio',
            'students': {}
        }
        
        # 統計每個學生的情況
        for student_id, student_files in pair_info['students'].items():
            status['students'][student_id] = {
                'files': [str(f.name) for f in student_files],
                'count': len(student_files)
            }
            
            validation_results['student_statistics'][student_id]['total_utterances'] += 1
            if student_files:
                validation_results['student_statistics'][student_id]['completed_utterances'] += 1
            else:
                validation_results['student_statistics'][student_id]['missing_utterances'] += 1
        
        if not pair_info['students']:
            validation_results['missing_student_audio'] += 1
            pair_info['status'] = 'invalid'
        else:
            validation_results['valid_pairs'] += sum(len(files) for files in pair_info['students'].values())
            pair_info['status'] = 'valid'
        
        validation_results['details'].append(status)
    
    validation_results['total_students'] = len(all_students)
    validation_results['invalid_pairs'] = validation_results['missing_student_audio']
    
    return validation_results

def save_validation_report(validation_results, baseline_dir):
    """保存驗證報告，包含學生間的比較"""
    report_path = baseline_dir / 'pairing_validation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=2)
    
    logger.info("\n配對驗證報告：")
    logger.info(f"總話語數：{validation_results['total_utterances']}")
    logger.info(f"總學生數：{validation_results['total_students']}")
    logger.info(f"有效配對數：{validation_results['valid_pairs']}")
    logger.info(f"無效配對數：{validation_results['invalid_pairs']}")
    logger.info(f"缺少學生音頻：{validation_results['missing_student_audio']}")
    
    logger.info("\n學生完成情況：")
    for student_id, stats in validation_results['student_statistics'].items():
        completion_rate = (stats['completed_utterances'] / stats['total_utterances']) * 100
        logger.info(f"\n{student_id}:")
        logger.info(f"  總話語數：{stats['total_utterances']}")
        logger.info(f"  已完成：{stats['completed_utterances']}")
        logger.info(f"  缺失：{stats['missing_utterances']}")
        logger.info(f"  完成率：{completion_rate:.1f}%")
    
    logger.info(f"\n詳細報告已保存至：{report_path}")

def generate_final_report(results, baseline_dir, total_pairs, valid_pairs):
    """生成最終報告，包含學生間的比較分析"""
    # 基本統計
    avg_rmse = np.mean([r['evaluation']['rmse'] for r in results])
    avg_max_deviation = np.mean([r['evaluation']['max_deviation'] for r in results])
    avg_correspondence_rate = np.mean([r['evaluation']['correspondence_rate'] for r in results])
    
    # 按課程和學生分組
    lessons = sorted(set(r['lesson'] for r in results))
    students = sorted(set(r['student_id'] for r in results))
    
    # 課程統計
    lesson_stats = {}
    for lesson in lessons:
        lesson_results = [r for r in results if r['lesson'] == lesson]
        lesson_stats[lesson] = {
            'total_utterances': len(lesson_results),
            'average_rmse': float(np.mean([r['evaluation']['rmse'] for r in lesson_results])),
            'average_max_deviation': float(np.mean([r['evaluation']['max_deviation'] for r in lesson_results])),
            'average_correspondence_rate': float(np.mean([r['evaluation']['correspondence_rate'] for r in lesson_results])),
            'student_comparison': {}
        }
        
        # 添加學生間的比較
        for student in students:
            student_lesson_results = [r for r in lesson_results if r['student_id'] == student]
            if student_lesson_results:
                lesson_stats[lesson]['student_comparison'][student] = {
                    'utterances': len(student_lesson_results),
                    'average_rmse': float(np.mean([r['evaluation']['rmse'] for r in student_lesson_results])),
                    'average_correspondence_rate': float(np.mean([r['evaluation']['correspondence_rate'] for r in student_lesson_results]))
                }
    
    # 學生統計
    student_stats = {}
    for student in students:
        student_results = [r for r in results if r['student_id'] == student]
        student_stats[student] = {
            'total_utterances': len(student_results),
            'average_rmse': float(np.mean([r['evaluation']['rmse'] for r in student_results])),
            'average_max_deviation': float(np.mean([r['evaluation']['max_deviation'] for r in student_results])),
            'average_correspondence_rate': float(np.mean([r['evaluation']['correspondence_rate'] for r in student_results])),
            'lesson_performance': {}
        }
        
        # 添加每個課程的表現
        for lesson in lessons:
            lesson_results = [r for r in student_results if r['lesson'] == lesson]
            if lesson_results:
                student_stats[student]['lesson_performance'][lesson] = {
                    'utterances': len(lesson_results),
                    'average_rmse': float(np.mean([r['evaluation']['rmse'] for r in lesson_results])),
                    'average_correspondence_rate': float(np.mean([r['evaluation']['correspondence_rate'] for r in lesson_results]))
                }
    
    # 學生間的比較分析
    student_comparison = {
        'best_overall': max(students, key=lambda s: student_stats[s]['average_correspondence_rate']),
        'most_consistent': min(students, key=lambda s: np.std([r['evaluation']['rmse'] for r in [r for r in results if r['student_id'] == s]])),
        'lesson_winners': {}
    }
    
    # 找出每個課程表現最好的學生
    for lesson in lessons:
        lesson_results = {s: [r for r in results if r['lesson'] == lesson and r['student_id'] == s] for s in students}
        if any(lesson_results.values()):
            best_student = max(students,
                             key=lambda s: np.mean([r['evaluation']['correspondence_rate'] for r in lesson_results[s]]) if lesson_results[s] else -float('inf'))
            student_comparison['lesson_winners'][lesson] = {
                'student': best_student,
                'average_correspondence_rate': float(np.mean([r['evaluation']['correspondence_rate'] for r in lesson_results[best_student]]))
            }
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(results),
        'total_pairs': total_pairs,
        'valid_pairs': valid_pairs,
        'average_rmse': float(avg_rmse),
        'average_max_deviation': float(avg_max_deviation),
        'average_correspondence_rate': float(avg_correspondence_rate),
        'lesson_statistics': lesson_stats,
        'student_statistics': student_stats,
        'student_comparison': student_comparison
    }
    
    # 保存報告
    with open(baseline_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 輸出報告
    logger.info("\n實驗結果總結：")
    logger.info(f"總配對數: {total_pairs}")
    logger.info(f"有效配對數: {valid_pairs}")
    logger.info(f"成功處理數: {len(results)}")
    logger.info(f"平均RMSE: {avg_rmse:.2f} ms")
    logger.info(f"平均最大偏差: {avg_max_deviation:.2f} ms")
    logger.info(f"平均對應率: {avg_correspondence_rate:.2f}%")
    
    logger.info("\n學生表現比較：")
    logger.info(f"整體最佳表現：{student_comparison['best_overall']}")
    logger.info(f"最穩定表現：{student_comparison['most_consistent']}")
    
    logger.info("\n各課程最佳表現：")
    for lesson, winner in student_comparison['lesson_winners'].items():
        logger.info(f"{lesson}: {winner['student']} (對應率: {winner['average_correspondence_rate']:.2f}%)")
    
    # 輸出每個學生的詳細統計
    logger.info("\n學生詳細統計：")
    for student, stats in student_stats.items():
        logger.info(f"\n{student}:")
        logger.info(f"  總話語數: {stats['total_utterances']}")
        logger.info(f"  平均RMSE: {stats['average_rmse']:.2f} ms")
        logger.info(f"  平均對應率: {stats['average_correspondence_rate']:.2f}%")
        logger.info("  課程表現:")
        for lesson, perf in stats['lesson_performance'].items():
            logger.info(f"    {lesson}: 對應率 {perf['average_correspondence_rate']:.2f}%")

if __name__ == "__main__":
    # 設置日誌記錄
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dtw_alignment_detailed.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 設置特徵文件目錄
    feature_dir = os.path.join("audio_feature_extraction", "04_feature_extraction_experiment")
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        
    logger.info(f"使用特徵文件目錄: {feature_dir}")
    
    # 處理所有文件
    process_all_files(feature_dir)
    
    try:
        # 設置日誌格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # 獲取當前目錄作為特徵目錄
        feature_dir = os.getcwd()
        
        # 處理所有文件
        process_all_files(feature_dir)
        
    except Exception as e:
        logging.error(f"程序執行出錯：{str(e)}\n{traceback.format_exc()}")
        sys.exit(1) 