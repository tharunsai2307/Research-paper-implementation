"""Quick multi-class pipeline test after bug fix."""
import sys
sys.path.insert(0, '.')
from src.health_monitoring.inference import HealthInferenceEngine
from src.health_monitoring.schemas import DEFAULT_OPERATIONAL_THRESHOLD, DEFAULT_CANDIDATE_THRESHOLD, DEFAULT_IOU_THRESHOLD
from pathlib import Path

MODEL = Path('outputs/training/EXP-002_imgsz512/weights/best.pt')
engine = HealthInferenceEngine(
    model_path=MODEL, imgsz=512,
    operational_threshold=DEFAULT_OPERATIONAL_THRESHOLD,
    candidate_threshold=DEFAULT_CANDIDATE_THRESHOLD,
    iou_threshold=DEFAULT_IOU_THRESHOLD, device='cpu'
)

tests = [
    ('bud rot (class 1)',          'data/processed/coconut_detection_clean/train/images/BudRot004.jpg'),
    ('stembleeding (class 4)',     'data/processed/coconut_detection_clean/train/images/StemBleeding002.jpg'),
    ('leaf rot (class 3)',         'data/processed/coconut_detection_clean/train/images/LeafRot001.jpg'),
    ('gray leaf spot (class 2)',   'data/processed/coconut_detection_clean/train/images/GrayLeafSpot001.jpg'),
    ('bud root dropping (class 0)','data/processed/coconut_detection_clean/train/images/BudRootDropping009.jpg'),
    ('healthy',                    'data/processed/coconut_detection_clean/train/images/healthy_coconut_001.jpg'),
]

print(f'Thresholds: operational={DEFAULT_OPERATIONAL_THRESHOLD}  candidate={DEFAULT_CANDIDATE_THRESHOLD}')
print('='*70)
for gt, img_str in tests:
    img = Path(img_str)
    if not img.exists():
        print(f'  SKIP {img.name}')
        continue
    rec = engine.process_image(img)
    status = rec['observation_status']
    n_a = rec.get('accepted_detection_count', 0)
    n_c = rec.get('candidate_detection_count', 0)
    dets = rec.get('accepted_detections', [])
    cands = rec.get('candidate_detections', [])
    print(f'GT={gt}  image={img.name}')
    print(f'  status={status}  accepted={n_a}  candidates={n_c}')
    for d in dets:
        cname = d['disease_class']
        conf = d['confidence']
        print(f'  [ACCEPTED]  class={cname}  conf={conf}')
    for d in cands:
        cname = d['disease_class']
        conf = d['confidence']
        print(f'  [CANDIDATE] class={cname}  conf={conf}')
    print()
